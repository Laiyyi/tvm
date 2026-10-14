import contextlib

from safetensors import safe_open

from runDisco import Checkpoint

MIB = 2 ** 20
MS = 1e6

CCL_OP_NAMES = {0: "allreduce", 1: "allgather", 2: "broadcast", 3: "scatter", 4: "gather"}
LINK_DIRECTION_NAMES = {0: "send", 1: "recv"}
KIB = 2 ** 10

ITERATIONS, BYTES, READ_NANOS, WRITE_NANOS = 0, 1, 2, 3
WRITE_CALLS, PARTIAL_WRITES, MIN_BYTES, MAX_BYTES = 4, 5, 6, 7
FIRST_BUCKET = 8


def weight_layout(checkpoint_path, weight_names, shard_axis_table, tp):
    location_of = Checkpoint.key_map(checkpoint_path)
    unmapped = [name for name in weight_names if name not in location_of]
    assert not unmapped, unmapped

    replicated_bytes = 0
    sharded_bytes = 0

    with contextlib.ExitStack() as open_files:
        reader_of_path = {}
        for index, name in enumerate(weight_names):
            shard_path, safetensors_key = location_of[name]
            if shard_path not in reader_of_path:
                reader_of_path[shard_path] = open_files.enter_context(
                    safe_open(shard_path, framework="pt"))

            weight_slice = reader_of_path[shard_path].get_slice(safetensors_key)
            full_shape = weight_slice.get_shape()
            checkpoint_dtype = weight_slice.get_dtype()
            shard_axis = shard_axis_table.get(name)

            worker_shape = list(full_shape)
            if shard_axis is None:
                placement = "replicated"
            else:
                shard_axis = int(shard_axis)
                worker_shape[shard_axis] = full_shape[shard_axis] // tp
                placement = f"sharded axis{shard_axis}"

            runtime_bytes_per_element = 4
            worker_bytes = runtime_bytes_per_element
            for dim in worker_shape:
                worker_bytes *= dim
            if shard_axis is None:
                replicated_bytes += worker_bytes
            else:
                sharded_bytes += worker_bytes

            print(f"  [{index + 1:3d}/{len(weight_names)}] {name:52s} "
                  f"{checkpoint_dtype:5s} {str(tuple(full_shape)):18s} "
                  f"{placement:16s} -> {str(tuple(worker_shape)):18s} "
                  f"{worker_bytes / MIB:8.2f} MiB/worker", flush=True)

    print(f"  {len(weight_names)} weights, "
          f"{(replicated_bytes + sharded_bytes) / MIB:.0f} MiB/worker "
          f"(replicated {replicated_bytes / MIB:.0f} + sharded {sharded_bytes / MIB:.0f})",
          flush=True)


def local_memory():
    import psutil
    info = psutil.Process().memory_full_info()
    return info.rss, info.uss


def single_process_report(elapsed_seconds, generated_tokens, memory_snapshot):
    rss, uss = memory_snapshot
    print(f"\n[1] end-to-end  {generated_tokens} 個新 token  {elapsed_seconds:.2f} s  "
          f"{elapsed_seconds / max(generated_tokens, 1) * 1e3:.1f} ms/token")
    print(f"      沒有 collective，這個時間全是計算加上 host 端的 numpy")

    print(f"\n[2] memory")
    print(f"      rss {rss / MIB:9.1f} MiB")
    print(f"      uss {uss / MIB:9.1f} MiB")


class RunReport:

    def __init__(self, session, num_nodes, workers_per_node):
        self.session = session
        self.num_nodes = num_nodes
        self.workers_per_node = workers_per_node
        self.tp = num_nodes * workers_per_node

        self.ccl_records = self._resolve("runtime.disco.ccl_timer.records")
        self.link_records = self._resolve("runtime.disco.ccl_timer.link_records")
        if self.ccl_records is None or self.link_records is None:
            print("ccl_timer doesn't exist, rebuild tvm_runtime_extra", flush=True)
            self.link_stats_width = 0
        else:
            self.link_stats_width = int(
                session.get_global_func("runtime.disco.ccl_timer.link_stats_width")()
                .debug_get_from_remote(0))

        session.import_python_module("runDisco.Memory")
        self.worker_memory = session.get_global_func("runDisco.Memory.usage")

    def _resolve(self, name):
        try:
            func = self.session.get_global_func(name)
            func()
            return func
        except Exception:
            return None

    def reset(self):
        if self.ccl_records is None:
            return
        self.session.get_global_func("runtime.disco.ccl_timer.reset")()

    def read_memory(self):
        handle = self.worker_memory()
        return [tuple(handle.debug_get_from_remote(worker)) for worker in range(self.tp)]

    def _read_triples(self, handle, worker):
        flat = list(handle().debug_get_from_remote(worker))
        return [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]

    def print_all(self, elapsed_seconds, memory_snapshot, generated_tokens):
        elapsed_nanos = elapsed_seconds * 1e9
        per_op, ccl_busiest_worker_nanos = self._ccl_summary()
        proxy_per_node = self._proxy_summary()
        has_proxy = any(stats[ITERATIONS] for node in proxy_per_node for stats in node)
        tcp_send_nanos = max((node[0][WRITE_NANOS] for node in proxy_per_node), default=0)
        tcp_recv_nanos = max((node[1][READ_NANOS] for node in proxy_per_node), default=0)

        def share(nanos):
            return f"{nanos / elapsed_nanos * 100:5.1f}%"

        print(f"\n[1] end-to-end  {generated_tokens} 個新 token  {elapsed_seconds:.2f} s  "
              f"{elapsed_seconds / max(generated_tokens, 1) * 1e3:.1f} ms/token")

        print(f"\n[2] ccl")
        total_calls = 0
        total_span_nanos = 0
        for op_name, call_count, span_nanos in per_op:
            total_calls += call_count
            total_span_nanos += span_nanos
            print(f"      {op_name:>10s} {call_count:7d} 次 {span_nanos / MS:10.2f} ms")
        print(f"      {'合計':>10s} {total_calls:7d} 次 {total_span_nanos / MS:10.2f} ms")

        self._print_proxy(proxy_per_node, has_proxy)

        print(f"\n[4] 佔 end-to-end")
        print(f"      {'ccl':>10s} {share(ccl_busiest_worker_nanos)}")
        if has_proxy:
            print(f"      {'tcp send':>10s} {share(tcp_send_nanos)}  proxy 執行緒，含在 ccl 之內")
            print(f"      {'tcp recv':>10s} {share(tcp_recv_nanos)}  proxy 執行緒，含在 ccl 之內")
        print(f"      {'非通訊':>10s} {share(elapsed_nanos - ccl_busiest_worker_nanos)}  ")

        print(f"\n[5] memory")
        for worker in range(self.tp):
            rss, _pss = memory_snapshot[worker]
            print(f"      worker {worker} rss {rss / MIB:9.1f} MiB")
        for node in range(self.num_nodes):
            workers = range(node * self.workers_per_node, (node + 1) * self.workers_per_node)
            node_pss = sum(memory_snapshot[worker][1] for worker in workers)
            print(f"      node {node} 共需 {node_pss / MIB:9.1f} MiB")

    def _ccl_summary(self):
        if self.ccl_records is None:
            return [], 0

        records_per_worker = [self._read_triples(self.ccl_records, worker)
                              for worker in range(self.tp)]
        ccl_busiest_worker_nanos = max(sum(nanos for _op, _bytes, nanos in records)
                                       for records in records_per_worker)

        per_op = []
        for op_id, op_name in CCL_OP_NAMES.items():
            durations_per_worker = [[nanos for op, _bytes, nanos in records if op == op_id]
                                    for records in records_per_worker]
            if not durations_per_worker[0]:
                continue
            call_count = min(len(durations) for durations in durations_per_worker)
            span_nanos = sum(max(durations[i] for durations in durations_per_worker)
                             for i in range(call_count))
            per_op.append((op_name, call_count, span_nanos))
        return per_op, ccl_busiest_worker_nanos

    def _proxy_summary(self):
        if self.link_records is None:
            return []

        width = self.link_stats_width
        per_node = []
        for node in range(self.num_nodes):
            flat = list(self.link_records().debug_get_from_remote(node * self.workers_per_node))
            per_node.append([flat[0:width], flat[width:2 * width]])
        return per_node

    def _print_proxy(self, proxy_per_node, has_proxy):
        if not proxy_per_node:
            return
        print(f"\n[3] proxy")
        if not has_proxy:
            print(f"      單機執行，collective 全走 pipe，沒有 TCP")
            return
        for node, directions in enumerate(proxy_per_node):
            for direction, stats in enumerate(directions):
                name = LINK_DIRECTION_NAMES[direction]
                iterations = stats[ITERATIONS]
                if iterations == 0:
                    print(f"      node {node}  {name}  沒有紀錄")
                    continue
                print(f"      node {node}  {name}  "
                      f"read {stats[READ_NANOS] / MS:9.1f} ms  "
                      f"write {stats[WRITE_NANOS] / MS:8.1f} ms  "
                      f"{iterations:7d} 次  {stats[BYTES] / MIB:8.1f} MiB  "
                      f"{stats[BYTES] / iterations / KIB:6.2f} KiB/次  "
                      f"partial {stats[PARTIAL_WRITES] / iterations * 100:4.1f}%")
