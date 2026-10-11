from safetensors import safe_open

MIB = 2 ** 20
MS = 1e6

CCL_OP_NAMES = {0: "allreduce", 1: "allgather", 2: "broadcast", 3: "scatter", 4: "gather"}
LINK_SEND = 0


def safetensors_key_map(weight_path):
    with safe_open(weight_path, framework="pt") as safetensors:
        mapping = {"p_" + key.replace(".", "_"): key for key in safetensors.keys()}
    mapping["p_lm_head_weight"] = "model.embed_tokens.weight"
    return mapping


def weight_layout(weight_path, weight_names, shard_axis_table, tp):
    safetensors_key_of = safetensors_key_map(weight_path)
    unmapped = [name for name in weight_names if name not in safetensors_key_of]
    assert not unmapped, unmapped

    replicated_bytes = 0
    sharded_bytes = 0

    with safe_open(weight_path, framework="pt") as safetensors:
        for index, name in enumerate(weight_names):
            weight_slice = safetensors.get_slice(safetensors_key_of[name])
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
        tcp_send_nanos, tcp_recv_nanos = self._tcp_summary()

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

        print(f"\n[3] tcp  ")
        print(f"      {'send':>10s} {tcp_send_nanos / MS:10.2f} ms")
        print(f"      {'recv':>10s} {tcp_recv_nanos / MS:10.2f} ms")

        print(f"\n[4] 佔 end-to-end")
        print(f"      {'ccl':>10s} {share(ccl_busiest_worker_nanos)}")
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

    def _tcp_summary(self):
        if self.link_records is None:
            return 0, 0

        send_per_node = []
        recv_per_node = []
        for node in range(self.num_nodes):
            node_send = []
            node_recv = []
            for worker in range(node * self.workers_per_node,
                                (node + 1) * self.workers_per_node):
                for op, _bytes, nanos in self._read_triples(self.link_records, worker):
                    (node_send if op == LINK_SEND else node_recv).append(nanos)
            send_per_node.append(sum(node_send[1:]))
            recv_per_node.append(sum(node_recv[1:]))
        return max(send_per_node), max(recv_per_node)
