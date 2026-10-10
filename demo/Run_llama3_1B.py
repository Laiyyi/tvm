"""Llama-3.2-1B TP=8 + lm_head vocab parallel only TVM runtime。

requirement:
    llama3_1B_TP8_vocab.so
    llama3_1B_TP8_vocab.meta.json
"""

import json
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import tvm
from tvm.runtime import disco as di

SO_PATH = "./llama3_1B_TP8.so"
META_PATH = "./llama3_1B_TP8.meta.json"
TOKENIZER_DIR = "./llama32_1b/tokenizer"
WEIGHTS_DIR = "./llama32_1b/weights"


HOST = ""
PORT = 18000
NUM_NODES = 4
WORKERS_PER_NODE = 2

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 512


meta = json.load(open(META_PATH))
NUM_INPUT = meta["num_input"]
WEIGHT_NAMES = meta["weight_names"]
SHARD_AXIS = meta["shard_axis"] 
NUM_LAYERS = meta["num_layers"]
NUM_KV_HEADS = meta["num_kv_heads"]  
HEAD_DIM = meta["head_dim"]
MAX_PAST = meta["max_past"]
VOCAB_SIZE = meta["vocab_size"]    
VOCAB_SHARD = meta["vocab_shard"]
TP = meta["tp"]
MASK_DTYPE = np.dtype(meta["mask_dtype"])
MASK_IS_BOOL = MASK_DTYPE == np.bool_

# --- check
assert NUM_NODES * WORKERS_PER_NODE == TP, (NUM_NODES * WORKERS_PER_NODE, TP)
assert SHARD_AXIS.get("p_lm_head_weight") == 0, "This isn't vocab parallel"
assert VOCAB_SHARD * TP == VOCAB_SIZE, (VOCAB_SHARD, TP, VOCAB_SIZE)
# --- check done

# Add "p_" and convert "." from safetensors key into  "_"
WEIGHT_PATH = f"{WEIGHTS_DIR}/model.safetensors"
WORKER_WEIGHT_PATH = os.path.abspath(WEIGHT_PATH)
# safe_open is the safetensors Reader, only read header no tensor
# pt = torch.Tensor , while np = numpy.ndarry
with safe_open(WEIGHT_PATH, framework="pt") as safetenosors:
    relax_key = {
        "p_" + key.replace(".", "_"): key for key in safetenosors.keys()
    }
relax_key["p_lm_head_weight"] = "model.embed_tokens.weight"

unmapped_weights = [n for n in WEIGHT_NAMES if n not in relax_key]
assert not unmapped_weights, unmapped_weights


# session
dev = tvm.cpu(0)
sess = di.SocketSession(
    num_nodes=NUM_NODES,
    num_workers_per_node=WORKERS_PER_NODE,
    num_groups=1,
    host=HOST,
    port=PORT,
    build_ring=True,
)
sess.init_ccl("cpuccl")


mod_path = sess.upload_vm_module(os.path.abspath(SO_PATH))
sess._sync_all()
vm = sess.load_vm_module(mod_path)
sess.import_python_module("runDisco.WeightLoader")
print("Load vm module done...")



replicated_bytes = 0
sharded_bytes = 0
hidden_size = None
with safe_open(WEIGHT_PATH, framework="pt") as safetenosors:

    for index, name in enumerate(WEIGHT_NAMES):
        weight_slice = safetenosors.get_slice(relax_key[name])
        full_shape = weight_slice.get_shape()
        _dtype = weight_slice.get_dtype()
        shard_axis = SHARD_AXIS.get(name)
        if name == "p_model_norm_weight":
            hidden_size = full_shape[0]

        worker_shape = list(full_shape)
        if shard_axis is None:
            placement = "replicated"
        else:
            shard_axis = int(shard_axis)
            worker_shape[shard_axis] = full_shape[shard_axis] // TP
            placement = f"sharded axis{shard_axis}"

        worker_bytes = 4
        for dim in worker_shape:
            worker_bytes *= dim
        if shard_axis is None:
            replicated_bytes += worker_bytes
        else:
            sharded_bytes += worker_bytes

        print(f"  [{index + 1:3d}/{len(WEIGHT_NAMES)}] {name:52s} "
              f"{_dtype:5s} {str(tuple(full_shape)):18s} "
              f"{placement:16s} -> {str(tuple(worker_shape)):18s} "
              f"{worker_bytes / 2 ** 20:8.2f} MiB/worker", flush=True)


load_worker_weights = sess.get_global_func("runDisco.WeightLoader.load")

print("Loading weights on workers...", flush=True)
weight_array = load_worker_weights(WORKER_WEIGHT_PATH, json.dumps(meta))

tuple_getitem = sess.get_global_func("vm.builtin.tuple_getitem")
weight_refs = [tuple_getitem(weight_array, index) for index in range(len(WEIGHT_NAMES))]
print(f"  {len(weight_refs)} weights, "
      f"{(replicated_bytes + sharded_bytes) / 2 ** 30:.2f} GiB/worker "
      f"(replicated {replicated_bytes / 2 ** 30:.2f} + sharded {sharded_bytes / 2 ** 30:.2f})",
      flush=True)


# --- Prepare KV cache
# Initial length:1
# KV will passed between workers, and in DRef and it won't back to host 
# The KV cache will be passed between workers as DRef objects 
# without being transferred back to the host.
kv_drefs = [
    sess.broadcast(np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype="float32"))
    for _ in range(NUM_LAYERS * 2)
]

# logists container
logits_host = tvm.runtime.tensor(np.empty((1, 1, VOCAB_SIZE), dtype="float32"), device=dev)


# ============================================================ 生成
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
prompt_ids = tokenizer(PROMPT)["input_ids"]
generated = list(prompt_ids)
next_input = generated[0]
n_real = 0            
past = 1              

print(f"prompt: {tokenizer.decode(prompt_ids)!r} -> {len(prompt_ids)} tokens")


_need = 1 + len(prompt_ids) + MAX_NEW_TOKENS 
assert _need <= MAX_PAST, (
    f"cache 不夠 dummy 1 + prompt {len(prompt_ids)} + MAX_NEW_TOKENS {MAX_NEW_TOKENS} "
    f"= {_need} > MAX_PAST {MAX_PAST}。"
    f"把 MAX_NEW_TOKENS 降到 {MAX_PAST - 1 - len(prompt_ids)} 以下，"
    f"或在 build 端加大 MAX_PAST 重編"
)

CCL_OP_NAMES = {0: "allreduce", 1: "allgather", 2: "broadcast",
                3: "scatter", 4: "gather"}
LINK_OP_NAMES = {0: "send", 1: "recv"}


def resolve_counter(name):
    try:
        func = sess.get_global_func(name)
        func()
        return func
    except Exception:                       # pylint: disable=broad-except
        return None


def read_triples(handle, worker):
    flat = list(handle().debug_get_from_remote(worker))
    return [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]


ccl_records = resolve_counter("runtime.disco.ccl_timer.records")
link_records = resolve_counter("runtime.disco.ccl_timer.link_records")
if ccl_records is None or link_records is None:
    print("  ccl_timer 不在 worker 的 runtime 裡，節點需要重建 tvm_runtime_extra",
          flush=True)
else:
    sess.get_global_func("runtime.disco.ccl_timer.reset")()

print("generating ...", flush=True)

stop_reason = "?"
while True:
    if MASK_IS_BOOL:
        mask_np = np.ones((1, 1, 1, past + 1), dtype=bool)
        mask_np[0, 0, 0, 0] = False
    else:
        mask_np = np.zeros((1, 1, 1, past + 1), dtype=MASK_DTYPE)
        mask_np[0, 0, 0, 0] = np.finfo(MASK_DTYPE).min

    x = sess.broadcast(np.array([[next_input]], dtype="int64"))
    m = sess.broadcast(mask_np)
    p = sess.broadcast(np.array([[n_real]], dtype="int64"))

    outs = vm["main"](x, m, p, *kv_drefs, *weight_refs)

    logits_dref = tuple_getitem(outs, 0)
    kv_drefs = [tuple_getitem(outs, i) for i in range(1, 1 + 2 * NUM_LAYERS)]

    sess.copy_from_worker_0(logits_host, logits_dref)
    sess.sync_worker_0()
    logits = logits_host.numpy()


    past += 1
    n_real += 1

    if n_real < len(generated):
        next_input = generated[n_real]      
        continue

    next_token = int(np.argmax(logits[0, -1]))   # greedy
    generated.append(next_token)
    next_input = next_token
    print(tokenizer.decode([next_token]), end="", flush=True)

    if next_token in tokenizer.all_special_ids and next_token != tokenizer.bos_token_id:
        stop_reason = "EOS"
        break
    if len(generated) - len(prompt_ids) >= MAX_NEW_TOKENS:
        stop_reason = f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}"
        break
    if past + 1 >= MAX_PAST:
        stop_reason = f"cache 到上限 MAX_PAST={MAX_PAST}"
        break


print(f"\n\n[stop] {stop_reason}")
print(tokenizer.decode(generated))

US = 1e3
MS = 1e6


def percentile(sorted_values, ratio):
    return sorted_values[min(int(len(sorted_values) * ratio), len(sorted_values) - 1)]


if ccl_records is not None:
    print(f"\n[collective] 每次呼叫一筆，"
          f"{len(generated) - len(prompt_ids)} 個生成 token")
    worker_totals = []
    for worker in range(TP):
        rows = read_triples(ccl_records, worker)
        worker_totals.append(sum(row[2] for row in rows))
        print(f"  worker {worker}")
        print(f"    {'op':>10s} {'次數':>6s} {'bytes':>11s} {'總時間':>10s} "
              f"{'平均':>9s} {'中位':>9s} {'p99':>9s} {'最慢':>9s}")
        for op_id, op_name in CCL_OP_NAMES.items():
            selected = sorted(row[2] for row in rows if row[0] == op_id)
            if not selected:
                continue
            total_bytes = sum(row[1] for row in rows if row[0] == op_id)
            print(f"    {op_name:>10s} {len(selected):6d} {total_bytes:11d} "
                  f"{sum(selected) / MS:9.2f}ms {sum(selected) / len(selected) / US:8.1f}us "
                  f"{percentile(selected, 0.5) / US:8.1f}us "
                  f"{percentile(selected, 0.99) / US:8.1f}us {selected[-1] / US:8.1f}us")
    low, high = min(worker_totals), max(worker_totals)
    print(f"  worker 之間 最少 {low / MS:.2f}ms  最多 {high / MS:.2f}ms")
    print(f"    min     ≈ 純傳輸（最晚到的 worker 等最少）  {low / MS:9.2f}ms")
    print(f"    max-min ≈ 負載不均的等待                   {(high - low) / MS:9.2f}ms")

if link_records is not None:
    print(f"\n[tcp] 每個 node 的 TCP 邊界。worker 1/3/5/7 是獨立行程，沒有記錄是正常的")
    for worker in range(TP):
        rows = read_triples(link_records, worker)
        if not rows:
            continue
        print(f"  worker {worker}")
        for op_id, op_name in LINK_OP_NAMES.items():
            selected = [row for row in rows if row[0] == op_id]
            if not selected:
                continue
            nanos = sum(row[2] for row in selected)
            nbytes = sum(row[1] for row in selected)
            line = (f"    {op_name:>4s} {len(selected):6d} 次 {nbytes / 2 ** 20:9.2f}MiB "
                    f"{nanos / MS:9.2f}ms")
            if op_id == 0 and nanos > 0:
                line += f"  有效頻寬 {nbytes / (nanos / 1e9) / 2 ** 20:8.1f} MiB/s"
            print(line)
        send_nanos = sum(row[2] for row in rows if row[0] == 0)
        if ccl_records is not None and low > 0:
            print(f"    寫 TCP 占 collective {send_nanos / low * 100:.1f}% "
                  f"-> 高=頻寬受限，低=延遲/不均受限")
    print("    send = 寫進 socket，接近真實傳送成本")
    print("    recv = 阻塞在 socket 讀，大部分是等對方，不是傳輸")

sess.shutdown()
