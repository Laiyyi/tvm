"""Llama-3.2-1B TP? + lm_head vocab parallel only TVM runtime, allreduce decomposed into rs + allgather。

requirement:
    llama3_1B_TP?_decompose.so
    llama3_1B_TP?_decompose.meta.json
"""

import json
import os
import sys
import time

import numpy as np
from transformers import AutoTokenizer


import tvm
from tvm.runtime import disco as di


TOKENIZER_DIR = "./llama32_1b/tokenizer"
WEIGHTS_DIR = "./llama32_1b/weights"

HOST = ""
PORT = 18000

USAGE = f"usage: python3 {os.path.basename(__file__)} <TP> <NUM_NODES> <WORKERS_PER_NODE>"
assert len(sys.argv) == 4, USAGE
TP = int(sys.argv[1])
NUM_NODES = int(sys.argv[2])
WORKERS_PER_NODE = int(sys.argv[3])

SO_PATH = f"./llama3_1B_TP{TP}_decompose.so"
META_PATH = f"./llama3_1B_TP{TP}_decompose.meta.json"
for required in (SO_PATH, META_PATH):
    assert os.path.exists(required), f"{required} not found, build it first"



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
MASK_DTYPE = np.dtype(meta["mask_dtype"])
MASK_IS_BOOL = MASK_DTYPE == np.bool_

# --- check
assert meta["tp"] == TP, (f"{META_PATH} has tp={meta['tp']}", f"argv tp={TP}")
assert NUM_NODES * WORKERS_PER_NODE == TP, (NUM_NODES * WORKERS_PER_NODE, TP)
assert SHARD_AXIS.get("p_lm_head_weight") == 0, "This isn't vocab parallel"
assert VOCAB_SHARD * TP == VOCAB_SIZE, (VOCAB_SHARD, TP, VOCAB_SIZE)
# --- check done

# Add "p_" and convert "." from safetensors key into  "_"
WEIGHT_PATH = f"{WEIGHTS_DIR}/model.safetensors"
WORKER_WEIGHT_PATH = os.path.abspath(WEIGHT_PATH)


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



# workers can load module invidually:runDisco.WeightLoader
sess.import_python_module("runDisco.WeightLoader")
load_worker_weights = sess.get_global_func("runDisco.WeightLoader.load")

# loading weights
print("Loading weights on workers...", flush=True)
weight_array = load_worker_weights(WORKER_WEIGHT_PATH, json.dumps(meta))


from runDisco import Report
# RunReport imports runDisco.Memory on the workers by itself
run_report = Report.RunReport(sess, NUM_NODES, WORKERS_PER_NODE)
Report.weight_layout(WEIGHT_PATH, WEIGHT_NAMES, SHARD_AXIS, TP)

tuple_getitem = sess.get_global_func("vm.builtin.tuple_getitem")
weight_refs = [tuple_getitem(weight_array, index) for index in range(len(WEIGHT_NAMES))]


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


# Generating
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
prompt_ids = tokenizer(PROMPT)["input_ids"]
generated = list(prompt_ids)
next_input = generated[0]
n_real = 0            
past = 1              

print(f"prompt: {tokenizer.decode(prompt_ids)!r} -> {len(prompt_ids)} tokens")


_need = len(prompt_ids) + MAX_NEW_TOKENS
assert _need <= MAX_PAST, (
    f"cache not enough : prompt {len(prompt_ids)} + MAX_NEW_TOKENS {MAX_NEW_TOKENS} "
    f"= {_need} > MAX_PAST {MAX_PAST}。"
    f" MAX_NEW_TOKENS down to {MAX_PAST - len(prompt_ids)}"
    f"or max MAX_PAST and rebuild it"
)

run_report.reset()

print("generating ...", flush=True)
t_generate_begin = time.perf_counter()

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
    if past > MAX_PAST:
        stop_reason = f"cache reaches the limit: MAX_PAST={MAX_PAST}"
        break


t_generate_end = time.perf_counter()
memory_snapshot = run_report.read_memory()

print(f"\n\n[stop] {stop_reason}")
print(tokenizer.decode(generated))


generated_tokens = len(generated) - len(prompt_ids)
run_report.print_all(t_generate_end - t_generate_begin, memory_snapshot, generated_tokens)

sess.shutdown()
