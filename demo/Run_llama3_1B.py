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

sess.shutdown()
