""" Llama-3-8B TP build with lm_head vocab parallel """

import os
import sys

import torch
from torch.export import Dim, export
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.transform.insert_ccl import InsertAllReduce, InsertLogitsAllGather
from tvm.support import cc


USAGE = f"usage: python3 {os.path.basename(__file__)} <TP>"
assert len(sys.argv) == 2, USAGE
TP = int(sys.argv[1])
SO_PATH = f"./llama3_8B_TP{TP}.so"
META_PATH = f"./llama3_8B_TP{TP}.meta.json"

MODEL_DIR = "./llama3_8B/model"
TOKENIZER_DIR = "./llama3_8B/tokenizer"
WEIGHTS_DIR = "./llama3_8B/weights"
MAX_NEW_TOKENS = 512
prompt = "The capital of France is"


model_config = AutoConfig.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)


# --- Tensor-Parallelism --------
assert model_config.num_attention_heads % TP == 0, model_config.num_attention_heads
assert model_config.num_key_value_heads % TP == 0, model_config.num_key_value_heads
assert model_config.intermediate_size % TP == 0, model_config.intermediate_size
assert model_config.vocab_size % TP == 0, model_config.vocab_size

FULL_VOCAB = model_config.vocab_size
# config.json 沒有 head_dim，AutoConfig 在這裡就把它算成 hidden_size // num_attention_heads
# 並存成欄位，所以下面改 num_attention_heads 不會動到它
HEAD_DIM = model_config.head_dim
assert HEAD_DIM == model_config.hidden_size // model_config.num_attention_heads

model_config.num_attention_heads = model_config.num_attention_heads // TP
model_config.num_key_value_heads = model_config.num_key_value_heads // TP
model_config.intermediate_size = model_config.intermediate_size // TP
model_config.vocab_size = model_config.vocab_size // TP
assert model_config.head_dim == HEAD_DIM
# --- Tensor-Parallelism --------


model = AutoModelForCausalLM.from_config(model_config, dtype=torch.float32)
# vocab_size 同時決定 embed_tokens 的列數與 lm_head 的輸出維度，
# 只有 lm_head 要切，所以 embed_tokens 換回完整尺寸
model.model.embed_tokens = torch.nn.Embedding(FULL_VOCAB, model_config.hidden_size)
model.eval()

from runDisco.HF_DymicCache_pytree import register_dynamic_cache
register_dynamic_cache()


NUM_LAYERS = model_config.num_hidden_layers
NUM_KV_HEADS = model_config.num_key_value_heads
PAST = 4


from transformers.cache_utils import DynamicCache
KVcache = DynamicCache(ddp_cache_data=[
    (torch.zeros(1, NUM_KV_HEADS, PAST, HEAD_DIM),
     torch.zeros(1, NUM_KV_HEADS, PAST, HEAD_DIM))
    for _ in range(NUM_LAYERS)
])

from transformers.masking_utils import create_causal_mask
attn_mask = create_causal_mask(
    config=model.config,
    inputs_embeds=torch.zeros(1, 1, model_config.hidden_size, dtype=torch.float32),
    attention_mask=None,
    past_key_values=KVcache,
    position_ids=None,
    allow_is_causal_skip=False,
)

input_ids = (torch.zeros(1, 1, dtype=torch.long),)

example_kwargs = {
    "attention_mask": attn_mask,
    "position_ids": torch.zeros(1, 1, dtype=torch.long),
    "past_key_values": KVcache,
}

MAX_KVCache = 1 + len(tokenizer(prompt)["input_ids"]) + MAX_NEW_TOKENS
KVCache_len = Dim("KVCache_len", min=1, max=MAX_KVCache)
dynamic_shapes = {
    "input_ids": None,
    "attention_mask": {3: KVCache_len + 1},
    "position_ids": None,
    "past_key_values": [{2: KVCache_len} for _ in range(2 * NUM_LAYERS)],
}
with torch.no_grad():
    torch_model = export(model, input_ids, example_kwargs, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(torch_model, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)


LM_HEAD = "p_lm_head_weight"

Shard_axis = {}
for layer_index in range(NUM_LAYERS):
    self_attn = f"p_model_layers_{layer_index}_self_attn"
    for qkv in ("q", "k", "v"):
        Shard_axis[f"{self_attn}_{qkv}_proj_weight"] = 0
    Shard_axis[f"{self_attn}_o_proj_weight"] = 1

    mlp = f"p_model_layers_{layer_index}_mlp"
    for gate_or_up in ("gate", "up"):
        Shard_axis[f"{mlp}_{gate_or_up}_proj_weight"] = 0
    Shard_axis[f"{mlp}_down_proj_weight"] = 1

Shard_axis[LM_HEAD] = 0


ccl_passes = [
    InsertAllReduce(shard_axis=Shard_axis),
    InsertLogitsAllGather([LM_HEAD], num_workers=TP),
]

check = mod
for insert_ccl in ccl_passes:
    check = insert_ccl(check)
script = check["main"].script()
n_allreduce = script.count("ccl.allreduce")
n_allgather = script.count("ccl.allgather")
assert n_allreduce == 2 * NUM_LAYERS, n_allreduce
assert n_allgather == 1, n_allgather


seq = tvm.transform.Sequential(ccl_passes + [
    relax.transform.FuseTransposeMatmul(),
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FoldConstant(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
    relax.transform.DeadCodeElimination(),
])
mod = seq(mod)


TARGET = tvm.target.Target({
    "kind": "llvm",
    "mtriple": "aarch64-linux-gnu",
    "mcpu": "cortex-a76",
})
CROSS_CC = "aarch64-linux-gnu-gcc"

with TARGET:
    ex = tvm.compile(mod, TARGET)
ex.export_library(SO_PATH, fcompile=cc.cross_compiler(CROSS_CC))
print(f"exported {SO_PATH}")


import json
NUM_INPUT = int(mod["main"].attrs["num_input"])
WEIGHT_NAMES = [parameter.name for parameter in mod["main"].params[NUM_INPUT:]]

json.dump({
    "num_input": NUM_INPUT,
    "weight_names": WEIGHT_NAMES,
    "shard_axis": Shard_axis,
    "mask_dtype": str(attn_mask.numpy().dtype),
    "num_layers": NUM_LAYERS,
    "num_kv_heads": NUM_KV_HEADS,
    "head_dim": HEAD_DIM,
    "max_past": MAX_KVCache,
    "vocab_size": FULL_VOCAB,
    "vocab_shard": FULL_VOCAB // TP,
    "tp": TP,
}, open(META_PATH, "w"), indent=2)

print(f"wrote {META_PATH}: num_input={NUM_INPUT} weights={len(WEIGHT_NAMES)}")
