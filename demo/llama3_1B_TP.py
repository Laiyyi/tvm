""" Llama-3.2-1B TP=8 build with lm_head vocab parallel """

import torch
from safetensors.torch import load_file
from torch.export import Dim, export
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.transform.insert_ccl import InsertAllReduce, InsertLogitsAllGather
from tvm.support import cc


MODEL_DIR = "./llama32_1b/model"
TOKENIZER_DIR = "./llama32_1b/tokenizer"
WEIGHTS_DIR = "./llama32_1b/weights"
MAX_NEW_TOKENS = 512
prompt = "The capital of France is"


model_config = AutoConfig.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
weight_param_dict = load_file(f"{WEIGHTS_DIR}/model.safetensors", device="cpu")
# tie_word_embeddings=True, lm_head.weight 相等
weight_param_dict["lm_head.weight"] = weight_param_dict["model.embed_tokens.weight"]


# --- Tensor-Parallelism --------
# The following config parameters must be evenly divisible by TP 
TP = 8
assert model_config.num_attention_heads % TP == 0, model_config.num_attention_heads
assert model_config.num_key_value_heads % TP == 0, model_config.num_key_value_heads
assert model_config.intermediate_size % TP == 0, model_config.intermediate_size
assert model_config.vocab_size % TP == 0, model_config.vocab_size

# for allgather
FULL_VOCAB = model_config.vocab_size   

model_config.num_attention_heads = model_config.num_attention_heads // TP
model_config.num_key_value_heads = model_config.num_key_value_heads // TP
model_config.intermediate_size = model_config.intermediate_size // TP
model_config.vocab_size = model_config.vocab_size // TP 
# When tie_word_embeddings=True, lm_head.weight and the input embedding weights share the same tensor.
# but now, lm_head must be sharded, its weights can no longer be tied to the input embedding weights.
model_config.tie_word_embeddings = False
# --- Tensor-Parallelism --------


model = AutoModelForCausalLM.from_config(model_config, dtype=torch.float32)
# because of tie_word_embeddings = False, so we need to construct the new one
# however, we just need the shape information
model.model.embed_tokens = torch.nn.Embedding(FULL_VOCAB, model_config.hidden_size)
model.eval() 

from runDisco.HF_DymicCache_pytree import register_dynamic_cache
# Torch cant recognize HF dynamicCache 
register_dynamic_cache()


NUM_LAYERS = model_config.num_hidden_layers
NUM_KV_HEADS = model_config.num_key_value_heads
HEAD_DIM = model_config.head_dim
# torch.export need to run a forward with true tensor
# must match Dim
PAST = 4


from transformers.cache_utils import DynamicCache
# past Key/Value cache in Transformer
KVcache = DynamicCache(ddp_cache_data=[
    (torch.zeros(1, NUM_KV_HEADS, PAST, HEAD_DIM),
     torch.zeros(1, NUM_KV_HEADS, PAST, HEAD_DIM))
    for _ in range(NUM_LAYERS)
])

# 4D must have mask
from transformers.masking_utils import create_causal_mask
attn_mask = create_causal_mask(
    config=model.config,
    inputs_embeds=torch.zeros(1, 1, model_config.hidden_size, dtype=torch.float32),
    attention_mask=None,
    past_key_values=KVcache,
    position_ids=None,
    allow_is_causal_skip=False,
)

#for decoder one token at a time
input_ids = (torch.zeros(1, 1, dtype=torch.long),)

example_kwargs = {
    "attention_mask": attn_mask,
    "position_ids": torch.zeros(1, 1, dtype=torch.long),
    "past_key_values": KVcache,
}

MAX_KVCache = 1 + len(prompt) + MAX_NEW_TOKENS
# set symbolic named
KVCache_len = Dim("KVCache_len", min=1, max=MAX_KVCache)
dynamic_shapes = {
    "input_ids": None,
    "attention_mask": {3: KVCache_len + 1},
    "position_ids": None,
    "past_key_values": [{2: KVCache_len} for _ in range(2 * NUM_LAYERS)],
}
# which dimension is dynamic，and what is their shape relationship。
with torch.no_grad():
    torch_model = export(model, input_ids, example_kwargs, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(torch_model, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)

# --- Export model in TVM done

# --- weight
# weight parameter name is named by relax
# Add "p_" and convert "." from safetensors key into  "_"
LM_HEAD = "p_lm_head_weight"

# Making dict to anntate which axis should be sharded
Shard_axis = {}
for layer_index in range(NUM_LAYERS):
    self_attn = f"p_model_layers_{layer_index}_self_attn"

    # column parallel → shard out
    for qkv in ("q", "k", "v"):
        Shard_axis[f"{self_attn}_{qkv}_proj_weight"] = 0

    # row parallel → shard in
    Shard_axis[f"{self_attn}_o_proj_weight"] = 1

    mlp = f"p_model_layers_{layer_index}_mlp"
    for gate_or_up in ("gate", "up"):
        Shard_axis[f"{mlp}_{gate_or_up}_proj_weight"] = 0
    Shard_axis[f"{mlp}_down_proj_weight"] = 1

# lm_head dimension of output is vocab so column parallel（ = axis 0 ）
Shard_axis[LM_HEAD] = 0



# --- insert collective pass 
ccl_passes = [
    InsertAllReduce(shard_axis=Shard_axis),
    InsertLogitsAllGather([LM_HEAD], num_workers=TP),
]

# --- Check the number of collective communication is right
check = mod
for p in ccl_passes:
    check = p(check)
script = check["main"].script()
n_allreduce = script.count("ccl.allreduce")
n_allgather = script.count("ccl.allgather")
print(f"allreduce={n_allreduce} (expect {2 * NUM_LAYERS})  allgather={n_allgather} (expect 1)")
# per layer：o_proj + down_proj
assert n_allreduce == 2 * NUM_LAYERS, n_allreduce
# only lm_head
assert n_allgather == 1, n_allgather
# --- Check done


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


SO_PATH = "./llama3_1B_TP8.so"
META_PATH = "./llama3_1B_TP8.meta.json"

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


# --- export json for run
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
    "vocab_size": FULL_VOCAB,       # full=128256 after all-gather
    "vocab_shard": FULL_VOCAB // TP,
    "tp": TP,
}, open(META_PATH, "w"), indent=2)

print(f"wrote {META_PATH}: num_input={NUM_INPUT} weights={len(WEIGHT_NAMES)}")
