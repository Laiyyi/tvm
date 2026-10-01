import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from safetensors.torch import load_file
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program



MODEL_DIR = "./llama32_1b/model"
TOKENIZER_DIR = "./llama32_1b/tokenizer"
WEIGHTS_DIR = "./llama32_1b/weights"


model_config = AutoConfig.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
weight_param_dict = load_file(f"{WEIGHTS_DIR}/model.safetensors", device="cpu")
# tie_word_embeddings=True, lm_head.weight 相等
weight_param_dict["lm_head.weight"] = weight_param_dict["model.embed_tokens.weight"]


# # --- Tensor-Parallelism
TP = 8
# assert 條件, 錯誤訊息
assert model_config.num_attention_heads % TP == 0, model_config.num_attention_heads
assert model_config.num_key_value_heads % TP == 0, model_config.num_key_value_heads  # 8 % 8 == 0，TP 的上限
assert model_config.intermediate_size % TP == 0, model_config.intermediate_size

model_config.num_attention_heads = model_config.num_attention_heads // TP
model_config.num_key_value_heads = model_config.num_key_value_heads // TP
model_config.intermediate_size = model_config.intermediate_size // TP
# # --- Tensor-Parallelism


model = AutoModelForCausalLM.from_config(model_config, dtype=torch.float32).eval()

# # Torch 認不出 HF 的 DynamicCache
import torch.utils._pytree as pytree
from transformers.cache_utils import DynamicCache


def _flatten(cache):
    flat = []
    for layer in cache.layers:
        flat.append(layer.keys)
        flat.append(layer.values)
    return flat, len(cache.layers)


def _unflatten(values, context):
    from transformers.cache_utils import DynamicLayer

    layers = []
    for i in range(context):
        layer = DynamicLayer()
        layer.lazy_initialization(values[2 * i], values[2 * i + 1])  
        layer.keys = values[2 * i]           
        layer.values = values[2 * i + 1]
        layers.append(layer)

    cache = DynamicCache()
    cache.layers = layers
    return cache


def _flatten_with_keys(cache):
    flat, ctx = _flatten(cache)
    return [(pytree.SequenceKey(i), v) for i, v in enumerate(flat)], ctx

# 認不出，所以註冊
if DynamicCache not in pytree.SUPPORTED_NODES:      # 重複註冊會 raise ValueError
    pytree.register_pytree_node(
        DynamicCache,
        _flatten,
        _unflatten,
        serialized_type_name="transformers.cache_utils.DynamicCache",
        flatten_with_keys_fn=_flatten_with_keys,
    )

from torch.export import Dim
NUM_LAYERS   = model_config.num_hidden_layers        
NUM_KV_HEADS = model_config.num_key_value_heads     
HEAD_DIM     = model_config.head_dim         
PAST         = 4                                    
MAX_PAST     = 512


past = DynamicCache(ddp_cache_data=[
    (torch.zeros(1, NUM_KV_HEADS, PAST, HEAD_DIM),
     torch.zeros(1, NUM_KV_HEADS, PAST, HEAD_DIM))
    for _ in range(NUM_LAYERS)
])

from transformers.masking_utils import create_causal_mask
attn_mask = create_causal_mask(
    config=model.config,
    inputs_embeds=torch.zeros(1, 1, model_config.hidden_size, dtype=torch.float32),
    attention_mask=None,
    past_key_values=past, 
    position_ids=None,
    allow_is_causal_skip=False,
)


input_ids = (torch.zeros(1, 1, dtype=torch.long),)


example_kwargs = {
    "attention_mask": attn_mask,
    "position_ids": torch.zeros(1, 1, dtype=torch.long),
    "past_key_values": past,
}


past_len = Dim("past_len", min=1, max=MAX_PAST)
dynamic_shapes = {
    "input_ids": None,
    "attention_mask": {3: past_len + 1},
    "position_ids": None,
    "past_key_values": [{2: past_len} for _ in range(2 * NUM_LAYERS)],
}


with torch.no_grad():
    torch_model = export(
        model, input_ids, example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    mod = from_exported_program(torch_model, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)



from tvm.s_tir import dlight
seq = tvm.transform.Sequential(
            [
                # We can enable cublas for further optimization
                relax.transform.FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                relax.transform.DeadCodeElimination(),          
            ]
        )

## --- insert allreduce
from tvm.relax.expr_functor import PyExprMutator, mutator

ROW_PARALLEL_RHS = {
    (model_config.num_attention_heads * HEAD_DIM, model_config.hidden_size),  
    (model_config.intermediate_size, model_config.hidden_size),            
}

def _static_shape(expr):
    ty = getattr(expr, "ty", None)
    if not isinstance(ty, relax.TensorType) or ty.shape is None:
        return None
    try:
        return tuple(int(d) for d in ty.shape.values)
    except (AttributeError, TypeError, ValueError):
        return None

@mutator
class InsertAllReduce(PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return call
        if _static_shape(call.args[1]) not in ROW_PARALLEL_RHS:
            return call
        return relax.op.ccl.allreduce(self.builder_.emit(call), "sum")

mod["main"] = InsertAllReduce(mod).visit_expr(mod["main"])
# mod.show()
## --- insert allreduce done

mod = seq(mod)
target = tvm.target.Target("llvm")
dev = tvm.cpu(0)
with target:
    ex = tvm.compile(mod, target)
ex.export_library("./llama3_1B_TP8.so")   


""" 

在 torch_model = export(model, ...) ... 就是 num_input
把權重與模型分開，所以權重不會被算在 graph 內
所以針對權重準備與載入才比較麻煩
num_input 後全為權重
所以要先算有幾個prompt,才知道權重數量

"""
NUM_INPUT = int(mod["main"].attrs["num_input"])
# ['p_model_embed_tokens_weight','p_model_layers_0_self_attn_q_proj_weight'...
# 這邊取的參數名稱是 Relax 給的
WEIGHT_NAMES = [parameter.name for parameter in mod["main"].params[NUM_INPUT:]]

# weight_param_dict["model.norm.weight"] = tensor([2.4688, 2.2812, 1.5078,  ..., 2.5156, 2.4062, 2.5000],
# dtype=torch.bfloat16)
# 這裡參數名稱是safetensors給的
# relax 與 safetensors 替參數取名不一，這裡將 safetensors 改成與 relax 一致
weight_param_dict = {
    "p_" + weight_key.replace(".", "_"): weight_value for weight_key, weight_value in weight_param_dict.items()
    }

import numpy as np

Shard_axis = {}
for layer_index in range(NUM_LAYERS):
    self_attn = f"p_model_layers_{layer_index}_self_attn"

    # column parallel → 切 out
    for qkv in ("q", "k", "v"):
        Shard_axis[f"{self_attn}_{qkv}_proj_weight"] = 0   

    # row parallel    → 切 in
    Shard_axis[f"{self_attn}_o_proj_weight"] = 1    
    mlp = f"p_model_layers_{layer_index}_mlp"

    for gate_or_up in ("gate", "up"):
        Shard_axis[f"{mlp}_{gate_or_up}_proj_weight"] = 0

    Shard_axis[f"{mlp}_down_proj_weight"] = 1


### --- Prepare for the weights stage1 done


from tvm.runtime import disco as di

# sess = di.ProcessSession(num_workers=TP, build_ring=True)
hosts = "127.0.0.1"
port = 18000
sess = di.SocketSession(num_nodes=4, num_workers_per_node=2, num_groups=1, host=hosts, port=port, build_ring=True)
sess.init_ccl("cpuccl")   
mod_path = "./llama3_1B_TP8.so"
# mod_path = sess.upload_vm_module("./llama3_1B_TP8.so")




MAX_NEW_TOKENS = 512
MASK_DTYPE = attn_mask.numpy().dtype          # 跟 export 時一致
MASK_IS_BOOL = MASK_DTYPE == np.bool_


prompt_ids = tokenizer("The capital of France is")["input_ids"]
generated = list(prompt_ids)
next_input = generated[0]
# cache 裡真實 token 的數量（不含 dummy）= 當前 token 的 position
n_real = 0

print(f"prompt: {tokenizer.decode(prompt_ids)!r} -> {len(prompt_ids)} tokens")


vm = sess.load_vm_module(mod_path) 


# Prepared shard weight (stage2)
w_drefs = []
for name in WEIGHT_NAMES:
    w = weight_param_dict[name].to(torch.float32).numpy()
    axis = Shard_axis.get(name)
    if axis is None:
        w_drefs.append(sess.broadcast(w))                    
    else:
        parts = np.split(w, TP, axis=axis)
        w_drefs.append(sess.scatter(np.ascontiguousarray(np.stack(parts, 0))))


# cache：形狀各 worker 相同、內容相同（都是 0），初始可以 broadcast
kv_drefs = []
for _ in range(NUM_LAYERS * 2):
    kv_drefs.append(sess.broadcast(np.zeros((1, NUM_KV_HEADS, 1, HEAD_DIM), dtype="float32")))

past = 1
# 取回 logits 用的容器，建一次重複用。形狀 / dtype 要跟 worker 上那個完全吻合
# （lm_head 不切 vocab，所以每個 worker 都是完整的 (1, 1, 128256) float32）
logits_host = tvm.runtime.tensor(
    np.empty((1, 1, model_config.vocab_size), dtype="float32"), device=dev
)

# vm["main"] 回傳的是「一個」DRef —— 整個 tuple 在 worker 上，Python 這側只有 handle，
# 不能切片。要拆開得在 worker 上呼叫 packed function（builtin.cc:665）。
tuple_getitem = sess.get_global_func("vm.builtin.tuple_getitem")

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

    outs = vm["main"](x, m, p, *kv_drefs, *w_drefs)

    # tuple_getitem 在 worker 上執行，回傳的還是 DRef —— cache 全程不離開 worker
    logits_dref = tuple_getitem(outs, 0)
    kv_drefs = [tuple_getitem(outs, i) for i in range(1, 1 + 2 * NUM_LAYERS)]
    past += 1

    # 只有 logits 要搬回 host。8 個 worker 的 logits 經過 all-reduce 之後內容相同，
    # 取 worker 0 的就好
    sess.copy_from_worker_0(logits_host, logits_dref)
    # copy_from_worker_0 只是「把 copy 指令排進 worker 的佇列」，不會等它完成。
    # 少了這行會讀到還沒被寫入的記憶體（1e23、4e-41 之類的垃圾值），
    # 而且不會有任何錯誤訊息。
    sess.sync_worker_0()
    logits = logits_host.numpy()

    n_real += 1

    if n_real < len(generated):
        next_input = generated[n_real]          # 還在餵 prompt，不用取樣
        continue

    # logits -> token id -> 文字
    next_token = int(np.argmax(logits[0, -1]))
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