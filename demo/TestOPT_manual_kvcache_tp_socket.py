
import argparse
import enum
import os
import tempfile
import time

import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM
from tvm_ffi import Shape

import tvm
from tvm import relax, tirx
from tvm.relax.frontend.nn import Tensor as NNTensor
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache

# 手動 Tensor Parallelism + TVM 內建 PagedKVCache + SocketSession（不用 DistIR）
#
# 手刻 Megatron TP + TVM  TIRPagedKVCache
#   - main 不吃 kv tensor，改成吃一個不透明的 kv_cache Object，
#     cache 由 C++ runtime in-place 維護
#   - 代價是多一組 sequence 生命週期管理：add_sequence / begin_forward / end_forward。
# TP 切法沒變：每個 worker 建自己的 cache，num_attention_heads 傳 HEADS_PER_WORKER。
#
# python -m tvm.exec.disco_remote_socket_session <HOST> <PORT> <NUM_WORKERS_PER_NODE>
#


class RopeMode(enum.IntEnum):
    NONE = 0
    NORMAL = 1
    INLINE = 2


parser = argparse.ArgumentParser()
parser.add_argument("--num-nodes", type=int, default=4)
parser.add_argument("--num-workers-per-node", type=int, default=1)
parser.add_argument("--num-groups", type=int, default=1)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=18000)
parser.add_argument("--build-ring", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
args = parser.parse_args()


# OPT is GPT-like decoder-only Transformer
model = OPTForCausalLM.from_pretrained(
    "facebook/opt-125m",
    torch_dtype=torch.float32,
    attn_implementation="sdpa", #eager, sdpa,flash_attention_2, flash_attention_3 ...
).eval()
print("Model.eval() done...\n")

target = tvm.target.Target("llvm")
dev = tvm.cpu(0)

NUM_WORKERS = args.num_nodes * args.num_workers_per_node
CCL = "cpuccl"
MAX_CACHE_LEN = 128
PAGE_SIZE = 16  # PagedKVCache 以 page 為單位配置記憶體：128 / 16 = 8 頁
SEQ_ID = 0  # 只有一條 sequence（batch=1）

cfg = model.config
HIDDEN = cfg.hidden_size
HEADS = cfg.num_attention_heads
HEAD_DIM = HIDDEN // HEADS
FFN = cfg.ffn_dim
NUM_HIDDEN_LAYERS = cfg.num_hidden_layers
MAX_POS = cfg.max_position_embeddings + 2  # OPT 的 learned position embedding 有 offset=2
print(f"Hidden={HIDDEN}")
print(f"Heads={HEADS}")
print(f"Head_dim={HEAD_DIM}")
print(f"Ffn={FFN}")
print(f"Num_hidden_layers={NUM_HIDDEN_LAYERS}")
print(f"Max_pos={MAX_POS}")

assert HEADS % NUM_WORKERS == 0, "num_attention_heads 必整除 NUM_WORKERS 才能按 head 切"
assert FFN % NUM_WORKERS == 0, "ffn_dim 必須整除 NUM_WORKERS"
HEADS_PER_WORKER = HEADS // NUM_WORKERS
LOCAL_HIDDEN = HEADS_PER_WORKER * HEAD_DIM
LOCAL_FFN = FFN // NUM_WORKERS
print(f"Check OK...\n")

DTYPE = "float32"


def shard_rows(w: np.ndarray, rank: int) -> np.ndarray:
    """column-parallel: 按 dim0 = out_features 切  NUM_WORKERS 份...取第 rank 份。"""

    chunk = w.shape[0] // NUM_WORKERS
    return np.ascontiguousarray(w[rank * chunk : (rank + 1) * chunk])


def shard_cols(w: np.ndarray, rank: int) -> np.ndarray:
    """row-parallel: 沿著 dim1 = n_features 切 NUM_WORKERS 份...取第 rank 份。"""

    chunk = w.shape[1] // NUM_WORKERS
    return np.ascontiguousarray(w[:, rank * chunk : (rank + 1) * chunk])


def np_(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy().astype("float32")

decoder = model.model.decoder
assert decoder.final_layer_norm is not None

#  取出權重，轉 numpy
shared_weights = {
    "embed_tokens_w": np_(decoder.embed_tokens.weight),
    "embed_positions_w": np_(decoder.embed_positions.weight),
    "final_ln_w": np_(decoder.final_layer_norm.weight),
    "final_ln_b": np_(decoder.final_layer_norm.bias),
}

per_layer_shared = []
per_layer_sharded = []

for layer_id in range(NUM_HIDDEN_LAYERS):
    layer = decoder.layers[layer_id]

    per_layer_shared.append({
        "ln1_w": np_(layer.self_attn_layer_norm.weight),
        "ln1_b": np_(layer.self_attn_layer_norm.bias),

        "ln2_w": np_(layer.final_layer_norm.weight),
        "ln2_b": np_(layer.final_layer_norm.bias),

        "out_b": np_(layer.self_attn.out_proj.bias),
        "fc2_b": np_(layer.fc2.bias),
    })

    q_w, q_b = np_(layer.self_attn.q_proj.weight), np_(layer.self_attn.q_proj.bias)
    k_w, k_b = np_(layer.self_attn.k_proj.weight), np_(layer.self_attn.k_proj.bias)
    v_w, v_b = np_(layer.self_attn.v_proj.weight), np_(layer.self_attn.v_proj.bias)

    out_w = np_(layer.self_attn.out_proj.weight)
    fc1_w, fc1_b = np_(layer.fc1.weight), np_(layer.fc1.bias)
    fc2_w = np_(layer.fc2.weight)

# qkv row/column shard and split to every rank
    per_rank = {}
    for r in range(NUM_WORKERS):
        per_rank[r] = {
            "q_w": shard_rows(q_w, r), "q_b": shard_rows(q_b, r),
            "k_w": shard_rows(k_w, r), "k_b": shard_rows(k_b, r),
            "v_w": shard_rows(v_w, r), "v_b": shard_rows(v_b, r),
            "out_w": shard_cols(out_w, r),
            "fc1_w": shard_rows(fc1_w, r), "fc1_b": shard_rows(fc1_b, r),
            "fc2_w": shard_cols(fc2_w, r),
        }
    per_layer_sharded.append(per_rank)


#  BlockBuilder 手刻 一次一個 token + per-worker sharded KV cache main
def build_tp_kv_module() -> tvm.IRModule:
    bb = relax.BlockBuilder()

    # create_tir_paged_kv_cache
    with bb.function("create_tir_paged_kv_cache", params=[]):
        with bb.dataflow():
            cache = TIRPagedKVCache(
                attn_kind="mha",
                max_batch_size=1,
                max_total_seq_len=MAX_CACHE_LEN,
                prefill_chunk_size=MAX_CACHE_LEN,
                page_size=PAGE_SIZE,
                support_sliding_window=0,
                layer_partition=relax.ShapeExpr([0, NUM_HIDDEN_LAYERS]),
                num_hidden_layers=NUM_HIDDEN_LAYERS,
                # TP：每個 worker 的 cache 只存自己那幾個頭
                num_attention_heads=HEADS_PER_WORKER,
                num_key_value_heads=HEADS_PER_WORKER,
                qk_head_dim=HEAD_DIM,
                v_head_dim=HEAD_DIM,
                mla_original_qk_head_dim=0,
                mla_original_v_head_dim=0,
                rope_mode=RopeMode.NONE,  # OPT 不用 RoPE
                rope_scale=1,
                rope_theta=10000.0,
                rope_scaling={},
                rope_ext_factors=tirx.IntImm("int64", 0),
                rotary_dim=HEAD_DIM,
                dtype=DTYPE,
                target=target,
                enable_disaggregation=False,
            )
            cache_out = bb.emit_output(cache._expr)
        bb.emit_func_output(cache_out)

    # main（一次算一個 token）----
    def new_var(name, shape, dtype=DTYPE):
        return relax.Var(name, relax.TensorType(shape, dtype))

    input_ids_var = relax.Var("input_ids", relax.TensorType((1, 1), "int64"))
    cache_position_var = relax.Var("cache_position", relax.TensorType((1,), "int64"))
    embed_tokens_w = new_var("embed_tokens_w", (cfg.vocab_size, HIDDEN))
    embed_positions_w = new_var("embed_positions_w", (MAX_POS, HIDDEN))
    final_ln_w = new_var("final_ln_w", (HIDDEN,))
    final_ln_b = new_var("final_ln_b", (HIDDEN,))
    # Object 型別的參數：PagedKVCache 的 handle
    kv_cache_var = relax.Var("kv_cache", relax.ObjectType())

    layer_vars = []
    for l in range(NUM_HIDDEN_LAYERS):
        layer_vars.append({
            "ln1_w": new_var(f"ln1_w_{l}", (HIDDEN,)),
            "ln1_b": new_var(f"ln1_b_{l}", (HIDDEN,)),
            "q_w": new_var(f"q_w_{l}", (LOCAL_HIDDEN, HIDDEN)),
            "q_b": new_var(f"q_b_{l}", (LOCAL_HIDDEN,)),
            "k_w": new_var(f"k_w_{l}", (LOCAL_HIDDEN, HIDDEN)),
            "k_b": new_var(f"k_b_{l}", (LOCAL_HIDDEN,)),
            "v_w": new_var(f"v_w_{l}", (LOCAL_HIDDEN, HIDDEN)),
            "v_b": new_var(f"v_b_{l}", (LOCAL_HIDDEN,)),
            "out_w": new_var(f"out_w_{l}", (HIDDEN, LOCAL_HIDDEN)),
            "out_b": new_var(f"out_b_{l}", (HIDDEN,)),
            "ln2_w": new_var(f"ln2_w_{l}", (HIDDEN,)),
            "ln2_b": new_var(f"ln2_b_{l}", (HIDDEN,)),
            "fc1_w": new_var(f"fc1_w_{l}", (LOCAL_FFN, HIDDEN)),
            "fc1_b": new_var(f"fc1_b_{l}", (LOCAL_FFN,)),
            "fc2_w": new_var(f"fc2_w_{l}", (HIDDEN, LOCAL_FFN)),
            "fc2_b": new_var(f"fc2_b_{l}", (HIDDEN,)),
        })

    
    func_params = [input_ids_var, cache_position_var, embed_tokens_w, embed_positions_w]
    for l in range(NUM_HIDDEN_LAYERS):
        lv = layer_vars[l]
        func_params += [
            lv["ln1_w"], lv["ln1_b"],
            lv["q_w"], lv["q_b"], lv["k_w"], lv["k_b"], lv["v_w"], lv["v_b"],
            lv["out_w"], lv["out_b"],
            lv["ln2_w"], lv["ln2_b"],
            lv["fc1_w"], lv["fc1_b"], lv["fc2_w"], lv["fc2_b"],
        ]
    func_params += [final_ln_w, final_ln_b, kv_cache_var]

    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    with bb.function("main", func_params):
        with bb.dataflow():
       
            kv_cache = PagedKVCache(_expr=kv_cache_var, _name="kv_cache")

            # 只有 rank0 的 input_ids 是真的，broadcast 給所有 worker
            ids = bb.emit(relax.op.ccl.broadcast_from_worker0(input_ids_var))
            ids_flat = bb.emit(relax.op.reshape(ids, (1,)))
            tok_embed = bb.emit(relax.op.take(embed_tokens_w, ids_flat, axis=0, mode="fast"))
            tok_embed = bb.emit(relax.op.reshape(tok_embed, (1, 1, HIDDEN)))

            pos_idx = bb.emit(relax.op.add(cache_position_var, relax.const(2, "int64")))
            pos_embed = bb.emit(relax.op.take(embed_positions_w, pos_idx, axis=0, mode="fast"))
            pos_embed = bb.emit(relax.op.reshape(pos_embed, (1, 1, HIDDEN)))

            hidden = bb.emit(relax.op.add(tok_embed, pos_embed))

            def to_heads_1(x):
                # attention_with_fused_qkv 要的是 (b, s, heads, d)，不用轉成 (b, heads, s, d)
                return bb.emit(relax.op.reshape(x, (1, 1, HEADS_PER_WORKER, HEAD_DIM)))

            for l in range(NUM_HIDDEN_LAYERS):
                lv = layer_vars[l]

                ln1_out = bb.emit(relax.op.nn.layer_norm(hidden, lv["ln1_w"], lv["ln1_b"], axes=[-1]))
                q = bb.emit(relax.op.linear(ln1_out, lv["q_w"], lv["q_b"]))
                k_new = bb.emit(relax.op.linear(ln1_out, lv["k_w"], lv["k_b"]))
                v_new = bb.emit(relax.op.linear(ln1_out, lv["v_w"], lv["v_b"]))


                qkv = bb.emit(
                    relax.op.concat(
                        [to_heads_1(q), to_heads_1(k_new), to_heads_1(v_new)], axis=2
                    )
                )

                # cache 的 append causal masking 全部在這裡面做掉，不用自己刻。
                attn_nn = kv_cache.attention_with_fused_qkv(
                    layer_id=l,
                    qkv=NNTensor(_expr=qkv),
                    num_qo_heads=HEADS_PER_WORKER,
                    sm_scale=sm_scale,
                )
                attn = bb.emit(relax.op.reshape(attn_nn._expr, (1, 1, LOCAL_HIDDEN)))

                # out_proj 是 row-parallel：allreduce 加總所有 worker 的部分貢獻，
                # bias 加總後才加一次。
                out_partial = bb.emit(relax.op.linear(attn, lv["out_w"]))
                out_full = bb.emit(relax.op.ccl.allreduce(out_partial, "sum"))
                out_full = bb.emit(relax.op.add(out_full, lv["out_b"]))
                hidden = bb.emit(relax.op.add(hidden, out_full))

                ln2_out = bb.emit(relax.op.nn.layer_norm(hidden, lv["ln2_w"], lv["ln2_b"], axes=[-1]))
                fc1_out = bb.emit(relax.op.linear(ln2_out, lv["fc1_w"], lv["fc1_b"]))
                fc1_act = bb.emit(relax.op.nn.relu(fc1_out))
                fc2_partial = bb.emit(relax.op.linear(fc1_act, lv["fc2_w"]))
                fc2_full = bb.emit(relax.op.ccl.allreduce(fc2_partial, "sum"))
                fc2_full = bb.emit(relax.op.add(fc2_full, lv["fc2_b"]))
                hidden = bb.emit(relax.op.add(hidden, fc2_full))

            hidden = bb.emit(relax.op.nn.layer_norm(hidden, final_ln_w, final_ln_b, axes=[-1]))
            logits = bb.emit(relax.op.linear(hidden, embed_tokens_w))

            output = bb.emit_output(logits)
        bb.emit_func_output(output)

    return bb.get()


tp_kv_mod = build_tp_kv_module()
# tp_kv_mod.show()


lib_tp_kv = tvm.compile(tp_kv_mod, target=target)
so_path = os.path.join(tempfile.mkdtemp(), "opt125m_tp_kv_socket.so")
lib_tp_kv.export_library(so_path)


# ---- disco 
from tvm.runtime import disco as di

num_remote = args.num_nodes - 1
print("=" * 60)
print(f"NUM_WORKERS = {NUM_WORKERS}  (num_nodes={args.num_nodes} x "
      f"num_workers_per_node={args.num_workers_per_node})")
print("=" * 60, flush=True)

sess = di.SocketSession(
    args.num_nodes, args.num_workers_per_node, args.num_groups, args.host, args.port, args.build_ring
)
sess.init_ccl(CCL, *range(NUM_WORKERS))
dmod = sess.load_vm_module(so_path)
print(f"Disco Run ! SocketSession {args.num_nodes}x{args.num_workers_per_node}, ccl={CCL}")


def upload_shared(shape, dtype, value):
    d = sess.empty(shape, dtype)
    for r in range(NUM_WORKERS):
        d.debug_copy_from(r, value)
    return d


def upload_sharded(shape, dtype, value_per_rank):
    d = sess.empty(shape, dtype)
    for r in range(NUM_WORKERS):
        d.debug_copy_from(r, value_per_rank[r])
    return d


d_weights = [
    upload_shared((cfg.vocab_size, HIDDEN), DTYPE, shared_weights["embed_tokens_w"]),
    upload_shared((MAX_POS, HIDDEN), DTYPE, shared_weights["embed_positions_w"]),
]
for l in range(NUM_HIDDEN_LAYERS):
    ls, lr = per_layer_shared[l], per_layer_sharded[l]
    d_weights += [
        upload_shared((HIDDEN,), DTYPE, ls["ln1_w"]),
        upload_shared((HIDDEN,), DTYPE, ls["ln1_b"]),
        upload_sharded((LOCAL_HIDDEN, HIDDEN), DTYPE, {r: lr[r]["q_w"] for r in range(NUM_WORKERS)}),
        upload_sharded((LOCAL_HIDDEN,), DTYPE, {r: lr[r]["q_b"] for r in range(NUM_WORKERS)}),
        upload_sharded((LOCAL_HIDDEN, HIDDEN), DTYPE, {r: lr[r]["k_w"] for r in range(NUM_WORKERS)}),
        upload_sharded((LOCAL_HIDDEN,), DTYPE, {r: lr[r]["k_b"] for r in range(NUM_WORKERS)}),
        upload_sharded((LOCAL_HIDDEN, HIDDEN), DTYPE, {r: lr[r]["v_w"] for r in range(NUM_WORKERS)}),
        upload_sharded((LOCAL_HIDDEN,), DTYPE, {r: lr[r]["v_b"] for r in range(NUM_WORKERS)}),
        upload_sharded((HIDDEN, LOCAL_HIDDEN), DTYPE, {r: lr[r]["out_w"] for r in range(NUM_WORKERS)}),
        upload_shared((HIDDEN,), DTYPE, ls["out_b"]),
        upload_shared((HIDDEN,), DTYPE, ls["ln2_w"]),
        upload_shared((HIDDEN,), DTYPE, ls["ln2_b"]),
        upload_sharded((LOCAL_FFN, HIDDEN), DTYPE, {r: lr[r]["fc1_w"] for r in range(NUM_WORKERS)}),
        upload_sharded((LOCAL_FFN,), DTYPE, {r: lr[r]["fc1_b"] for r in range(NUM_WORKERS)}),
        upload_sharded((HIDDEN, LOCAL_FFN), DTYPE, {r: lr[r]["fc2_w"] for r in range(NUM_WORKERS)}),
        upload_shared((HIDDEN,), DTYPE, ls["fc2_b"]),
    ]
d_weights += [
    upload_shared((HIDDEN,), DTYPE, shared_weights["final_ln_w"]),
    upload_shared((HIDDEN,), DTYPE, shared_weights["final_ln_b"]),
]

sess._sync_all()


# PagedKVCache 的生命週期管理：cache 建一次，之後每步用 begin/end_forward 包住 main。
# 這幾個都是註冊在 runtime 的 global func，跟之前的 tuple_getitem 一樣透過 session
# 取得後直接呼叫，會自動 dispatch 到每個 worker（各自操作自己那份 cache）。
fadd_sequence = sess.get_global_func("vm.builtin.kv_state_add_sequence")
fbegin_forward = sess.get_global_func("vm.builtin.kv_state_begin_forward")
fend_forward = sess.get_global_func("vm.builtin.kv_state_end_forward")

kv_cache = dmod["create_tir_paged_kv_cache"]()
fadd_sequence(kv_cache, SEQ_ID)
sess._sync_all()
print("PagedKVCache created + sequence added", flush=True)


def tp_kv_step(token_id: int, pos: int):
    """跑一步 1 個 token 回傳 logits numpy (1,1,VOCAB)。

    cache 是 in-place 更新的，不用像手刻版那樣把 24 個 tensor 傳進傳出。
    每步前後都印時間戳、flush=True 卡住時最後一行就是卡住的那一步。
    """
    t0 = time.time()
    d_ids = sess.empty((1, 1), "int64")
    d_ids.debug_copy_from(0, np.array([[token_id]], dtype="int64"))
    d_pos = sess.empty((1,), "int64")
    for r in range(NUM_WORKERS):
        d_pos.debug_copy_from(r, np.array([pos], dtype="int64"))
    print(f"    [step pos={pos}] d_ids/d_pos uploaded ({time.time() - t0:.2f}s)", flush=True)

    t1 = time.time()
    print(f"    [step pos={pos}] begin_forward ...", flush=True)
    fbegin_forward(kv_cache, Shape([SEQ_ID]), Shape([1]))  # 這一步要 append 1 個 token

    print(f"    [step pos={pos}] calling dmod['main'] ...", flush=True)
    logits_dref = dmod["main"](d_ids, d_pos, *d_weights, kv_cache)

    fend_forward(kv_cache)
    print(f"    [step pos={pos}] main + end_forward done ({time.time() - t1:.2f}s)", flush=True)

    t2 = time.time()
    got_nd = tvm.runtime.empty((1, 1, cfg.vocab_size), "float32", device=dev)
    sess.copy_from_worker_0(got_nd, logits_dref)
    sess._sync_all()
    print(f"    [step pos={pos}] copy back done ({time.time() - t2:.2f}s), "
          f"total {time.time() - t0:.2f}s", flush=True)
    return got_nd.numpy()


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
prompt = "The capital of France is"
prompt_ids = tokenizer(prompt)["input_ids"]
MAX_NEW_TOKENS = 8
REPETITION_PENALTY = 1.3  # >1：對已經出現過的 token 打折扣，避免一直重複同一句


def sample_next(logits_row: np.ndarray, generated: list) -> int:
    """貪婪取樣 + repetition penalty。

    TP 跟 ref 兩邊一定要用同一套取樣邏輯，否則最後的逐 token 比對會因為取樣方式
    不同而失敗（那是取樣差異，不是計算錯誤）。
    """
    row = np.asarray(logits_row, dtype="float32").copy()
    for tok_id in set(generated):
        if row[tok_id] > 0:
            row[tok_id] /= REPETITION_PENALTY
        else:
            row[tok_id] *= REPETITION_PENALTY
    return int(np.argmax(row))


print(f"prompt: {prompt!r} -> {len(prompt_ids)} tokens")

generated_tp = list(prompt_ids)
print(f"[TP]  {tokenizer.decode(generated_tp)!r}")
for pos, token_id in enumerate(prompt_ids):
    logits = tp_kv_step(token_id, pos)
next_pos = len(prompt_ids)
for _ in range(MAX_NEW_TOKENS):
    if next_pos >= MAX_CACHE_LEN:
        break
    next_token = sample_next(logits[0, -1], generated_tp)
    generated_tp.append(next_token)
    print(f"[TP]  + {next_token:6d} {tokenizer.decode([next_token])!r}  -> {tokenizer.decode(generated_tp)!r}")
    if next_token == tokenizer.eos_token_id:
        break
    logits = tp_kv_step(next_token, next_pos)
    next_pos += 1

sess.shutdown()

generated_ref = list(prompt_ids)
print(f"[ref] {tokenizer.decode(generated_ref)!r}")
with torch.no_grad():
    for _ in range(MAX_NEW_TOKENS):
        logits_ref = model(input_ids=torch.tensor([generated_ref]), use_cache=False).logits
        next_token = sample_next(logits_ref[0, -1].numpy(), generated_ref)
        generated_ref.append(next_token)
        print(f"[ref] + {next_token:6d} {tokenizer.decode([next_token])!r}  -> {tokenizer.decode(generated_ref)!r}")
        if next_token == tokenizer.eos_token_id:
            break

print()
print(f"[TP]  最終生成: {tokenizer.decode(generated_tp)!r}")
print(f"[ref] 最終生成: {tokenizer.decode(generated_ref)!r}")
assert generated_tp == generated_ref, f"TP 跟單機生成的 token 不一致！\nTP : {generated_tp}\nref: {generated_ref}"
print(f"OK: TP={NUM_WORKERS}（SocketSession）+ KV cache 生成結果跟單機完全一致")
