import torch
from transformers import OPTForCausalLM

model = OPTForCausalLM.from_pretrained(
    "facebook/opt-125m",
    torch_dtype=torch.float32,
    attn_implementation="eager",   # 讓 attention 展開成 MatMul/Softmax，比較容易分析
).eval()
model.config.use_cache = False

input_ids = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(1, 16),
    dtype=torch.long,
)

attention_mask = torch.ones_like(input_ids)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "opt-125m.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    },
    opset_version=17,
    do_constant_folding=True,
)