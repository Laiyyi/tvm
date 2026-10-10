import json
import os

import torch
from safetensors import safe_open
from tvm_ffi import get_global_func, register_global_func

import tvm


def _worker_rank():
    return int(get_global_func("runtime.disco.worker_rank")())


def _read_shard(weight_slice, shard_axis, shard_index, shard_length):
    begin = shard_index * shard_length
    end = begin + shard_length
    if shard_axis == 0:
        return weight_slice[begin:end]
    return weight_slice[:, begin:end]


@register_global_func("runDisco.WeightLoader.load")
def load(weight_path: str, meta_json: str):
    meta = json.loads(meta_json)
    weight_names = meta["weight_names"]
    shard_axis_table = meta["shard_axis"]
    tp = meta["tp"]

    rank = _worker_rank()
    device = tvm.cpu(0)

    with safe_open(weight_path, framework="pt") as safetensors:
        checkpoint_key_of = {
            "p_" + key.replace(".", "_"): key for key in safetensors.keys()
        }
        checkpoint_key_of["p_lm_head_weight"] = "model.embed_tokens.weight"

        weights = []
        for name in weight_names:
            checkpoint_key = checkpoint_key_of[name]
            shard_axis = shard_axis_table.get(name)
            if shard_axis is None:
                part = safetensors.get_tensor(checkpoint_key)
            else:
                shard_axis = int(shard_axis)
                weight_slice = safetensors.get_slice(checkpoint_key)
                shard_length = weight_slice.get_shape()[shard_axis] // tp
                part = _read_shard(weight_slice, shard_axis, rank, shard_length)
            weights.append(
                tvm.runtime.tensor(part.to(torch.float32).numpy(), device=device)
            )

    return weights


@register_global_func("runDisco.WeightLoader.report_rank")
def report_rank():
    return _worker_rank()


@register_global_func("runDisco.WeightLoader.rss")
def rss():
    with open("/proc/self/statm") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")
