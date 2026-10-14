import contextlib
import json
import os

import torch
from safetensors import safe_open
from tvm_ffi import get_global_func, register_global_func

import tvm

INDEX_FILENAME = "model.safetensors.index.json"


def key_map(checkpoint_path):
    if checkpoint_path.endswith(".safetensors"):
        with safe_open(checkpoint_path, framework="pt") as single_shard:
            file_of_key = {key: checkpoint_path for key in single_shard.keys()}
    else:
        index = json.load(open(os.path.join(checkpoint_path, INDEX_FILENAME)))
        file_of_key = {key: os.path.join(checkpoint_path, filename)
                       for key, filename in index["weight_map"].items()}

    location_of = {"p_" + key.replace(".", "_"): (path, key)
                   for key, path in file_of_key.items()}
    # tie_word_embeddings=True 的 checkpoint 沒有獨立的 lm_head.weight
    if "p_lm_head_weight" not in location_of:
        location_of["p_lm_head_weight"] = location_of["p_model_embed_tokens_weight"]
    return location_of


def _worker_rank():
    return int(get_global_func("runtime.disco.worker_rank")())


def _read_shard(weight_slice, shard_axis, shard_index, shard_length):
    begin = shard_index * shard_length
    end = begin + shard_length
    if shard_axis == 0:
        return weight_slice[begin:end]
    return weight_slice[:, begin:end]


@register_global_func("runDisco.Checkpoint.load")
def load(checkpoint_path: str, meta_json: str):
    meta = json.loads(meta_json)
    weight_names = meta["weight_names"]
    shard_axis_table = meta["shard_axis"]
    tp = meta["tp"]

    rank = _worker_rank()
    device = tvm.cpu(0)
    location_of = key_map(checkpoint_path)

    weights = []
    with contextlib.ExitStack() as open_files:
        reader_of_path = {}
        for name in weight_names:
            shard_path, safetensors_key = location_of[name]
            if shard_path not in reader_of_path:
                reader_of_path[shard_path] = open_files.enter_context(
                    safe_open(shard_path, framework="pt"))
            reader = reader_of_path[shard_path]

            shard_axis = shard_axis_table.get(name)
            if shard_axis is None:
                part = reader.get_tensor(safetensors_key)
            else:
                shard_axis = int(shard_axis)
                weight_slice = reader.get_slice(safetensors_key)
                shard_length = weight_slice.get_shape()[shard_axis] // tp
                part = _read_shard(weight_slice, shard_axis, rank, shard_length)

            weights.append(tvm.runtime.tensor(part.to(torch.float32).numpy(), device=device))
    return weights
