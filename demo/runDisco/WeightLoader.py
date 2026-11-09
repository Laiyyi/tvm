import contextlib
import json

import torch
from safetensors import safe_open
from tvm_ffi import get_global_func, register_global_func

import tvm

from runDisco import Checkpoint


def _worker_rank():
    return int(get_global_func("runtime.disco.worker_rank")())


def _read_shard(weight_slice, shard_axis, shard_index, shard_length):
    begin = shard_index * shard_length
    end = begin + shard_length
    if shard_axis == 0:
        return weight_slice[begin:end]
    return weight_slice[:, begin:end]


@register_global_func("runDisco.WeightLoader.load")
def load(checkpoint_path: str, meta_json: str):
    meta = json.loads(meta_json)
    weight_names = meta["weight_names"]
    shard_axis_table = meta["shard_axis"]
    tp = meta["tp"]

    rank = _worker_rank()
    device = tvm.cpu(0)
    location_of = Checkpoint.key_map(checkpoint_path)

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
