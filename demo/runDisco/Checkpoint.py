import json
import os

from safetensors import safe_open

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
    if "p_lm_head_weight" not in location_of:
        location_of["p_lm_head_weight"] = location_of["p_model_embed_tokens_weight"]
    return location_of
