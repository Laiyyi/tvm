import torch.utils._pytree as pytree
from transformers.cache_utils import DynamicCache, DynamicLayer


# cache = DynamicCache(layers=[DynamicLayer, DynamicLayer....])
# both of keys and values are tensor([[[[0., 0.,......]]]])
def _flatten(cache):
    flat = []
    for layer in cache.layers:
        flat.append(layer.keys)
        flat.append(layer.values)
    return flat, len(cache.layers)


def _unflatten(values, context):
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


# recursive
def _flatten_with_keys(cache):
    flat, ctx = _flatten(cache)
    # set index for flat [k0, v0, k1, v1....]
    # [(SequenceKey(0),  k0),
    #  (SequenceKey(1),  v0),
    return [(pytree.SequenceKey(i), leaf) for i, leaf in enumerate(flat)], ctx


def register_dynamic_cache():
    if DynamicCache in pytree.SUPPORTED_NODES:
        return
    # how to flatten DynamicCache
    # how to reconstruct DynamicCache
    # how to call DynamicCache when it serialize
    # how to get the path to leaf
    pytree.register_pytree_node(
        DynamicCache,
        _flatten,
        _unflatten,
        serialized_type_name="transformers.cache_utils.DynamicCache",
        flatten_with_keys_fn=_flatten_with_keys,
    )
