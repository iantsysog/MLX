from dataclasses import dataclass

import mlx.core as mx


@dataclass
class KVCache:
    keys: list[mx.array | None]
    values: list[mx.array | None]
    offset: int = 0


def new_kv_cache(num_layers: int) -> KVCache:
    return KVCache([None] * num_layers, [None] * num_layers)


def kv_update(
    cache: KVCache,
    key: mx.array,
    value: mx.array,
    layer_idx: int,
) -> tuple[mx.array, mx.array]:
    prev_key = cache.keys[layer_idx]
    prev_value = cache.values[layer_idx]
    if prev_key is not None and prev_value is not None:
        full_k = mx.concatenate([prev_key, key], axis=2)
        full_v = mx.concatenate([prev_value, value], axis=2)
    else:
        full_k = key
        full_v = value
    cache.keys[layer_idx] = full_k
    cache.values[layer_idx] = full_v

    if layer_idx == len(cache.keys) - 1:
        cache.offset += key.shape[2]

    return full_k, full_v


def fork_cache(cache: KVCache) -> KVCache:
    return KVCache(list(cache.keys), list(cache.values), cache.offset)


def trim_cache(cache: KVCache, count: int) -> None:
    if count <= 0:
        return
    if count >= cache.offset:
        cache.keys = [None] * len(cache.keys)
        cache.values = [None] * len(cache.values)
        cache.offset = 0
        return
    keep = cache.offset - count
    cache.keys = [None if k is None else k[:, :, :keep, :] for k in cache.keys]
    cache.values = [None if v is None else v[:, :, :keep, :] for v in cache.values]
    cache.offset = keep
