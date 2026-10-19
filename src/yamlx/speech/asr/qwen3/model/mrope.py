from dataclasses import dataclass

import mlx.core as mx
import numpy as np

_NUM_SPATIAL_DIMS = 3
DEFAULT_MROPE_SECTION = (24, 20, 20)
_INTERLEAVE_STRIDE = 3
_ROPE_COS_RANK = 3


@dataclass(frozen=True)
class MRoPEState:
    head_dim: int
    base: float
    mrope_section: tuple[int, int, int]
    interleaved: bool
    inv_freq: mx.array
    overwrite_masks: list[mx.array]


def build_mrope(
    head_dim: int,
    base: float = 1000000.0,
    mrope_section: tuple[int, int, int] | None = None,
    *,
    interleaved: bool = True,
) -> MRoPEState:
    section = mrope_section or DEFAULT_MROPE_SECTION
    half_dim = head_dim // 2
    if len(section) != _NUM_SPATIAL_DIMS:
        message = (
            f"mrope_section {list(section)} must have 3 entries "
            "(temporal, height, width)."
        )
        raise ValueError(message)
    if sum(section) != half_dim:
        message = (
            f"mrope_section {list(section)} sums to {sum(section)} "
            f"but; head_dim // 2 is {half_dim}"
        )
        raise ValueError(message)

    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    mx.eval(inv_freq)

    owner = _owner_indices(section, half_dim, interleaved=interleaved)
    masks = [mx.array((owner == dim)[None, None, :]) for dim in (1, 2)]
    return MRoPEState(head_dim, base, section, interleaved, inv_freq, masks)


def apply_mrope(
    state: MRoPEState,
    position_ids: mx.array,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]:
    n_dims = position_ids.shape[1]
    if n_dims != _NUM_SPATIAL_DIMS:
        message = (
            f"position_ids must carry 3 spatial dims "
            f"(temporal, height, width); got {n_dims}."
        )
        raise ValueError(message)

    pos = position_ids.astype(mx.float32).transpose(1, 0, 2)[..., None]
    freqs = pos * state.inv_freq

    combined = freqs[0]
    for dim, mask in enumerate(state.overwrite_masks, start=1):
        combined = mx.where(mask, freqs[dim], combined)

    emb = mx.concatenate([combined, combined], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _owner_indices(
    section: tuple[int, int, int],
    half_dim: int,
    *,
    interleaved: bool,
) -> np.ndarray:
    owner = np.zeros(half_dim, dtype=np.int64)
    if interleaved:
        for dim in (1, 2):
            stop = min(section[dim] * _INTERLEAVE_STRIDE, half_dim)
            owner[dim:stop:_INTERLEAVE_STRIDE] = dim
    else:
        start = 0
        for dim, count in enumerate(section):
            owner[start : start + count] = dim
            start += count
    return owner


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    if cos.ndim == _ROPE_COS_RANK:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def _rotate_half(x: mx.array) -> mx.array:
    mid = x.shape[-1] // 2
    return mx.concatenate([-x[..., mid:], x[..., :mid]], axis=-1)
