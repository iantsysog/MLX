import mlx.core as mx


def plain_positions(start: int, length: int, dtype: mx.Dtype = mx.int32) -> mx.array:
    length = max(0, length)
    base = mx.arange(start, start + length, dtype=dtype)
    return mx.broadcast_to(base[None, None, :], (1, 3, length))
