import math

import mlx.core as mx


def split_heads(x: mx.array, heads: int) -> mx.array:
    b, seq, _ = x.shape
    d = x.shape[-1] // heads
    return x.reshape(b, seq, heads, d).transpose(0, 2, 1, 3)


def merge_heads(x: mx.array) -> mx.array:
    b, h, seq, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, seq, h * d)


def _scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    result: mx.array | None = None
    if hasattr(mx.fast, "scaled_dot_product_attention"):
        try:
            result = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask=mask
            )
        except TypeError, ValueError:
            result = None
        except RuntimeError as error:
            message = str(error).lower()
            if "not implemented" not in message and "unsupported" not in message:
                raise
            result = None

    if result is not None:
        return result

    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1] != 0:
            message = (
                f"Incompatible attention-head counts: q={q.shape[1]}, k={k.shape[1]}."
            )
            raise ValueError(message)
        groups = q.shape[1] // k.shape[1]
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores += mask
    return mx.softmax(scores, axis=-1) @ v
