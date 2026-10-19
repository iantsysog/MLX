lazy import mlx.core as mx


def scalar_int(value: mx.array) -> int:
    item: object = value.item()
    if isinstance(item, int | float):
        return int(item)
    message = f"Expected an integer scalar; got {item!r}."
    raise TypeError(message)
