import threading
from dataclasses import dataclass
from enum import StrEnum

import mlx.core as mx

__all__ = [
    "CachePolicy",
    "Runtime",
    "release",
    "release_cache",
    "resolve_runtime",
    "retain",
]


class CachePolicy(StrEnum):
    CHUNK = "chunk"
    SESSION = "session"
    MANUAL = "manual"


@dataclass(frozen=True)
class Runtime:
    dtype: mx.Dtype = mx.float16
    quantization: str | None = None
    device: mx.Device | None = None
    cache_policy: CachePolicy = CachePolicy.CHUNK


def resolve_runtime(
    runtime: Runtime | None,
    *,
    dtype: mx.Dtype | None = None,
    quantization: str | None = None,
) -> Runtime:
    if runtime is not None:
        if dtype is not None or quantization is not None:
            message = (
                "Pass dtype/quantization either directly or through Runtime, not both."
            )
            raise TypeError(message)
        return runtime
    return Runtime(
        dtype=mx.float16 if dtype is None else dtype,
        quantization=quantization,
    )


def apply_device(runtime: Runtime) -> None:
    if runtime.device is not None:
        mx.set_default_device(runtime.device)


_live_lock = threading.Lock()


@dataclass
class _LiveSessions:
    count: int = 0


_live = _LiveSessions()


def retain() -> None:
    with _live_lock:
        _live.count += 1


def release(runtime: Runtime) -> None:
    with _live_lock:
        _live.count = max(0, _live.count - 1)
        last = _live.count == 0
    if last and runtime.cache_policy is not CachePolicy.MANUAL:
        release_cache()


def release_cache() -> None:
    mx.clear_cache()
