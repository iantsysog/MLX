import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yamlx.common.audio import SAMPLE_RATE
from yamlx.common.scope import ResourceScope
from yamlx.runtime import Runtime, apply_device, release, resolve_runtime, retain
from yamlx.speech.asr.qwen3.tokenizer import Tokenizer, load_tokenizer

from .load import load_model, resolve_model_path
from .transcribe import transcribe_audio

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import mlx.core as mx

    from yamlx.common.results import TranscriptionResult
    from yamlx.speech.asr.qwen3.model.model import Qwen3ASRModel
    from yamlx.speech.asr.qwen3.option import Qwen3ASROption

    from .transcribe import Aligner

_WARM_UP_SECONDS = 0.5
_CLOSED_MESSAGE = "The Qwen3ASR session is closed."


@dataclass
class Qwen3ASR:
    model: Qwen3ASRModel | None
    option: Qwen3ASROption | None
    tokenizer: Tokenizer | None
    runtime: Runtime
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    closed: bool = False
    active_stream: object | None = None
    scope: ResourceScope = field(default_factory=ResourceScope, repr=False)

    @property
    def dtype(self) -> mx.Dtype:
        return self.runtime.dtype


def load_session(
    source: str | Path,
    *,
    dtype: mx.Dtype | None = None,
    quantization: str | None = None,
    cache_dir: str | Path | None = None,
    runtime: Runtime | None = None,
) -> Qwen3ASR:
    config = resolve_runtime(runtime, dtype=dtype, quantization=quantization)
    apply_device(config)
    retain()
    scope = ResourceScope()
    scope.defer(lambda: release(config))
    try:
        path = Path(resolve_model_path(source, cache_dir=cache_dir))
        model, option = load_model(
            path,
            dtype=config.dtype,
            quantization=config.quantization,
        )
        tokenizer = load_tokenizer(path)
    except BaseException:
        scope.close()
        raise
    session = Qwen3ASR(
        model=model,
        option=option,
        tokenizer=tokenizer,
        runtime=config,
        scope=scope,
    )
    scope.defer(lambda: _drop_references(session))
    return session


from_pretrained = load_session


@contextmanager
def open_session(
    source: str | Path,
    *,
    dtype: mx.Dtype | None = None,
    quantization: str | None = None,
    cache_dir: str | Path | None = None,
    runtime: Runtime | None = None,
) -> Generator[Qwen3ASR]:
    session = load_session(
        source,
        dtype=dtype,
        quantization=quantization,
        cache_dir=cache_dir,
        runtime=runtime,
    )
    try:
        yield session
    finally:
        close(session)


@contextmanager
def _open(session: Qwen3ASR) -> Generator[tuple[Qwen3ASRModel, Tokenizer]]:
    with session.lock:
        model, tokenizer = session.model, session.tokenizer
        if session.closed or model is None or tokenizer is None:
            raise RuntimeError(_CLOSED_MESSAGE)
        yield model, tokenizer


def transcribe(
    session: Qwen3ASR,
    audio: np.ndarray,
    *,
    language: str | None = None,
    max_new_tokens: int | None = None,
    timestamps: bool = False,
    aligner: Aligner | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> TranscriptionResult:
    with _open(session) as (model, tokenizer):
        return transcribe_audio(
            model,
            tokenizer,
            audio,
            dtype=session.dtype,
            language=language,
            max_new_tokens=max_new_tokens,
            timestamps=timestamps,
            aligner=aligner,
            on_progress=on_progress,
            cache_policy=session.runtime.cache_policy,
        )


def warm_up(session: Qwen3ASR) -> None:
    with _open(session) as (model, tokenizer):
        silence = np.zeros(int(_WARM_UP_SECONDS * SAMPLE_RATE), dtype=np.float32)
        transcribe_audio(
            model,
            tokenizer,
            silence,
            dtype=session.dtype,
            cache_policy=session.runtime.cache_policy,
        )


def close(session: Qwen3ASR) -> None:
    with session.lock:
        if session.closed:
            return
        session.closed = True
    session.scope.close()


def _drop_references(session: Qwen3ASR) -> None:
    with session.lock:
        session.model = None
        session.option = None
        session.tokenizer = None
