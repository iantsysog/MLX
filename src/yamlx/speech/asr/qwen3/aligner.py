import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from yamlx.common.positions import plain_positions
from yamlx.common.results import AlignedWord
from yamlx.runtime import Runtime, apply_device, resolve_runtime
from yamlx.speech.asr.qwen3.tokenizer import Tokenizer, encode, load_tokenizer

from .protocol import build_aligner_prompt as _build_prompt
from .runtime.load import load_forced_aligner, resolve_model_path

if TYPE_CHECKING:
    from pathlib import Path

    from .model.model import Qwen3ASRModel

DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

_MAX_SNAP_RUN = 2

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
)


@dataclass
class ForcedAligner:
    model: Qwen3ASRModel
    tokenizer: Tokenizer
    runtime: Runtime
    timestamp_token_id: int
    segment_time: float

    @property
    def dtype(self) -> mx.Dtype:
        return self.runtime.dtype


def load_aligner(
    source: str | Path = DEFAULT_FORCED_ALIGNER_MODEL,
    *,
    dtype: mx.Dtype | None = None,
    quantization: str | None = None,
    cache_dir: str | Path | None = None,
    runtime: Runtime | None = None,
) -> ForcedAligner:
    config = resolve_runtime(runtime, dtype=dtype, quantization=quantization)
    apply_device(config)
    model_path = resolve_model_path(source, cache_dir=cache_dir)
    model, option = load_forced_aligner(
        source,
        dtype=config.dtype,
        quantization=config.quantization,
        cache_dir=cache_dir,
    )
    timestamp_token_id = option.timestamp_token_id
    segment_time = option.timestamp_segment_time
    if timestamp_token_id is None or segment_time is None:
        raise ValueError(
            "Forced-aligner checkpoints require timestamp_token_id and "
            "timestamp_segment_time in config.json"
        )
    return ForcedAligner(
        model=model,
        tokenizer=load_tokenizer(model_path),
        runtime=config,
        timestamp_token_id=timestamp_token_id,
        segment_time=segment_time,
    )


def align(fa: ForcedAligner, audio: np.ndarray, text: str) -> list[AlignedWord]:
    if not text.strip():
        return []
    words = tokenize_words(text)
    if not words:
        return []
    if audio.size == 0:
        raise ValueError("Cannot align text against an empty audio chunk.")

    audio_features, n_audio_tokens = fa.model.encode_waveform(audio, fa.dtype)
    prompt = _build_prompt(words, n_audio_tokens)
    token_ids = encode(fa.tokenizer, prompt)
    input_ids = mx.array([token_ids], dtype=mx.int32)
    position_ids = plain_positions(0, len(token_ids))

    logits = fa.model.forward_from_features(input_ids, audio_features, position_ids)
    predictions = np.asarray(mx.argmax(logits, axis=-1))[0]
    inputs = np.asarray(input_ids)[0]
    timestamp_ms = (
        predictions[inputs == fa.timestamp_token_id].astype(np.float32)
        * fa.segment_time
    )
    return _parse_timestamp_ms(words, timestamp_ms)


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    return unicodedata.category(ch).startswith(("L", "N"))


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _split_segment_with_cjk(segment: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    for ch in segment:
        if _is_cjk_char(ch):
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(ch)
        else:
            buffer.append(ch)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def tokenize_words(text: str) -> list[str]:
    tokens: list[str] = []
    for segment in text.split():
        cleaned = "".join(ch for ch in segment if _is_kept_char(ch))
        if cleaned:
            tokens.extend(_split_segment_with_cjk(cleaned))
    return tokens


def _lis_non_decreasing_indices(arr: list[float]) -> list[int]:
    n = len(arr)
    if n == 0:
        return []
    best = [1] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if arr[j] <= arr[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                parent[i] = j
    max_len = 0
    end = 0
    for i in range(n):
        if best[i] > max_len:
            max_len = best[i]
            end = i
    indices: list[int] = []
    cur = end
    while cur != -1:
        indices.append(cur)
        cur = parent[cur]
    indices.reverse()
    return indices


def _snap_run(
    out: list[float],
    start: int,
    stop: int,
    left_val: float | None,
    right_val: float | None,
) -> None:
    for k in range(start, stop):
        if left_val is None:
            if right_val is not None:
                out[k] = right_val
        elif right_val is None:
            out[k] = left_val
        else:
            out[k] = left_val if (k - (start - 1)) <= (stop - k) else right_val


def _interpolate_run(
    out: list[float],
    start: int,
    stop: int,
    left_val: float,
    right_val: float,
) -> None:
    count = stop - start
    step = (right_val - left_val) / (count + 1)
    for k in range(start, stop):
        out[k] = left_val + step * (k - start + 1)


def _repair_run(
    out: list[float],
    start: int,
    stop: int,
    left_val: float | None,
    right_val: float | None,
) -> None:
    if stop - start <= _MAX_SNAP_RUN:
        _snap_run(out, start, stop, left_val, right_val)
    elif left_val is not None and right_val is not None:
        _interpolate_run(out, start, stop, left_val, right_val)
    elif left_val is not None:
        for k in range(start, stop):
            out[k] = left_val
    elif right_val is not None:
        for k in range(start, stop):
            out[k] = right_val


def repair_timestamps(data: np.ndarray) -> list[int]:
    arr = [float(v) for v in data.tolist()]
    n = len(arr)
    if n == 0:
        return []
    normal = [False] * n
    for i in _lis_non_decreasing_indices(arr):
        normal[i] = True
    out = list(arr)
    i = 0
    while i < n:
        if normal[i]:
            i += 1
            continue
        j = i
        while j < n and not normal[j]:
            j += 1
        left_val = next((out[k] for k in range(i - 1, -1, -1) if normal[k]), None)
        right_val = next((out[k] for k in range(j, n) if normal[k]), None)
        _repair_run(out, i, j, left_val, right_val)
        i = j
    return [int(v) for v in out]


def _parse_timestamp_ms(
    words: list[str], timestamp_ms: np.ndarray
) -> list[AlignedWord]:
    fixed = repair_timestamps(timestamp_ms)
    out: list[AlignedWord] = []
    for i, word in enumerate(words):
        start_i = i * 2
        end_i = start_i + 1
        if end_i >= len(fixed):
            break
        out.append(
            AlignedWord(
                word=word,
                start=round(fixed[start_i] / 1000.0, 3),
                end=round(fixed[end_i] / 1000.0, 3),
            )
        )
    return out
