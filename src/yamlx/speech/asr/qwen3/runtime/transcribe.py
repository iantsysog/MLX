from typing import TYPE_CHECKING

import mlx.core as mx

from yamlx.common.audio import SAMPLE_RATE
from yamlx.common.chunking import split_audio_into_chunks
from yamlx.common.generation import (
    DEFAULT_EOS_TOKEN_IDS,
    FINISH_REASON_LENGTH,
    generate_greedy,
    resolve_max_new_tokens,
)
from yamlx.common.languages import canonicalize_language, join_text_parts
from yamlx.common.positions import plain_positions
from yamlx.common.results import (
    AlignedWord,
    FinishReason,
    TranscriptionChunk,
    TranscriptionResult,
)
from yamlx.runtime import CachePolicy, release_cache
from yamlx.speech.asr.qwen3.protocol import build_prompt_tokens, parse_asr_output
from yamlx.speech.asr.qwen3.tokenizer import decode

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from yamlx.speech.asr.qwen3.model.model import Qwen3ASRModel
    from yamlx.speech.asr.qwen3.tokenizer import Tokenizer

__all__ = ["Aligner", "transcribe_audio"]

type Aligner = Callable[[np.ndarray, str], list[AlignedWord]]


def _clear_mlx_cache() -> None:
    release_cache()


def _encode_audio(
    model: Qwen3ASRModel,
    chunk: np.ndarray,
    dtype: mx.Dtype,
) -> tuple[mx.array, int]:
    if chunk.size == 0:
        raise ValueError("Cannot transcribe an empty audio chunk.")
    return model.encode_waveform(chunk, dtype)


def _prompt_arrays(prompt_tokens: list[int]) -> tuple[mx.array, mx.array]:
    return (
        mx.array([prompt_tokens], dtype=mx.int32),
        plain_positions(0, len(prompt_tokens)),
    )


def transcribe_audio(
    model: Qwen3ASRModel,
    tokenizer: Tokenizer,
    audio: np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
    language: str | None = None,
    max_new_tokens: int | None = None,
    timestamps: bool = False,
    aligner: Aligner | None = None,
    on_progress: Callable[[float], None] | None = None,
    cache_policy: CachePolicy = CachePolicy.CHUNK,
) -> TranscriptionResult:
    if timestamps and aligner is None:
        raise ValueError("timestamps=True requires an aligner.")
    if audio.size == 0:
        raise ValueError("Cannot transcribe empty audio.")

    forced = canonicalize_language(language)
    duration = float(audio.size) / SAMPLE_RATE
    pieces = split_audio_into_chunks(audio)

    result_chunks: list[TranscriptionChunk] = []
    words: list[AlignedWord] = []
    detected: str | None = forced
    any_length = False

    for chunk, offset in pieces:
        chunk_duration = float(chunk.size) / SAMPLE_RATE
        budget = resolve_max_new_tokens(
            max_new_tokens, audio_duration_sec=chunk_duration
        )
        audio_features, n_audio_tokens = _encode_audio(model, chunk, dtype)
        prompt_tokens = build_prompt_tokens(
            tokenizer, n_audio_tokens=n_audio_tokens, language=forced
        )
        input_ids, position_ids = _prompt_arrays(prompt_tokens)
        generation = generate_greedy(
            model,
            input_ids,
            audio_features,
            position_ids,
            max_new_tokens=budget,
            eos_token_ids=DEFAULT_EOS_TOKEN_IDS,
        )
        chunk_language, text = parse_asr_output(
            decode(tokenizer, generation.tokens), forced_language=forced
        )
        if detected is None:
            detected = chunk_language

        result_chunks.append(
            TranscriptionChunk(text=text, offset_seconds=float(offset))
        )
        if aligner is not None and timestamps and text:
            words.extend(
                AlignedWord(
                    word=word.word,
                    start=word.start + float(offset),
                    end=word.end + float(offset),
                )
                for word in aligner(chunk, text)
            )
        if generation.finish_reason == FINISH_REASON_LENGTH:
            any_length = True

        del audio_features, input_ids, position_ids
        if cache_policy is CachePolicy.CHUNK:
            _clear_mlx_cache()
        if on_progress is not None:
            processed_sec = offset + chunk_duration
            on_progress(min(processed_sec / duration, 1.0) if duration > 0 else 1.0)

    finish_reason = FinishReason.LENGTH if any_length else FinishReason.STOP
    return TranscriptionResult(
        text=join_text_parts((chunk.text for chunk in result_chunks), detected),
        language=detected,
        duration=duration,
        chunks=tuple(result_chunks),
        words=tuple(words) if timestamps else None,
        finish_reason=finish_reason,
    )
