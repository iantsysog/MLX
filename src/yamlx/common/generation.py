import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from yamlx.common.kv import KVCache
from yamlx.common.positions import plain_positions

REPETITION_THRESHOLD = 20

_MIN_REPETITION_THRESHOLD = 3

AUTO_MAX_NEW_TOKENS_FLOOR = 128

AUTO_MAX_NEW_TOKENS_CAP = 512

AUTO_MAX_NEW_TOKENS_PER_SECOND = 12.0

AUTO_MAX_NEW_TOKENS_MARGIN = 32

FINISH_REASON_EOS = "eos"
FINISH_REASON_REPETITION = "repetition"
FINISH_REASON_LENGTH = "length"

# Qwen chat-model stop tokens; other vocabularies should pass their own eos_token_ids.
DEFAULT_EOS_TOKEN_IDS: tuple[int, ...] = (151643, 151645)

type StepFn = Callable[[mx.array, mx.array, KVCache], mx.array]


class Decoder(Protocol):
    def create_cache(self) -> KVCache: ...

    def prefill(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array: ...

    def step(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array: ...


@dataclass(frozen=True)
class GenerationResult:
    tokens: list[int]
    finish_reason: str
    generated_tokens: int
    max_new_tokens: int

    @property
    def truncated(self) -> bool:
        return self.finish_reason == FINISH_REASON_LENGTH


def resolve_max_new_tokens(
    max_new_tokens: int | None,
    *,
    audio_duration_sec: float | None = None,
) -> int:
    if max_new_tokens is not None:
        if max_new_tokens < 0:
            message = f"max_new_tokens must be >= 0; got {max_new_tokens}"
            raise ValueError(message)
        return max_new_tokens
    if audio_duration_sec is None:
        return AUTO_MAX_NEW_TOKENS_FLOOR
    proportional = math.ceil(audio_duration_sec * AUTO_MAX_NEW_TOKENS_PER_SECOND)
    return min(
        max(AUTO_MAX_NEW_TOKENS_FLOOR, proportional + AUTO_MAX_NEW_TOKENS_MARGIN),
        AUTO_MAX_NEW_TOKENS_CAP,
    )


def detect_repetition(tokens: Sequence[int], threshold: int) -> bool:
    if threshold < _MIN_REPETITION_THRESHOLD or len(tokens) < threshold:
        return False
    for n in range(1, threshold // 3 + 1):
        if threshold % n != 0:
            continue
        pattern = list(tokens[-n:])
        if list(tokens[-threshold:]) == pattern * (threshold // n):
            return True
    return False


def _strip_trailing_eos(tokens: list[int], eos_token_ids: tuple[int, ...]) -> None:
    while tokens and tokens[-1] in eos_token_ids:
        tokens.pop()


def greedy_steps(
    step_fn: StepFn,
    cache: KVCache,
    logits: mx.array,
    step_positions: mx.array,
    max_new_tokens: int,
    *,
    input_dtype: mx.Dtype,
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS,
    repetition_threshold: int = REPETITION_THRESHOLD,
) -> tuple[list[int], str]:
    tokens: list[int] = []
    finish_reason = FINISH_REASON_LENGTH
    for step in range(max_new_tokens):
        next_token = int(mx.argmax(logits[0, -1, :]))
        tokens.append(next_token)

        if next_token in eos_token_ids:
            finish_reason = FINISH_REASON_EOS
            break
        if detect_repetition(tokens, repetition_threshold):
            finish_reason = FINISH_REASON_REPETITION
            break
        if step + 1 == max_new_tokens:
            break

        step_input = mx.array([[next_token]], dtype=input_dtype)
        logits = step_fn(
            step_input,
            step_positions[:, :, step : step + 1],
            cache,
        )
        mx.eval(logits)

    _strip_trailing_eos(tokens, eos_token_ids)
    return tokens, finish_reason


def generate_greedy(
    decoder: Decoder,
    input_ids: mx.array,
    audio_features: mx.array,
    position_ids: mx.array,
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS,
    repetition_threshold: int = REPETITION_THRESHOLD,
) -> GenerationResult:
    if max_new_tokens < 0:
        message = f"max_new_tokens must be >= 0; got {max_new_tokens}"
        raise ValueError(message)
    if max_new_tokens == 0:
        return GenerationResult(
            tokens=[],
            finish_reason=FINISH_REASON_LENGTH,
            generated_tokens=0,
            max_new_tokens=0,
        )

    cache = decoder.create_cache()
    logits = decoder.prefill(input_ids, audio_features, position_ids, cache)
    mx.eval(logits)

    seq_len = int(input_ids.shape[1])
    step_positions = plain_positions(seq_len, max_new_tokens - 1, position_ids.dtype)

    tokens, finish_reason = greedy_steps(
        decoder.step,
        cache,
        logits,
        step_positions,
        max_new_tokens,
        input_dtype=input_ids.dtype,
        eos_token_ids=eos_token_ids,
        repetition_threshold=repetition_threshold,
    )
    return GenerationResult(
        tokens=tokens,
        finish_reason=finish_reason,
        generated_tokens=len(tokens),
        max_new_tokens=max_new_tokens,
    )
