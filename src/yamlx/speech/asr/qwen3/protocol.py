from typing import TYPE_CHECKING

from yamlx.common.languages import canonicalize_language
from yamlx.speech.asr.qwen3.tokenizer import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    Tokenizer,
    encode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "build_aligner_prompt",
    "build_prompt_tokens",
    "parse_asr_output",
]

_ASR_TEXT = "<asr_text>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END = "<|audio_end|>"
_AUDIO_PAD = "<|audio_pad|>"
_TIMESTAMP_SLOT = "<timestamp><timestamp>"

_PROMPT_PREFIX = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

_TRAILING_SPECIALS = ("<|im_end|>", "<|endoftext|>")


def build_prompt_tokens(
    tokenizer: Tokenizer,
    *,
    n_audio_tokens: int,
    language: str | None = None,
) -> list[int]:
    tokens = encode(tokenizer, _PROMPT_PREFIX)
    tokens.append(AUDIO_START_TOKEN_ID)
    tokens.extend([AUDIO_PAD_TOKEN_ID] * n_audio_tokens)
    tokens.append(AUDIO_END_TOKEN_ID)
    tokens.extend(encode(tokenizer, _PROMPT_SUFFIX))
    canonical = canonicalize_language(language)
    if canonical is not None:
        tokens.extend(encode(tokenizer, f"language {canonical}{_ASR_TEXT}"))
    return tokens


def build_aligner_prompt(words: Sequence[str], n_audio_tokens: int) -> str:
    body = _TIMESTAMP_SLOT.join(words) + _TIMESTAMP_SLOT
    return f"{_AUDIO_START}{_AUDIO_PAD * n_audio_tokens}{_AUDIO_END}{body}"


def parse_asr_output(
    text: str,
    *,
    forced_language: str | None = None,
) -> tuple[str | None, str]:
    s = text.strip()
    if forced_language is not None:
        forced = canonicalize_language(forced_language) or forced_language
        if _ASR_TEXT in s:
            s = s.split(_ASR_TEXT, 1)[1]
        return forced, _strip_trailing_special_tokens(s)
    if _ASR_TEXT not in s:
        return None, _strip_trailing_special_tokens(s)
    head, _, tail = s.partition(_ASR_TEXT)
    language = head.strip().removeprefix("language ").strip()
    transcript = _strip_trailing_special_tokens(tail)
    if not language or language.casefold() == "none":
        return None, transcript
    return canonicalize_language(language) or language, transcript


def _strip_trailing_special_tokens(text: str) -> str:
    out = text.strip()
    while marker := next((m for m in _TRAILING_SPECIALS if out.endswith(m)), None):
        out = out[: -len(marker)].rstrip()
    return out
