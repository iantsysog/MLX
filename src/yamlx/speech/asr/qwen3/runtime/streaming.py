from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from yamlx.common.audio import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    log_mel_spectrogram,
    sanitize_pcm,
)
from yamlx.common.errors import StreamingStateError
from yamlx.common.generation import (
    AUTO_MAX_NEW_TOKENS_FLOOR,
    greedy_steps,
    resolve_max_new_tokens,
)
from yamlx.common.kv import KVCache, fork_cache, trim_cache
from yamlx.common.languages import canonicalize_language, join_text_parts
from yamlx.common.numeric import scalar_int
from yamlx.common.positions import plain_positions
from yamlx.common.scope import Disposable, ResourceScope
from yamlx.runtime import CachePolicy, release_cache
from yamlx.speech.asr.qwen3.protocol import build_prompt_tokens, parse_asr_output
from yamlx.speech.asr.qwen3.tokenizer import decode

from .session import _CLOSED_MESSAGE

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from yamlx.speech.asr.qwen3.model.encoder import AudioEncoder
    from yamlx.speech.asr.qwen3.model.model import Qwen3ASRModel
    from yamlx.speech.asr.qwen3.tokenizer import Tokenizer

    from .session import Qwen3ASR

UNFIXED_CHUNK_NUM = 2

UNFIXED_TOKEN_NUM = 5

_REPLACEMENT_CHAR = "\ufffd"
_FINAL_FRAME_SLACK_SAMPLES = N_FFT // 2 - HOP_LENGTH


def _empty_f32() -> np.ndarray:
    return np.array([], dtype=np.float32)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class _WindowPrefixCache:
    mel_max: float
    blocks: list[mx.array] = field(default_factory=list)
    kv: KVCache | None = None
    kv_positions: int = 0


@dataclass(frozen=True)
class _WindowPlan:
    prompt: list[int]
    split: int
    prefix_ids: list[int]
    prefix_cache: _WindowPrefixCache | None = None
    new_features: mx.array | None = None
    n_new_blocks: int = 0
    tokens_per_block: int = 0
    head: int = 0


@dataclass
class StreamingState:
    session: Qwen3ASR
    status: str = "streaming"
    buffer: np.ndarray = field(default_factory=_empty_f32)
    audio_accum: np.ndarray = field(default_factory=_empty_f32)
    text: str = ""
    committed_text: str = ""
    window_text: str = ""
    stable_text: str = ""
    language: str | None = None
    forced_language: str | None = None
    chunk_id: int = 0
    committed_windows: int = 0
    unfixed_chunk_num: int = UNFIXED_CHUNK_NUM
    unfixed_token_num: int = UNFIXED_TOKEN_NUM
    chunk_size_samples: int = int(2.0 * SAMPLE_RATE)
    max_context_samples: int = int(30.0 * SAMPLE_RATE)
    max_new_tokens: int = AUTO_MAX_NEW_TOKENS_FLOOR
    adaptive_budget: bool = True
    reuse_window_prefix: bool = True
    _window_language: str | None = None
    _window_raw_ids: list[int] = field(default_factory=list)
    _window_chunk_id: int = 0
    _prefix_cache: _WindowPrefixCache | None = None
    _cascade: Disposable | None = None
    scope: ResourceScope = field(default_factory=ResourceScope, repr=False)

    @property
    def chunks_processed(self) -> int:
        return self.chunk_id

    @property
    def window_audio_seconds(self) -> float:
        return float(self.audio_accum.size) / SAMPLE_RATE

    @property
    def unstable_text(self) -> str:
        if self.text.startswith(self.stable_text):
            return self.text[len(self.stable_text) :]
        return self.text


def init_streaming(
    session: Qwen3ASR,
    *,
    language: str | None = None,
    chunk_size_sec: float = 2.0,
    max_context_sec: float = 30.0,
    unfixed_chunk_num: int = UNFIXED_CHUNK_NUM,
    unfixed_token_num: int = UNFIXED_TOKEN_NUM,
    max_new_tokens: int | None = None,
    reuse_window_prefix: bool = True,
) -> StreamingState:
    _require(chunk_size_sec > 0, f"chunk_size_sec must be > 0; got {chunk_size_sec}.")
    _require(
        max_context_sec > 0, f"max_context_sec must be > 0; got {max_context_sec}."
    )
    _require(
        max_context_sec >= chunk_size_sec,
        f"max_context_sec ({max_context_sec}) must be >= "
        f"chunk_size_sec ({chunk_size_sec}).",
    )
    chunk_size_samples = round(chunk_size_sec * SAMPLE_RATE)
    max_context_samples = round(max_context_sec * SAMPLE_RATE)
    _require(
        chunk_size_samples >= 1,
        f"chunk_size_sec ({chunk_size_sec}) is too small; it rounds "
        f"to {chunk_size_samples} samples.",
    )
    _require(
        unfixed_chunk_num >= 0,
        f"unfixed_chunk_num must be >= 0; got {unfixed_chunk_num}.",
    )
    _require(
        unfixed_token_num >= 0,
        f"unfixed_token_num must be >= 0; got {unfixed_token_num}.",
    )
    budget = resolve_max_new_tokens(max_new_tokens, audio_duration_sec=chunk_size_sec)
    forced = canonicalize_language(language)
    with session.lock:
        if session.closed or session.model is None or session.tokenizer is None:
            raise RuntimeError(_CLOSED_MESSAGE)
        if session.active_stream is not None:
            raise StreamingStateError("streaming", "init_streaming")
        if reuse_window_prefix:
            _check_block_alignment(session)
        state = StreamingState(
            session=session,
            language=forced,
            forced_language=forced,
            chunk_size_samples=chunk_size_samples,
            max_context_samples=max_context_samples,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            max_new_tokens=budget,
            adaptive_budget=max_new_tokens is None,
            reuse_window_prefix=reuse_window_prefix,
            _window_language=forced,
        )
        state.scope.defer(partial(_clear_active_slot, session, state))
        state.scope.defer(partial(_drop_prefix_cache, state))
        state._cascade = session.scope.defer(
            partial(_abort_session_stream, session, state)
        )
        session.active_stream = state
    return state


def feed_audio(state: StreamingState, pcm: np.ndarray) -> StreamingState:
    _require_streaming(state, "feed_audio")
    audio = sanitize_pcm(pcm)
    if audio.size == 0:
        return state
    session = state.session
    with session.lock:
        _check_open_locked(state, "feed_audio")
        with _aborted_on_error(state):
            state.buffer = np.concatenate([state.buffer, audio])
            while state.buffer.size >= state.chunk_size_samples:
                chunk = state.buffer[: state.chunk_size_samples]
                state.buffer = state.buffer[state.chunk_size_samples :]
                _advance_chunk(state, chunk)
    return state


def finish_streaming(state: StreamingState) -> StreamingState:
    _require_streaming(state, "finish_streaming")
    session = state.session
    with session.lock:
        _check_open_locked(state, "finish_streaming")
        with _aborted_on_error(state):
            if state.buffer.size > 0:
                tail = state.buffer
                state.buffer = _empty_f32()
                _advance_chunk(state, tail)
            state.stable_text = state.text
            state.status = "finished"
            _release(state)
    return state


def abort_streaming(state: StreamingState) -> None:
    session = state.session
    with session.lock:
        _abort_locked(state)


@contextmanager
def _aborted_on_error(state: StreamingState) -> Generator[None]:
    try:
        yield
    except BaseException:
        _abort_locked(state)
        raise


def _require_streaming(state: StreamingState, action: str) -> None:
    if state.status != "streaming":
        raise StreamingStateError(state.status, action)


def _check_open_locked(state: StreamingState, action: str) -> None:
    _require_streaming(state, action)
    session = state.session
    if session.closed or session.model is None or session.tokenizer is None:
        _abort_locked(state)
        raise RuntimeError(_CLOSED_MESSAGE)


def _abort_locked(state: StreamingState) -> None:
    if state.status != "streaming":
        return
    state.status = "aborted"
    _release(state)


def _release(state: StreamingState) -> None:
    if state._cascade is not None:
        state._cascade.discard()
        state._cascade = None
    state.scope.close()


def _abort_session_stream(session: Qwen3ASR, state: StreamingState) -> None:
    with session.lock:
        _abort_locked(state)


def _clear_active_slot(session: Qwen3ASR, state: StreamingState) -> None:
    if session.active_stream is state:
        session.active_stream = None


def _drop_prefix_cache(state: StreamingState) -> None:
    state._prefix_cache = None


def _check_block_alignment(session: Qwen3ASR) -> None:
    option = session.option
    if option is None:
        raise RuntimeError(_CLOSED_MESSAGE)
    block = option.encoder.n_window_infer
    chunk = option.encoder.n_window * 2
    if block < 1 or chunk < 1 or block % chunk != 0:
        message = (
            "Streaming prefix reuse requires n_window_infer to be a "
            f"positive multiple of 2*n_window; got n_window_infer={block}, "
            f"n_window={option.encoder.n_window})."
        )
        raise ValueError(message)


def _session_runtime(
    state: StreamingState,
) -> tuple[Qwen3ASRModel, Tokenizer, mx.Dtype]:
    session = state.session
    model = session.model
    tokenizer = session.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError(_CLOSED_MESSAGE)
    return model, tokenizer, session.dtype


def _token_budget(state: StreamingState) -> int:
    if not state.adaptive_budget:
        return state.max_new_tokens
    return max(
        state.max_new_tokens,
        resolve_max_new_tokens(None, audio_duration_sec=state.window_audio_seconds),
    )


def _extend_window(state: StreamingState, audio: np.ndarray) -> None:
    if (
        state.audio_accum.size > 0
        and state.audio_accum.size + audio.size > state.max_context_samples
    ):
        _commit_window(state)
    state.audio_accum = np.concatenate([state.audio_accum, audio])


def _commit_window(state: StreamingState) -> None:
    state.committed_text = join_text_parts(
        (state.committed_text, state.window_text), state._window_language
    )
    state.committed_windows += 1
    state.stable_text = state.committed_text
    state.window_text = ""
    state.audio_accum = _empty_f32()
    state._window_raw_ids = []
    state._window_chunk_id = 0
    state._prefix_cache = None
    state._window_language = state.forced_language or state.language


def _advance_chunk(state: StreamingState, chunk: np.ndarray) -> None:
    _extend_window(state, chunk)
    _decode_and_update(state)
    if state.chunk_id >= state.unfixed_chunk_num:
        state.stable_text, _ = _split_stable_unstable(
            state.stable_text,
            state.text,
            unfixed_tokens=state.unfixed_token_num,
        )
    state.chunk_id += 1


def _decode_and_update(state: StreamingState) -> None:
    window_text, language = _decode_window(state)
    if language is not None:
        state.language = language
    state.window_text = window_text
    state.text = join_text_parts(
        (state.committed_text, state.window_text), state.language
    )


def _decode_window(state: StreamingState) -> tuple[str, str | None]:
    model, tokenizer, dtype = _session_runtime(state)
    plan, audio_features, cache = _prepare_prefill(state, model, tokenizer, dtype)
    input_ids = mx.array([plan.prompt[plan.split :] + plan.prefix_ids], dtype=mx.int32)
    position_ids = plain_positions(plan.split, int(input_ids.shape[1]))
    logits = model.prefill(input_ids, audio_features, position_ids, cache)
    mx.eval(logits)
    seq_len = plan.split + int(input_ids.shape[1])
    budget = _token_budget(state)
    tokens, _ = greedy_steps(
        model.step,
        cache,
        logits,
        plain_positions(seq_len, budget - 1),
        budget,
        input_dtype=mx.int32,
    )
    if plan.prefix_cache is not None:
        _adopt_decoded_prefix(plan, cache)
    raw_ids = plan.prefix_ids + tokens
    state._window_raw_ids = raw_ids
    state._window_chunk_id += 1
    language, text = parse_asr_output(
        decode(tokenizer, raw_ids), forced_language=state._window_language
    )
    del plan, logits, input_ids, position_ids, audio_features, cache
    if state.session.runtime.cache_policy is CachePolicy.CHUNK:
        release_cache()
    return text, language


def _prepare_prefill(
    state: StreamingState,
    model: Qwen3ASRModel,
    tokenizer: Tokenizer,
    dtype: mx.Dtype,
) -> tuple[_WindowPlan, mx.array, KVCache]:
    prefix_ids: list[int] = []
    prefix_cache: _WindowPrefixCache | None = None
    new_features: mx.array | None = None
    n_new_blocks = tokens_per_block = 0
    split = 0
    if state.reuse_window_prefix:
        mel = log_mel_spectrogram(
            np.ascontiguousarray(state.audio_accum, dtype=np.float32)
        )
        prefix_cache, new_features, n_new_blocks, tokens_per_block = (
            _encode_with_prefix_cache(state, model, mel, dtype)
        )
        if state._window_chunk_id >= state.unfixed_chunk_num:
            prefix_ids = _rollback_prefix_ids(
                state._window_raw_ids,
                state.unfixed_token_num,
                partial(decode, tokenizer),
            )
        n_audio_tokens = len(prefix_cache.blocks) * tokens_per_block + int(
            new_features.shape[0]
        )
        kv = prefix_cache.kv
        if kv is not None:
            cache = fork_cache(kv)
            split = prefix_cache.kv_positions
            audio_features = new_features[None, :, :]
        else:
            cache = model.create_cache()
            audio_features = mx.concatenate(
                [*prefix_cache.blocks, new_features], axis=0
            )[None, :, :]
    else:
        audio_features, n_audio_tokens = model.encode_waveform(state.audio_accum, dtype)
        cache = model.create_cache()
    prompt = build_prompt_tokens(
        tokenizer, n_audio_tokens=n_audio_tokens, language=state._window_language
    )
    head = _prompt_audio_start(prompt, int(model.audio_token_id), n_audio_tokens)
    plan = _WindowPlan(
        prompt=prompt,
        split=split,
        prefix_ids=prefix_ids,
        prefix_cache=prefix_cache,
        new_features=new_features,
        n_new_blocks=n_new_blocks,
        tokens_per_block=tokens_per_block,
        head=head,
    )
    return plan, audio_features, cache


def _encode_with_prefix_cache(
    state: StreamingState,
    model: Qwen3ASRModel,
    mel: np.ndarray,
    dtype: mx.Dtype,
) -> tuple[_WindowPrefixCache, mx.array, int, int]:
    encoder: AudioEncoder = model.audio_tower
    block = encoder.attention_window_frames
    mel_max = float(mel.max())
    n_samples = int(state.audio_accum.size)
    stable_frames = max(0, (n_samples - _FINAL_FRAME_SLACK_SAMPLES) // HOP_LENGTH)
    n_complete = stable_frames // block
    mel2d = mx.array(mel).astype(dtype)
    prefix_cache = state._prefix_cache
    if (
        prefix_cache is None
        or prefix_cache.mel_max != mel_max
        or len(prefix_cache.blocks) > n_complete
    ):
        prefix_cache = _WindowPrefixCache(mel_max)
        for j in range(n_complete):
            feats = encoder.encode_frames(mel2d[:, j * block : (j + 1) * block])
            mx.eval(feats)
            prefix_cache.blocks.append(feats)
        state._prefix_cache = prefix_cache
    tokens_per_block = scalar_int(
        encoder.get_output_lengths(mx.array([block], dtype=mx.int32))[0]
    )
    n_new_blocks = n_complete - len(prefix_cache.blocks)
    tail_start = len(prefix_cache.blocks) * block
    if mel.shape[1] > tail_start:
        new_features = encoder.encode_frames(mel2d[:, tail_start:])
        mx.eval(new_features)
    else:
        new_features = mx.zeros((0, int(encoder.option.output_dim)), dtype=dtype)
    return prefix_cache, new_features, n_new_blocks, tokens_per_block


def _adopt_decoded_prefix(plan: _WindowPlan, cache: KVCache) -> None:
    prefix_cache = plan.prefix_cache
    new_features = plan.new_features
    if prefix_cache is None or new_features is None:
        return
    total_blocks = len(prefix_cache.blocks) + plan.n_new_blocks
    if total_blocks <= 0:
        return
    for j in range(plan.n_new_blocks):
        block_feats = new_features[
            j * plan.tokens_per_block : (j + 1) * plan.tokens_per_block
        ]
        mx.eval(block_feats)
        prefix_cache.blocks.append(block_feats)
    target = plan.head + total_blocks * plan.tokens_per_block
    if target > prefix_cache.kv_positions and cache.offset >= target:
        adopted = fork_cache(cache)
        trim_cache(adopted, adopted.offset - target)
        for key in adopted.keys:
            if key is not None:
                mx.eval(key)
        for value in adopted.values:
            if value is not None:
                mx.eval(value)
        prefix_cache.kv = adopted
        prefix_cache.kv_positions = target


def _prompt_audio_start(
    prompt: Sequence[int],
    audio_token_id: int,
    n_audio_tokens: int,
) -> int:
    try:
        head = prompt.index(audio_token_id)
    except ValueError:
        message = "The prompt is missing audio placeholder tokens."
        raise RuntimeError(message) from None
    run_end = head + n_audio_tokens
    if (
        run_end > len(prompt)
        or any(token != audio_token_id for token in prompt[head:run_end])
        or (run_end < len(prompt) and prompt[run_end] == audio_token_id)
    ):
        message = (
            f"prompt must contain a contiguous run of {n_audio_tokens} "
            f"audio placeholder tokens starting at index {head}."
        )
        raise RuntimeError(message)
    return head


def _split_stable_unstable(
    prev_stable: str,
    new_text: str,
    *,
    unfixed_tokens: int,
) -> tuple[str, str]:
    if unfixed_tokens <= 0:
        return new_text, ""
    units: list[str]
    joiner: str
    if not new_text:
        units = []
        joiner = ""
    elif any(char.isspace() for char in new_text):
        units, joiner = new_text.split(), " "
    else:
        units, joiner = list(new_text), ""
    if len(units) <= unfixed_tokens:
        return prev_stable, new_text
    stable = joiner.join(units[:-unfixed_tokens])
    if len(stable) < len(prev_stable) and new_text.startswith(prev_stable):
        stable = prev_stable
    if not new_text.startswith(stable):
        return stable, new_text
    return stable, new_text[len(stable) :]


def _rollback_prefix_ids(
    raw_ids: Sequence[int],
    unfixed_tokens: int,
    decode: Callable[[Sequence[int]], str],
) -> list[int]:
    end = max(0, len(raw_ids) - unfixed_tokens)
    while end > 0 and _REPLACEMENT_CHAR in decode(raw_ids[:end]):
        end -= 1
    return list(raw_ids[:end])
