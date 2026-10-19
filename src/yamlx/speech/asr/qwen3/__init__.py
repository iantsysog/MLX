from typing import TYPE_CHECKING

from yamlx._lazy import lazy_getattr

if TYPE_CHECKING:
    from .aligner import ForcedAligner as ForcedAligner
    from .aligner import align as align
    from .aligner import load_aligner as load_aligner
    from .model.model import Qwen3ASRModel as Qwen3ASRModel
    from .option import AudioEncoderOption as AudioEncoderOption
    from .option import Qwen3ASROption as Qwen3ASROption
    from .option import TextDecoderOption as TextDecoderOption
    from .protocol import build_prompt_tokens as build_prompt_tokens
    from .protocol import parse_asr_output as parse_asr_output
    from .runtime.load import load_forced_aligner as load_forced_aligner
    from .runtime.load import load_model as load_model
    from .runtime.load import resolve_model_path as resolve_model_path
    from .runtime.session import Qwen3ASR as Qwen3ASR
    from .runtime.session import close as close
    from .runtime.session import from_pretrained as from_pretrained
    from .runtime.session import load_session as load_session
    from .runtime.session import open_session as open_session
    from .runtime.session import transcribe as transcribe
    from .runtime.session import warm_up as warm_up
    from .runtime.streaming import abort_streaming as abort_streaming
    from .runtime.streaming import feed_audio as feed_audio
    from .runtime.streaming import finish_streaming as finish_streaming
    from .runtime.streaming import init_streaming as init_streaming
    from .runtime.transcribe import Aligner as Aligner
    from .runtime.transcribe import transcribe_audio as transcribe_audio
    from .tokenizer import Tokenizer as Tokenizer
    from .tokenizer import load_tokenizer as load_tokenizer

_LAZY_ATTRS = {
    "AudioEncoderOption": ".option",
    "Qwen3ASROption": ".option",
    "TextDecoderOption": ".option",
    "ForcedAligner": ".aligner",
    "align": ".aligner",
    "load_aligner": ".aligner",
    "load_forced_aligner": ".runtime.load",
    "load_model": ".runtime.load",
    "resolve_model_path": ".runtime.load",
    "Qwen3ASRModel": ".model.model",
    "Qwen3ASR": ".runtime.session",
    "close": ".runtime.session",
    "from_pretrained": ".runtime.session",
    "load_session": ".runtime.session",
    "open_session": ".runtime.session",
    "transcribe": ".runtime.session",
    "warm_up": ".runtime.session",
    "abort_streaming": ".runtime.streaming",
    "feed_audio": ".runtime.streaming",
    "finish_streaming": ".runtime.streaming",
    "init_streaming": ".runtime.streaming",
    "Tokenizer": ".tokenizer",
    "load_tokenizer": ".tokenizer",
    "Aligner": ".runtime.transcribe",
    "transcribe_audio": ".runtime.transcribe",
    "build_prompt_tokens": ".protocol",
    "parse_asr_output": ".protocol",
}

__all__ = [
    "Aligner",
    "AudioEncoderOption",
    "ForcedAligner",
    "Qwen3ASR",
    "Qwen3ASRModel",
    "Qwen3ASROption",
    "TextDecoderOption",
    "Tokenizer",
    "abort_streaming",
    "align",
    "build_prompt_tokens",
    "close",
    "feed_audio",
    "finish_streaming",
    "from_pretrained",
    "init_streaming",
    "load_aligner",
    "load_forced_aligner",
    "load_model",
    "load_session",
    "load_tokenizer",
    "open_session",
    "parse_asr_output",
    "resolve_model_path",
    "transcribe",
    "transcribe_audio",
    "warm_up",
]


def __getattr__(name: str) -> object:
    return lazy_getattr(__name__, _LAZY_ATTRS, name)
