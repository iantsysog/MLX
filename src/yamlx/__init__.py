from typing import TYPE_CHECKING

from yamlx._lazy import lazy_getattr
from yamlx.capabilities import (
    Availability,
    CapabilitySpec,
    Domain,
    Requirement,
    RequirementKind,
    Task,
    available,
    checkpoint_requirement,
    extra_requirement,
    get,
    module_requirement,
    refresh,
    register,
)
from yamlx.common.errors import (
    AudioDecodeError,
    IncompleteDownloadError,
    ModelNotCachedError,
    StreamingStateError,
    WeightsValidationError,
    YamlxError,
)
from yamlx.common.results import (
    AlignedWord,
    FinishReason,
    TranscriptionChunk,
    TranscriptionResult,
)
from yamlx.common.scope import Disposable, ResourceScope

if TYPE_CHECKING:
    from yamlx.runtime import CachePolicy as CachePolicy
    from yamlx.runtime import Runtime as Runtime
    from yamlx.runtime import release_cache as release_cache
    from yamlx.runtime import resolve_runtime as resolve_runtime
    from yamlx.speech.asr.qwen3 import ForcedAligner as ForcedAligner
    from yamlx.speech.asr.qwen3 import Qwen3ASR as Qwen3ASR
    from yamlx.speech.asr.qwen3 import Qwen3ASRModel as Qwen3ASRModel
    from yamlx.speech.asr.qwen3 import Qwen3ASROption as Qwen3ASROption
    from yamlx.speech.asr.qwen3 import Tokenizer as Tokenizer
    from yamlx.speech.asr.qwen3 import abort_streaming as abort_streaming
    from yamlx.speech.asr.qwen3 import align as align
    from yamlx.speech.asr.qwen3 import build_prompt_tokens as build_prompt_tokens
    from yamlx.speech.asr.qwen3 import close as close
    from yamlx.speech.asr.qwen3 import feed_audio as feed_audio
    from yamlx.speech.asr.qwen3 import finish_streaming as finish_streaming
    from yamlx.speech.asr.qwen3 import from_pretrained as from_pretrained
    from yamlx.speech.asr.qwen3 import init_streaming as init_streaming
    from yamlx.speech.asr.qwen3 import load_aligner as load_aligner
    from yamlx.speech.asr.qwen3 import load_model as load_model
    from yamlx.speech.asr.qwen3 import load_session as load_session
    from yamlx.speech.asr.qwen3 import load_tokenizer as load_tokenizer
    from yamlx.speech.asr.qwen3 import open_session as open_session
    from yamlx.speech.asr.qwen3 import parse_asr_output as parse_asr_output
    from yamlx.speech.asr.qwen3 import transcribe as transcribe
    from yamlx.speech.asr.qwen3 import transcribe_audio as transcribe_audio
    from yamlx.speech.asr.qwen3 import warm_up as warm_up

__version__ = "0.1.0"

_LAZY_ATTRS: dict[str, str] = dict.fromkeys(
    (
        "ForcedAligner",
        "Qwen3ASR",
        "Qwen3ASRModel",
        "Qwen3ASROption",
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
        "load_model",
        "load_session",
        "load_tokenizer",
        "open_session",
        "parse_asr_output",
        "transcribe",
        "transcribe_audio",
        "warm_up",
    ),
    "yamlx.speech.asr.qwen3",
)
_LAZY_ATTRS.update({
    "CachePolicy": "yamlx.runtime",
    "Runtime": "yamlx.runtime",
    "release_cache": "yamlx.runtime",
    "resolve_runtime": "yamlx.runtime",
})

__all__ = [
    "AlignedWord",
    "AudioDecodeError",
    "Availability",
    "CachePolicy",
    "CapabilitySpec",
    "Disposable",
    "Domain",
    "FinishReason",
    "ForcedAligner",
    "IncompleteDownloadError",
    "ModelNotCachedError",
    "Qwen3ASR",
    "Qwen3ASRModel",
    "Qwen3ASROption",
    "Requirement",
    "RequirementKind",
    "ResourceScope",
    "Runtime",
    "StreamingStateError",
    "Task",
    "Tokenizer",
    "TranscriptionChunk",
    "TranscriptionResult",
    "WeightsValidationError",
    "YamlxError",
    "abort_streaming",
    "align",
    "available",
    "build_prompt_tokens",
    "checkpoint_requirement",
    "close",
    "extra_requirement",
    "feed_audio",
    "finish_streaming",
    "from_pretrained",
    "get",
    "init_streaming",
    "load_aligner",
    "load_model",
    "load_session",
    "load_tokenizer",
    "module_requirement",
    "open_session",
    "parse_asr_output",
    "refresh",
    "register",
    "release_cache",
    "resolve_runtime",
    "transcribe",
    "transcribe_audio",
    "warm_up",
]


def __getattr__(name: str) -> object:
    return lazy_getattr(__name__, _LAZY_ATTRS, name)
