from dataclasses import dataclass
from enum import Enum


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True)
class AlignedWord:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptionChunk:
    text: str
    offset_seconds: float


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    chunks: tuple[TranscriptionChunk, ...]
    words: tuple[AlignedWord, ...] | None
    finish_reason: FinishReason

    @property
    def truncated(self) -> bool:
        return self.finish_reason is FinishReason.LENGTH
