import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from yamlx.common.languages import is_no_space_language, join_text_parts
from yamlx.common.results import AlignedWord, TranscriptionResult

type ResultSink = Callable[[TranscriptionResult, str | Path], None]

_MAX_WORDS_PER_CUE = 10
_MAX_CUE_CHARS = 42
_MAX_CUE_DURATION_SEC = 6.0
_MAX_GAP_SEC = 0.8


def _emit(output_path: str | Path, text: str) -> None:
    Path(output_path).write_text(text, encoding="utf-8")


def _result_payload(result: TranscriptionResult) -> dict[str, object]:
    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "chunks": [
            {"text": chunk.text, "offset_seconds": chunk.offset_seconds}
            for chunk in result.chunks
        ],
        "words": None
        if result.words is None
        else [
            {"word": word.word, "start": word.start, "end": word.end}
            for word in result.words
        ],
        "finish_reason": result.finish_reason.value,
        "truncated": result.truncated,
    }


def sink_txt(result: TranscriptionResult, output_path: str | Path) -> None:
    _emit(output_path, result.text + "\n")


def sink_json(result: TranscriptionResult, output_path: str | Path) -> None:
    _emit(
        output_path,
        json.dumps(_result_payload(result), indent=2, ensure_ascii=False) + "\n",
    )


def _format_timestamp(seconds: float, sep: str) -> str:
    total_ms = max(0, round(seconds * 1000.0))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _cues_text(cues: list[_Cue], *, numbered: bool, sep: str) -> str:
    blocks = (
        (f"{index}\n" if numbered else "") + f"{_format_timestamp(cue.start, sep)} --> "
        f"{_format_timestamp(cue.end, sep)}\n{cue.text}\n\n"
        for index, cue in enumerate(cues, 1)
    )
    return "".join(blocks)


def sink_srt(result: TranscriptionResult, output_path: str | Path) -> None:
    _emit(
        output_path, _cues_text(_subtitle_cues(result, "SRT"), numbered=True, sep=",")
    )


def sink_vtt(result: TranscriptionResult, output_path: str | Path) -> None:
    text = _cues_text(_subtitle_cues(result, "VTT"), numbered=False, sep=".")
    _emit(output_path, f"WEBVTT\n\n{text}")


def get_sink(fmt: str) -> ResultSink:
    try:
        return _SINKS[fmt]
    except KeyError:
        message = (
            f"Unknown output format '{fmt}'. Supported formats: txt, json, srt, vtt."
        )
        raise ValueError(message) from None


_SINKS: frozendict[str, ResultSink] = frozendict({
    "txt": sink_txt,
    "json": sink_json,
    "srt": sink_srt,
    "vtt": sink_vtt,
})


class _Cue(NamedTuple):
    text: str
    start: float
    end: float


def _subtitle_cues(result: TranscriptionResult, format_name: str) -> list[_Cue]:
    if result.words is None:
        message = (
            f"{format_name} output requires word timestamps. Re-run with --timestamps."
        )
        raise ValueError(message)
    words = _restore_punctuation(result.words, result.text)
    groups = _group_words(words, language=result.language)
    return [
        _Cue(
            text=_join_words(group, language=result.language),
            start=group[0].start,
            end=group[-1].end,
        )
        for group in groups
    ]


def _group_words(
    words: tuple[AlignedWord, ...], *, language: str | None
) -> list[list[AlignedWord]]:
    if not words:
        return []
    no_space = is_no_space_language(language)
    groups: list[list[AlignedWord]] = []
    current: list[AlignedWord] = []
    width_broken: set[int] = set()

    for word in words:
        if not word.word.strip():
            continue
        if not current:
            current.append(word)
            continue
        current_text = _join_words(current, language=language)
        candidate_text = _join_words([*current, word], language=language)
        last = current[-1]
        hard_break = (
            word.start - last.end >= _MAX_GAP_SEC
            or word.end - current[0].start > _MAX_CUE_DURATION_SEC
            or (not no_space and len(current) >= _MAX_WORDS_PER_CUE)
            or _ends_sentence(last.word)
            or (
                _ends_clause(last.word)
                and _display_width(current_text) * 2 >= _MAX_CUE_CHARS
            )
        )
        if hard_break or _display_width(candidate_text) > _MAX_CUE_CHARS:
            groups.append(current)
            if not hard_break:
                width_broken.add(len(groups))
            current = [word]
        else:
            current.append(word)

    if current:
        groups.append(current)
    for index in width_broken:
        if index < len(groups):
            _balance_pair(groups[index - 1], groups[index], language=language)
    return groups


def _balance_pair(
    left: list[AlignedWord],
    right: list[AlignedWord],
    *,
    language: str | None,
) -> None:
    threshold = _MAX_CUE_CHARS // 3

    def width(items: list[AlignedWord]) -> int:
        return _display_width(_join_words(items, language=language))

    while len(left) > 1 and width(right) < threshold:
        moved = [left[-1], *right]
        if width(moved) > _MAX_CUE_CHARS or width(left[:-1]) < width(moved):
            break
        right.insert(0, left.pop())


def _restore_punctuation(
    words: tuple[AlignedWord, ...], text: str
) -> tuple[AlignedWord, ...]:
    n = len(text)
    i = 0
    restored: list[AlignedWord] = []
    for word in words:
        chars = [ch for ch in word.word if not ch.isspace()]
        if not chars:
            restored.append(word)
            continue
        leading: list[str] = []
        while i < n and (text[i].isspace() or _is_punct(text[i])):
            if not text[i].isspace():
                leading.append(text[i])
            i += 1
        for ch in chars:
            if i < n and text[i].casefold() == ch.casefold():
                i += 1
            else:
                return words
        trailing: list[str] = []
        while i < n and _is_punct(text[i]) and not _is_opening_punct(text[i]):
            trailing.append(text[i])
            i += 1
        restored.append(
            AlignedWord(
                word="".join(leading) + word.word.strip() + "".join(trailing),
                start=word.start,
                end=word.end,
            )
        )
    return tuple(restored)


def _is_punct(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _is_opening_punct(char: str) -> bool:
    return unicodedata.category(char) in {"Ps", "Pi"}


def _display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text
    )


_CLOSING_PUNCT = re.escape("\"'”’」』）)】》〉]")
_ENDS_SENTENCE = re.compile(r"[.!?。！？…][" + _CLOSING_PUNCT + r"]*$")
_ENDS_CLAUSE = re.compile(r"[,;:，、；：][" + _CLOSING_PUNCT + r"]*$")


def _ends_sentence(text: str) -> bool:
    return bool(_ENDS_SENTENCE.search(text.strip()))


def _ends_clause(text: str) -> bool:
    return bool(_ENDS_CLAUSE.search(text.strip()))


def _join_words(words: list[AlignedWord], *, language: str | None) -> str:
    joined = join_text_parts([word.word for word in words], language)
    spaced = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    return re.sub(r"([(\[{])\s+", r"\1", spaced)
