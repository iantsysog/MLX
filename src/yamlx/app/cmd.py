import argparse
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yamlx import __version__
from yamlx.speech.asr.qwen3.presets import MODEL_DTYPE_NAMES, QUANTIZATION_CHOICES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from yamlx.common.results import TranscriptionResult
    from yamlx.speech.asr.qwen3.runtime.session import Qwen3ASR
    from yamlx.speech.asr.qwen3.runtime.transcribe import Aligner

lazy from yamlx.common.audio import load_audio
lazy from yamlx.common.errors import ModelNotCachedError, YamlxError
lazy from yamlx.speech.asr.qwen3.aligner import align, load_aligner
lazy from yamlx.speech.asr.qwen3.runtime.load import MODEL_DTYPES, resolve_model_path
lazy from yamlx.speech.asr.qwen3.runtime.session import close, load_session, transcribe

lazy from .sinks import get_sink

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        message = "must be an integer"
        raise argparse.ArgumentTypeError(message) from None
    if parsed <= 0:
        message = "must be > 0"
        raise argparse.ArgumentTypeError(message)
    return parsed


_ARGUMENT_SPECS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (("audio",), {"nargs": "+", "help": "Audio file(s) to transcribe."}),
    (
        ("--model",),
        {
            "default": DEFAULT_MODEL_ID,
            "help": f"ASR model repository ID or local directory (default: {DEFAULT_MODEL_ID})",
        },
    ),
    (
        ("--language",),
        {
            "default": None,
            "help": "Force a language (for example, English or Chinese).",
        },
    ),
    (
        ("--output-dir", "-o"),
        {"default": ".", "help": "Output directory (default: the current directory)."},
    ),
    (
        ("--output-format", "-f"),
        {
            "default": "txt",
            "choices": ["txt", "json", "srt", "vtt"],
            "help": "Output format (default: txt); srt and vtt enable --timestamps automatically.",
        },
    ),
    (
        ("--timestamps",),
        {
            "action": "store_true",
            "help": "Attach word-level timestamps from the forced aligner.",
        },
    ),
    (
        ("--aligner",),
        {
            "default": DEFAULT_ALIGNER_ID,
            "help": f"Forced aligner repository ID or local directory (default: {DEFAULT_ALIGNER_ID})",
        },
    ),
    (
        ("--dtype",),
        {
            "default": "float16",
            "choices": MODEL_DTYPE_NAMES,
            "help": "Model dtype (default: float16).",
        },
    ),
    (
        ("--quantization",),
        {
            "default": None,
            "choices": QUANTIZATION_CHOICES,
            "help": (
                'Quantization preset. "4-bit" mixes a 4-bit decoder with an '
                '8-bit encoder; "8-bit" quantizes everything (default: none).'
            ),
        },
    ),
    (
        ("--max-new-tokens",),
        {
            "type": _positive_int,
            "default": None,
            "help": (
                "Maximum tokens to generate per chunk. "
                "(default: adaptive, 128-512 tokens, about 12 tokens per second of audio)."
            ),
        },
    ),
    (
        ("--verbose",),
        {"action": "store_true", "help": "Print per-file diagnostics to stderr."},
    ),
    (
        ("--no-progress",),
        {"action": "store_true", "help": "Disable progress updates on stderr."},
    ),
)
_OUTPUT_GROUP_SPECS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("--quiet",),
        {"action": "store_true", "help": "Suppress transcript text on stdout."},
    ),
    (
        ("--stdout-only",),
        {
            "action": "store_true",
            "help": "Print transcripts to stdout without writing output files.",
        },
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yamlx",
        description=(
            "Transcribe audio files with Qwen3-ASR on Apple Silicon "
            "using MLX. Models load from the local HuggingFace cache "
            "only; no network access."
        ),
    )
    for names, kwargs in _ARGUMENT_SPECS:
        parser.add_argument(*names, **kwargs)
    output_group = parser.add_mutually_exclusive_group()
    for names, kwargs in _OUTPUT_GROUP_SPECS:
        output_group.add_argument(*names, **kwargs)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _preflight_aligner_cache(aligner_source: str) -> None:
    try:
        resolve_model_path(aligner_source)
    except ModelNotCachedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _print_progress(value: float) -> None:
    print(f"Progress: {value * 100.0:.1f}%", file=sys.stderr)


def _process_file(
    args: argparse.Namespace,
    session: Qwen3ASR,
    aligner: Aligner | None,
    sink: Callable[[TranscriptionResult, str | Path], None],
    audio_path: str,
) -> bool:
    timestamps = aligner is not None
    if not Path(audio_path).exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        return True
    if args.verbose:
        print(f"Transcribing: {audio_path}", file=sys.stderr)
    start_time = time.monotonic()
    try:
        audio = load_audio(audio_path)
        result = transcribe(
            session,
            audio,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            timestamps=timestamps,
            aligner=aligner,
            on_progress=None if args.no_progress else _print_progress,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return True
    if not args.quiet:
        print(result.text)
    if args.verbose:
        elapsed = time.monotonic() - start_time
        print(f"Language: {result.language}", file=sys.stderr)
        print(
            f"Duration: {result.duration:.2f}s of audio in {elapsed:.2f}s",
            file=sys.stderr,
        )
    if args.stdout_only:
        return False
    out_path = Path(args.output_dir) / f"{Path(audio_path).stem}.{args.output_format}"
    try:
        sink(result, out_path)
    except Exception as exc:
        print(f"Error: failed to write {out_path}: {exc}", file=sys.stderr)
        return True
    if args.verbose:
        print(f"Written: {out_path}", file=sys.stderr)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    wants_subtitles = args.output_format in {"srt", "vtt"}
    timestamps = bool(args.timestamps or wants_subtitles)
    if wants_subtitles and not args.timestamps:
        print(
            "Info: enabling timestamps because subtitle output was requested.",
            file=sys.stderr,
        )
    if timestamps:
        _preflight_aligner_cache(args.aligner)

    dtype = MODEL_DTYPES[args.dtype]
    aligner: Aligner | None = (
        partial(align, load_aligner(args.aligner, dtype=dtype)) if timestamps else None
    )
    sink = get_sink(args.output_format)
    if not args.stdout_only:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        session = load_session(args.model, dtype=dtype, quantization=args.quantization)
    except YamlxError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        had_error = False
        for audio_path in args.audio:
            had_error = (
                _process_file(args, session, aligner, sink, audio_path) or had_error
            )
    finally:
        close(session)
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
