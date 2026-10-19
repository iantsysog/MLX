import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re

_MERGE_RANK_PARTS = 2

if TYPE_CHECKING:
    from collections.abc import Sequence

AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
ENDOFTEXT_TOKEN_ID = 151643
ASR_TEXT_TOKEN_ID = 151704
NON_SPEECH_TOKEN_ID = 151675

PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@dataclass
class Tokenizer:
    vocab: dict[str, int]
    id_to_token: dict[int, str]
    byte_encoder: dict[int, str]
    byte_decoder: dict[str, int]
    pattern: re.Pattern[str]
    bpe_ranks: dict[tuple[str, str], int]
    errors: str
    added_token_ids: dict[str, int]
    added_id_to_token: dict[int, str]
    special_ids: set[int]
    unk_token_id: int
    added_pattern: re.Pattern[str] | None
    bpe_cache: dict[str, list[str]] = field(default_factory=dict)


def load_tokenizer(model_dir: str | Path) -> Tokenizer:
    path = Path(model_dir)
    vocab_raw = _load_json(path / "vocab.json")
    vocab = {str(token): int(idx) for token, idx in vocab_raw.items()}
    byte_encoder = _bytes_to_unicode()
    meta = _load_json(path / "tokenizer_config.json")

    added_token_ids: dict[str, int] = {}
    added_id_to_token: dict[int, str] = {}
    special_ids: set[int] = set()
    for id_str, token_meta in (meta.get("added_tokens_decoder") or {}).items():
        token_id = int(id_str)
        content = str(token_meta.get("content", ""))
        if not content:
            continue
        added_token_ids[content] = token_id
        added_id_to_token[token_id] = content
        if bool(token_meta.get("special", False)):
            special_ids.add(token_id)

    fallback_unk = str(meta.get("eos_token") or "<|im_end|>")
    unk_token = str(meta.get("unk_token") or fallback_unk)
    unk_token_id = added_token_ids.get(
        unk_token, vocab.get(unk_token, ENDOFTEXT_TOKEN_ID)
    )

    if added_token_ids:
        escaped = "|".join(
            re.escape(token) for token in sorted(added_token_ids, key=len, reverse=True)
        )
        added_pattern: re.Pattern[str] | None = re.compile(escaped)
    else:
        added_pattern = None

    return Tokenizer(
        vocab=vocab,
        id_to_token={v: k for k, v in vocab.items()},
        byte_encoder=byte_encoder,
        byte_decoder={v: k for k, v in byte_encoder.items()},
        pattern=re.compile(PRETOKENIZE_PATTERN),
        bpe_ranks=_load_bpe_ranks(path / "merges.txt"),
        errors=str(meta.get("errors", "replace")),
        added_token_ids=added_token_ids,
        added_id_to_token=added_id_to_token,
        special_ids=special_ids,
        unk_token_id=unk_token_id,
        added_pattern=added_pattern,
    )


def encode(tok: Tokenizer, text: str) -> list[int]:
    ids: list[int] = []
    for token in _token_strings(tok, text):
        token_id = tok.added_token_ids.get(token)
        if token_id is None:
            token_id = tok.vocab.get(token, tok.unk_token_id)
        ids.append(token_id)
    return ids


def decode(
    tok: Tokenizer, ids: Sequence[int], *, skip_special_tokens: bool = False
) -> str:
    pieces: list[str] = []
    byte_tokens: list[str] = []

    def flush_bytes() -> None:
        if not byte_tokens:
            return
        joined = "".join(byte_tokens)
        data = bytearray(tok.byte_decoder[ch] for ch in joined)
        pieces.append(data.decode("utf-8", errors=tok.errors))
        byte_tokens.clear()

    for token_id in ids:
        added = tok.added_id_to_token.get(token_id)
        if added is not None:
            if skip_special_tokens and token_id in tok.special_ids:
                continue
            flush_bytes()
            pieces.append(added)
            continue
        token = tok.id_to_token.get(token_id)
        if token is not None:
            byte_tokens.append(token)
    flush_bytes()
    return "".join(pieces)


def tokenize(tok: Tokenizer, text: str) -> list[str]:
    return _token_strings(tok, text)


def pretokenize(tok: Tokenizer, text: str) -> list[str]:
    return [_byte_encode(tok, piece) for piece in tok.pattern.findall(text)]


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _load_bpe_ranks(path: Path) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for rank, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != _MERGE_RANK_PARTS:
            continue
        ranks[parts[0], parts[1]] = rank
    return ranks


def _merge_pair(pieces: list[str], pair: tuple[str, str]) -> list[str]:
    first, second = pair
    merged = f"{first}{second}"
    out: list[str] = []
    i = 0
    count = len(pieces)
    while i < count:
        if i + 1 < count and pieces[i] == first and pieces[i + 1] == second:
            out.append(merged)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def _byte_encode(tok: Tokenizer, piece: str) -> str:
    return "".join(tok.byte_encoder[b] for b in piece.encode("utf-8"))


def _token_strings(tok: Tokenizer, text: str) -> list[str]:
    tokens: list[str] = []
    for is_added, segment in _split_added_tokens(tok, text):
        if is_added:
            tokens.append(segment)
            continue
        for piece in tok.pattern.findall(segment):
            tokens.extend(_bpe(tok, _byte_encode(tok, piece)))
    return tokens


def _split_added_tokens(tok: Tokenizer, text: str) -> list[tuple[bool, str]]:
    if not text:
        return []
    if tok.added_pattern is None:
        return [(False, text)]
    parts: list[tuple[bool, str]] = []
    last = 0
    for match in tok.added_pattern.finditer(text):
        start, end = match.span()
        if start > last:
            parts.append((False, text[last:start]))
        parts.append((True, text[start:end]))
        last = end
    if last < len(text):
        parts.append((False, text[last:]))
    return parts


def _bpe(tok: Tokenizer, token: str) -> list[str]:
    cached = tok.bpe_cache.get(token)
    if cached is not None:
        return cached
    pieces = list(token)
    while len(pieces) > 1:
        pair = _best_pair(tok, pieces)
        if pair is None:
            break
        pieces = _merge_pair(pieces, pair)
    tok.bpe_cache[token] = pieces
    return pieces


def _best_pair(tok: Tokenizer, pieces: list[str]) -> tuple[str, str] | None:
    best_pair: tuple[str, str] | None = None
    best_rank = math.inf
    for pair in itertools.pairwise(pieces):
        rank = tok.bpe_ranks.get(pair, math.inf)
        if rank < best_rank:
            best_rank = rank
            best_pair = pair
    return best_pair if best_rank != math.inf else None
