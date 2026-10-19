from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_LANGUAGE_CANONICAL: dict[str, str] = {
    alias: name
    for name, aliases in {
        "English": "en en-us en-gb english",
        "Chinese": "zh zh-cn zh-tw cmn mandarin chinese",
        "Japanese": "ja ja-jp japanese",
        "Korean": "ko ko-kr korean",
        "German": "de de-de german",
        "French": "fr fr-fr french",
        "Spanish": "es es-es es-419 spanish",
        "Russian": "ru ru-ru russian",
        "Arabic": "ar ar-eg arabic",
        "Hindi": "hi hi-in hindi",
        "Italian": "it it-it italian",
        "Portuguese": "pt pt-br pt-pt portuguese",
        "Turkish": "tr tr-tr turkish",
        "Dutch": "nl nl-nl dutch",
    }.items()
    for alias in aliases.split()
}

NO_SPACE_LANGUAGES: frozenset[str] = frozenset([
    "chinese",
    "zh",
    "zh-cn",
    "zh-tw",
    "cantonese",
    "yue",
    "japanese",
    "ja",
    "jp",
])


def _lookup_key(language: str) -> str:
    return language.strip().casefold().replace("_", "-")


def canonicalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = language.strip()
    if not value:
        return None
    return _LANGUAGE_CANONICAL.get(_lookup_key(value), value)


def is_no_space_language(language: str | None) -> bool:
    return _lookup_key(str(language or "")) in NO_SPACE_LANGUAGES


def join_text_parts(parts: Iterable[str], language: str | None) -> str:
    joiner = "" if is_no_space_language(language) else " "
    return joiner.join(part for part in (str(p).strip() for p in parts) if part)
