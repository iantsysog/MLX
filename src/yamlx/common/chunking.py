import numpy as np

MAX_CHUNK_SECONDS = 30.0
MIN_CHUNK_SECONDS = 0.5
SNAP_SEARCH_SECONDS = 5.0

_RMS_WINDOW_SECONDS = 0.1
_SILENCE_EPSILON = 1e-10
_MERGE_ITERATION_LIMIT = 10_000


def split_audio_into_chunks(
    audio: np.ndarray,
    sr: int = 16000,
    max_chunk_sec: float = MAX_CHUNK_SECONDS,
    min_chunk_sec: float = MIN_CHUNK_SECONDS,
) -> list[tuple[np.ndarray, float]]:
    if sr < 1:
        raise ValueError(f"sr must be positive; got {sr}")
    if max_chunk_sec <= 0.0 or min_chunk_sec <= 0.0:
        raise ValueError("max_chunk_sec and min_chunk_sec must be positive.")
    samples = np.asarray(audio, dtype=np.float32)
    pieces = _split(samples, sr, max_chunk_sec, 0.0)
    return _merge_short_chunks(pieces, sr, max_chunk_sec, min_chunk_sec)


def _split(
    audio: np.ndarray, sr: int, max_chunk_sec: float, offset: float
) -> list[tuple[np.ndarray, float]]:
    pieces: list[tuple[np.ndarray, float]] = []
    while audio.size / sr > max_chunk_sec:
        position = _find_low_energy_split(audio, sr, int(max_chunk_sec * sr))
        pieces.extend(_split(audio[:position], sr, max_chunk_sec, offset))
        audio = audio[position:]
        offset += position / sr
    pieces.append((audio, offset))
    return pieces


def _find_low_energy_split(audio: np.ndarray, sr: int, target_sample: int) -> int:
    last = audio.size - 1
    if last <= 1:
        return max(1, last)
    window = max(1, int(_RMS_WINDOW_SECONDS * sr))
    lo = max(1, target_sample - int(SNAP_SEARCH_SECONDS * sr))
    hi = min(last, target_sample + int(SNAP_SEARCH_SECONDS * sr))
    if hi - lo < window:
        return int(min(max(target_sample, 1), last))
    starts = np.arange(lo, hi - window + 1)
    cumulative = np.concatenate((
        np.zeros(1, dtype=np.float64),
        np.cumsum(audio.astype(np.float64) ** 2),
    ))
    rms = np.sqrt((cumulative[starts + window] - cumulative[starts]) / window)
    if float(rms.max() - rms.min()) <= _SILENCE_EPSILON:
        return int(min(max(target_sample, 1), last))
    best_start = int(starts[int(np.argmin(rms))])
    return int(min(max(best_start + window // 2, 1), last))


def _merge_short_chunks(
    pieces: list[tuple[np.ndarray, float]],
    sr: int,
    max_chunk_sec: float,
    min_chunk_sec: float,
) -> list[tuple[np.ndarray, float]]:
    items = list(pieces)
    for _ in range(_MERGE_ITERATION_LIMIT):
        short = next(
            (
                i
                for i, (chunk, _) in enumerate(items)
                if chunk.size / sr < min_chunk_sec
            ),
            None,
        )
        if short is None or len(items) == 1:
            return items
        if short == 0:
            neighbor = 1
        elif short == len(items) - 1:
            neighbor = short - 1
        else:
            neighbor = (
                short - 1
                if items[short - 1][0].size <= items[short + 1][0].size
                else short + 1
            )
        first, second = min(short, neighbor), max(short, neighbor)
        chunk_a, offset_a = items[first]
        merged = np.concatenate((chunk_a, items[second][0]))
        if merged.size / sr > max_chunk_sec:
            mid = merged.size // 2
            items[first : second + 1] = [
                (merged[:mid], offset_a),
                (merged[mid:], offset_a + mid / sr),
            ]
        else:
            items[first : second + 1] = [(merged, offset_a)]
    return items
