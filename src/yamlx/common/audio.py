import math
import stat
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from yamlx.common.errors import AudioDecodeError

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
NUM_MEL_BINS = 128
MAX_FILE_BYTES = 2 * 1024**3

_MAX_AUDIO_DIMS = 2
_LOG_FLOOR = 1e-10
_LOG_MAX_DROP = 8.0
_LOG_SCALE = 4.0
_LOG_OFFSET = 4.0
_RESAMPLE_HALF_TAPS = 32
_CONVOLVE_BLOCK = 1 << 21

type AudioSource = str | Path | np.ndarray | tuple[np.ndarray, int]


def load_audio(source: AudioSource, sr: int = SAMPLE_RATE) -> np.ndarray:
    if isinstance(source, tuple):
        samples, orig_sr = source
        sanitized = sanitize_pcm(samples)
        if orig_sr == sr:
            return sanitized
        return _resample(sanitized, orig_sr, sr)
    if isinstance(source, np.ndarray):
        return sanitize_pcm(source)
    if isinstance(source, str | Path):
        return _load_audio_file(Path(source), sr)
    msg = f"Unsupported audio source type: {type(source).__name__}"
    raise ValueError(msg)


def sanitize_pcm(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim == 0 or arr.ndim > _MAX_AUDIO_DIMS:
        msg = f"PCM arrays must be 1-D or 2-D; got shape {arr.shape}"
        raise ValueError(msg)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        arr = _normalize_integer_pcm(arr)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim == _MAX_AUDIO_DIMS:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        arr = arr.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _normalize_integer_pcm(arr: np.ndarray) -> np.ndarray:
    info = np.iinfo(arr.dtype)
    values = arr.astype(np.float32)
    if info.min >= 0:
        midpoint = float(info.max + 1) / 2.0
        return (values - midpoint) / midpoint
    scale = float(max(abs(int(info.min)), int(info.max)))
    return values / scale


def _load_audio_file(path: Path, sr: int) -> np.ndarray:
    _validate_input_file(path)
    try:
        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:
        decoded = _try_decode_container(path, sr)
        if decoded is not None:
            return decoded
        raise AudioDecodeError(
            str(path),
            f"Cannot decode audio: {exc}. For video containers "
            "(mp4, mkv, mov), install PyAV: pip install yamlx[av]",
        ) from exc
    mono = _mixdown(data)
    if mono.size == 0:
        return mono
    if orig_sr != sr:
        return _resample(mono, orig_sr, sr)
    return mono


def _try_decode_container(path: Path, sr: int) -> np.ndarray | None:
    try:
        import av
    except ImportError:
        return None
    try:
        from av.audio.resampler import AudioResampler

        chunks: list[np.ndarray] = []
        with av.open(str(path)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = AudioResampler(format="flt", layout="mono", rate=sr)
            for frame in container.decode(stream):
                if not isinstance(frame, av.AudioFrame):
                    continue
                chunks.extend(
                    resampled.to_ndarray().reshape(-1)
                    for resampled in resampler.resample(frame)
                )
            chunks.extend(
                resampled.to_ndarray().reshape(-1)
                for resampled in resampler.resample(None)
            )
    except av.error.FFmpegError, ValueError, StopIteration:
        return None
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def _mixdown(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        return np.ascontiguousarray(data[:, 0], dtype=np.float32)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)


def _validate_input_file(path: Path) -> None:
    try:
        file_stat = path.stat()
    except OSError as exc:
        detail = f"Cannot access input file: {exc.strerror or exc}"
        raise AudioDecodeError(str(path), detail) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        detail = "Not a regular file (FIFOs, devices, and directories are rejected)."
        raise AudioDecodeError(str(path), detail)
    if file_stat.st_size > MAX_FILE_BYTES:
        detail = f"File exceeds the {MAX_FILE_BYTES // (1024**3)} GB decode cap."
        raise AudioDecodeError(str(path), detail)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if samples.size == 0 or orig_sr == target_sr:
        return np.ascontiguousarray(samples, dtype=np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    cutoff = min(1.0 / (2.0 * up), 1.0 / (2.0 * down))
    half = _RESAMPLE_HALF_TAPS * max(up, down)
    n_taps = 2 * half + 1
    t = np.arange(n_taps, dtype=np.float64) - half
    kernel = up * 2.0 * cutoff * np.sinc(2.0 * cutoff * t) * np.blackman(n_taps)
    if up > 1:
        upsampled = np.zeros(samples.size * up, dtype=np.float64)
        upsampled[::up] = samples
    else:
        upsampled = samples.astype(np.float64)
    full = _fft_convolve(upsampled, kernel)
    out = full[half::down]
    target_len = max(1, round(samples.size * (target_sr / orig_sr)))
    if out.size >= target_len:
        out = out[:target_len]
    else:
        out = np.pad(out, (0, target_len - out.size), mode="edge")
    return np.ascontiguousarray(out, dtype=np.float32)


def _fft_convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    total = x.size + kernel.size - 1
    out = np.zeros(total, dtype=np.float64)
    spectra: dict[int, np.ndarray] = {}
    for start in range(0, x.size, _CONVOLVE_BLOCK):
        chunk = x[start : start + _CONVOLVE_BLOCK]
        n = chunk.size + kernel.size - 1
        nfft = 1 << (n - 1).bit_length()
        spectrum = spectra.get(nfft)
        if spectrum is None:
            spectrum = np.fft.rfft(kernel, nfft)
            spectra[nfft] = spectrum
        seg = np.fft.irfft(np.fft.rfft(chunk, nfft) * spectrum, nfft)[:n]
        out[start : start + n] += seg
    return out


def _slaney_mel_from_hz(freqs: np.ndarray) -> np.ndarray:
    linear_slope = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_slope
    logstep = math.log(6.4) / 27.0
    values = np.asarray(freqs, dtype=np.float64)
    linear = values / linear_slope
    logarithmic = (
        min_log_mel + np.log(np.maximum(values, min_log_hz) / min_log_hz) / logstep
    )
    return np.where(values < min_log_hz, linear, logarithmic)


def _slaney_hz_from_mel(mels: np.ndarray) -> np.ndarray:
    linear_slope = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / linear_slope
    logstep = math.log(6.4) / 27.0
    values = np.asarray(mels, dtype=np.float64)
    linear = values * linear_slope
    logarithmic = min_log_hz * np.exp(logstep * (values - min_log_mel))
    return np.where(values < min_log_mel, linear, logarithmic)


@lru_cache(maxsize=8)
def mel_filters(
    n_mels: int = NUM_MEL_BINS,
    *,
    n_fft: int = N_FFT,
    sr: int = SAMPLE_RATE,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    top_hz = sr / 2.0 if fmax is None else fmax
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_edges = _slaney_mel_from_hz(np.array([fmin, top_hz]))
    mel_points = np.linspace(mel_edges[0], mel_edges[1], n_mels + 2)
    hz_points = _slaney_hz_from_mel(mel_points)
    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    return (weights * enorm[:, None]).astype(np.float32)


@lru_cache(maxsize=4)
def _hann_window(n_fft: int) -> np.ndarray:
    return np.hanning(n_fft + 1)[:-1].astype(np.float32)


def log_mel_spectrogram(
    samples: np.ndarray,
    n_mels: int = NUM_MEL_BINS,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        msg = "Cannot compute a mel spectrogram from empty audio."
        raise ValueError(msg)
    pad = n_fft // 2
    if audio.size <= pad:
        msg = (
            f"Audio too short for the mel front-end: {audio.size} samples; "
            f"more than {pad} required."
        )
        raise ValueError(msg)
    padded = np.pad(audio, pad, mode="reflect")
    n_frames = 1 + (padded.size - n_fft) // hop_length
    frame_stride = padded.strides[0] * hop_length
    frames = np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, n_fft), strides=(frame_stride, padded.strides[0])
    )
    spectrum = np.fft.rfft(frames * _hann_window(n_fft))
    power = (np.abs(spectrum) ** 2).T
    mel = mel_filters(n_mels, n_fft=n_fft, sr=sr) @ power
    log_spec = np.log10(np.maximum(mel, _LOG_FLOOR))[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - _LOG_MAX_DROP)
    log_spec = (log_spec + _LOG_OFFSET) / _LOG_SCALE
    return log_spec.astype(np.float32)
