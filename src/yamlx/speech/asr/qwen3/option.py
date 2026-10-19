import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

_NUM_SPATIAL_DIMS = 3

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _coerced(
    data: Mapping[str, Any],
    fields: tuple[tuple[str, Callable[[Any], Any]], ...],
) -> dict[str, Any]:
    return {name: conv(data[name]) for name, conv in fields}


@dataclass(frozen=True)
class AudioEncoderOption:
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    num_mel_bins: int
    output_dim: int
    conv_chunksize: int
    downsample_hidden_size: int
    n_window: int
    n_window_infer: int
    max_source_positions: int
    activation_function: str

    _FIELDS = (
        ("d_model", int),
        ("encoder_layers", int),
        ("encoder_attention_heads", int),
        ("encoder_ffn_dim", int),
        ("num_mel_bins", int),
        ("output_dim", int),
        ("conv_chunksize", int),
        ("downsample_hidden_size", int),
        ("n_window", int),
        ("n_window_infer", int),
        ("max_source_positions", int),
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AudioEncoderOption:
        return cls(
            **_coerced(data, cls._FIELDS),
            activation_function=str(data["activation_function"]),
        )


@dataclass(frozen=True)
class TextDecoderOption:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    hidden_act: str
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool
    attention_bias: bool
    tie_word_embeddings: bool

    _FIELDS = (
        ("hidden_size", int),
        ("num_hidden_layers", int),
        ("num_attention_heads", int),
        ("num_key_value_heads", int),
        ("head_dim", int),
        ("intermediate_size", int),
        ("vocab_size", int),
        ("rms_norm_eps", float),
        ("rope_theta", float),
        ("max_position_embeddings", int),
        ("hidden_act", str),
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TextDecoderOption:
        rope_scaling = data.get("rope_scaling") or {}
        mrope_section = rope_scaling.get("mrope_section")
        if mrope_section is None:
            raise ValueError("text_config.rope_scaling.mrope_section is required.")
        if len(mrope_section) != _NUM_SPATIAL_DIMS:
            raise ValueError(
                "text_config.rope_scaling.mrope_section must have 3 "
                f"entries (temporal, height, width); got {len(mrope_section)}."
            )
        t_dim, h_dim, w_dim = (int(v) for v in mrope_section)
        return cls(
            **_coerced(data, cls._FIELDS),
            mrope_section=(t_dim, h_dim, w_dim),
            mrope_interleaved=bool(rope_scaling.get("mrope_interleaved", False)),
            attention_bias=bool(data["attention_bias"]),
            tie_word_embeddings=bool(data["tie_word_embeddings"]),
        )


@dataclass(frozen=True)
class Qwen3ASROption:
    encoder: AudioEncoderOption
    decoder: TextDecoderOption
    audio_start_token_id: int
    audio_end_token_id: int
    audio_token_id: int
    dtype: str
    timestamp_token_id: int | None = None
    timestamp_segment_time: float | None = None
    classify_num: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3ASROption:
        thinker = data["thinker_config"]
        timestamp_token_id = data.get("timestamp_token_id")
        timestamp_segment_time = data.get("timestamp_segment_time")
        classify_num = thinker.get("classify_num")
        return cls(
            encoder=AudioEncoderOption.from_dict(thinker["audio_config"]),
            decoder=TextDecoderOption.from_dict(thinker["text_config"]),
            audio_start_token_id=int(thinker["audio_start_token_id"]),
            audio_end_token_id=int(thinker["audio_end_token_id"]),
            audio_token_id=int(thinker["audio_token_id"]),
            dtype=str(thinker.get("dtype", "bfloat16")),
            timestamp_token_id=(
                None if timestamp_token_id is None else int(timestamp_token_id)
            ),
            timestamp_segment_time=(
                None
                if timestamp_segment_time is None
                else float(timestamp_segment_time)
            ),
            classify_num=None if classify_num is None else int(classify_num),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Qwen3ASROption:
        with Path(path).open(encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return cls.from_dict(data)
