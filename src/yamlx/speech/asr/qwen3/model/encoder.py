import math
from itertools import pairwise
from typing import TYPE_CHECKING, cast, override

import mlx.core as mx
from mlx import nn

from yamlx.common.numeric import scalar_int

from .attention import _scaled_dot_product_attention, merge_heads, split_heads

if TYPE_CHECKING:
    from yamlx.speech.asr.qwen3.option import AudioEncoderOption

_WINDOWED_SEGMENT_MIN_WINDOWS = 20

_MAX_SINGLE_WINDOW_ENTRIES = 2

_CONV_DOWNSAMPLE = 8

_MEL_RANK = 2


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        log_timescale_increment = math.log(10000.0) / (half_dim - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(half_dim, dtype=mx.float32)
        )
        positions = mx.arange(0, num_positions, dtype=mx.float32)
        scaled_time = positions[:, None] * inv_timescales[None, :]
        pe = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
        if embedding_dim % 2 == 1:
            pe = pe[:, :embedding_dim]
        mx.eval(pe)
        self._pe = pe

    @override
    def __call__(self, positions: int | mx.array) -> mx.array:
        if isinstance(positions, int):
            return self._pe[:positions]
        return self._pe[positions]


class AudioAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    @override
    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        q = split_heads(self.q_proj(x), self.num_heads)
        k = split_heads(self.k_proj(x), self.num_heads)
        v = split_heads(self.v_proj(x), self.num_heads)
        out = _scaled_dot_product_attention(q, k, v, mask=mask)
        return self.out_proj(merge_heads(out))


class AudioEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, encoder_ffn_dim: int) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = AudioAttention(d_model, num_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(encoder_ffn_dim, d_model, bias=True)

    @override
    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        residual = x
        x = self.self_attn(self.self_attn_layer_norm(x), mask=mask)
        x = residual + x

        residual = x
        x = self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        x = residual + x

        if x.dtype == mx.float16:
            clamp_value = float(mx.finfo(mx.float16).max) - 1000.0
            x = mx.clip(x, -clamp_value, clamp_value)

        return x


class AudioEncoder(nn.Module):
    def __init__(self, option: AudioEncoderOption) -> None:
        super().__init__()
        self.option = option
        dhs = option.downsample_hidden_size

        self.conv2d1 = nn.Conv2d(1, dhs, kernel_size=3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(dhs, dhs, kernel_size=3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(dhs, dhs, kernel_size=3, stride=2, padding=1)

        freq_after_conv = option.num_mel_bins // _CONV_DOWNSAMPLE
        self.conv_out = nn.Linear(dhs * freq_after_conv, option.d_model, bias=False)

        self.embed_positions = SinusoidalPositionEmbedding(
            option.max_source_positions, option.d_model
        )

        self.layers = [
            AudioEncoderLayer(
                option.d_model,
                option.encoder_attention_heads,
                option.encoder_ffn_dim,
            )
            for _ in range(option.encoder_layers)
        ]

        self.ln_post = nn.LayerNorm(option.d_model)

        self.proj1 = nn.Linear(option.d_model, option.d_model)
        self.proj2 = nn.Linear(option.d_model, option.output_dim)

    def _apply_conv_stem(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        return nn.gelu(self.conv2d3(x))

    def get_output_lengths(self, input_lengths: mx.array) -> mx.array:
        chunk_size = self.option.n_window * 2
        tail_frames = input_lengths % chunk_size
        tail_after_conv1 = (tail_frames - 1) // 2 + 1
        tail_after_conv2 = (tail_after_conv1 - 1) // 2 + 1
        tail_tokens = (tail_after_conv2 - 1) // 2 + 1
        tail_tokens = mx.where(tail_frames > 0, tail_tokens, mx.zeros_like(tail_tokens))
        full_after_conv1 = (chunk_size - 1) // 2 + 1
        full_after_conv2 = (full_after_conv1 - 1) // 2 + 1
        full_chunk_tokens = (full_after_conv2 - 1) // 2 + 1
        return tail_tokens + (input_lengths // chunk_size) * full_chunk_tokens

    @override
    def __call__(
        self,
        input_features: mx.array,
        feature_lens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        chunk_size = self.option.n_window * 2

        all_features: list[mx.array] = []
        for b in range(input_features.shape[0]):
            feat_len = scalar_int(feature_lens[b])
            features = self._encode_single(input_features[b, :, :feat_len], chunk_size)
            all_features.append(features)

        max_len = max(f.shape[0] for f in all_features)
        output = mx.stack([_pad_to_length(feats, max_len) for feats in all_features])
        return output, mx.array([f.shape[0] for f in all_features])

    @property
    def attention_window_frames(self) -> int:
        return int(self.option.n_window_infer)

    def encode_frames(self, mel: mx.array) -> mx.array:
        if mel.ndim != _MEL_RANK:
            message = f"encode_frames expects (n_mels, n_frames); got {mel.shape}"
            raise ValueError(message)
        if int(mel.shape[1]) == 0:
            message = "encode_frames requires at least one mel frame."
            raise ValueError(message)
        return self._encode_single(mel, self.option.n_window * 2)

    def _encode_single(self, mel: mx.array, chunk_size: int) -> mx.array:
        token_lens, conv_outputs = self._convolve_chunks(mel, chunk_size)
        x = self.conv_out(mx.concatenate(conv_outputs, axis=0))
        x = self._add_positions(x, token_lens)
        x = self._run_layers(x, token_lens, chunk_size)
        return self._project(x)

    def _convolve_chunks(
        self, mel: mx.array, chunk_size: int
    ) -> tuple[list[int], list[mx.array]]:
        total_frames = mel.shape[1]
        chunk_token_lens: list[int] = []
        chunk_conv_outputs: list[mx.array] = []
        n_full_chunks = total_frames // chunk_size

        if n_full_chunks > 0:
            full_mel = mel[:, : n_full_chunks * chunk_size]
            full_mel = full_mel.reshape(
                int(mel.shape[0]), n_full_chunks, chunk_size
            ).transpose(1, 0, 2)
            x_full = self._apply_conv_stem(full_mel[:, :, :, None])
            _, freq_down, time_down, channels = x_full.shape
            x_full = x_full.transpose(0, 2, 3, 1)
            x_full = x_full.reshape(n_full_chunks, time_down, channels * freq_down)
            chunk_conv_outputs.append(
                x_full.reshape(n_full_chunks * time_down, channels * freq_down)
            )
            chunk_token_lens.extend([int(time_down)] * n_full_chunks)

        if n_full_chunks * chunk_size < total_frames:
            chunk_mel = mel[:, n_full_chunks * chunk_size :]
            valid_tail_tokens = (
                int(chunk_mel.shape[1]) + _CONV_DOWNSAMPLE - 1
            ) // _CONV_DOWNSAMPLE
            if n_full_chunks > 0:
                chunk_mel = mx.pad(
                    chunk_mel, [(0, 0), (0, chunk_size - int(chunk_mel.shape[1]))]
                )
            x_tail = self._apply_conv_stem(chunk_mel[None, :, :, None])
            x_tail = x_tail[:, :, :valid_tail_tokens, :]
            _, freq_down, time_down, channels = x_tail.shape
            x_tail = x_tail.transpose(0, 2, 3, 1).reshape(
                1, time_down, channels * freq_down
            )
            chunk_token_lens.append(int(time_down))
            chunk_conv_outputs.append(x_tail[0])

        return chunk_token_lens, chunk_conv_outputs

    def _add_positions(self, x: mx.array, token_lens: list[int]) -> mx.array:
        pe = self.embed_positions(max(token_lens))
        pe_full = mx.concatenate([pe[:n] for n in token_lens], axis=0)
        return x + pe_full.astype(x.dtype)

    def _run_layers(
        self, x: mx.array, token_lens: list[int], chunk_size: int
    ) -> mx.array:
        total_tokens = x.shape[0]
        tokens_per_window = token_lens[0] * (self.option.n_window_infer // chunk_size)

        cu_seqlens = [0]
        pos = 0
        while pos < total_tokens:
            pos = min(pos + tokens_per_window, total_tokens)
            cu_seqlens.append(pos)

        x = x[None, :, :]
        if len(cu_seqlens) - 1 >= _WINDOWED_SEGMENT_MIN_WINDOWS:
            x = _apply_windowed_encoder_layers(x, self.layers, cu_seqlens)
        else:
            mask = _create_windowed_mask(total_tokens, cu_seqlens, x.dtype)
            for layer in self.layers:
                x = layer(x, mask=mask)
        return x[0]

    def _project(self, x: mx.array) -> mx.array:
        x = cast("mx.array", self.ln_post(x))
        x = nn.gelu(self.proj1(x))
        return self.proj2(x)


def _pad_to_length(feats: mx.array, target_len: int) -> mx.array:
    if feats.shape[0] >= target_len:
        return feats
    pad_arr = mx.zeros(
        (target_len - feats.shape[0], feats.shape[-1]), dtype=feats.dtype
    )
    return mx.concatenate([feats, pad_arr], axis=0)


def _create_windowed_mask(
    seq_len: int,
    cu_seqlens: list[int],
    dtype: mx.Dtype = mx.float32,
) -> mx.array | None:
    if len(cu_seqlens) <= _MAX_SINGLE_WINDOW_ENTRIES:
        return None

    boundaries = mx.array(cu_seqlens[1:-1])
    positions = mx.arange(seq_len)
    window_ids = mx.sum(positions[:, None] >= boundaries[None, :], axis=1)
    same_window = window_ids[:, None] == window_ids[None, :]
    masked_value = float(mx.finfo(dtype).min)
    mask = mx.where(
        same_window,
        mx.array(0.0, dtype=dtype),
        mx.array(masked_value, dtype=dtype),
    )
    return mask[None, None, :, :]


def _apply_windowed_encoder_layers(
    x: mx.array,
    layers: list[AudioEncoderLayer],
    cu_seqlens: list[int],
) -> mx.array:
    if len(cu_seqlens) <= _MAX_SINGLE_WINDOW_ENTRIES:
        for layer in layers:
            x = layer(x, mask=None)
        return x

    for layer in layers:
        parts: list[mx.array] = [
            layer(x[:, start:end, :], mask=None) for start, end in pairwise(cu_seqlens)
        ]
        x = mx.concatenate(parts, axis=1)
    return x
