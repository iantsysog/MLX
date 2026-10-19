from typing import TYPE_CHECKING, cast, override

import mlx.core as mx
from mlx import nn

from yamlx.common.kv import kv_update

from .attention import _scaled_dot_product_attention, merge_heads, split_heads
from .mrope import apply_mrope, apply_rotary_pos_emb, build_mrope

if TYPE_CHECKING:
    from yamlx.common.kv import KVCache
    from yamlx.speech.asr.qwen3.option import TextDecoderOption


class TextAttention(nn.Module):
    def __init__(self, option: TextDecoderOption) -> None:
        super().__init__()
        self.hidden_size = option.hidden_size
        self.num_heads = option.num_attention_heads
        self.num_kv_heads = option.num_key_value_heads
        self.head_dim = option.head_dim

        self.q_proj = nn.Linear(
            option.hidden_size,
            self.num_heads * self.head_dim,
            bias=option.attention_bias,
        )
        self.k_proj = nn.Linear(
            option.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=option.attention_bias,
        )
        self.v_proj = nn.Linear(
            option.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=option.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            option.hidden_size,
            bias=option.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    @override
    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        q = self.q_norm(split_heads(self.q_proj(x), self.num_heads))
        k = self.k_norm(split_heads(self.k_proj(x), self.num_kv_heads))
        v = split_heads(self.v_proj(x), self.num_kv_heads)

        cos, sin = rope
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = kv_update(cache, k, v, layer_idx)

        out = _scaled_dot_product_attention(q, k, v, mask=mask)
        return self.o_proj(merge_heads(out))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    @override
    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    def __init__(self, option: TextDecoderOption) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(option.hidden_size, eps=option.rms_norm_eps)
        self.self_attn = TextAttention(option)
        self.post_attention_layernorm = nn.RMSNorm(
            option.hidden_size, eps=option.rms_norm_eps
        )
        self.mlp = SwiGLU(option.hidden_size, option.intermediate_size)

    @override
    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> mx.array:
        residual = x
        x = self.self_attn(
            self.input_layernorm(x), rope, mask=mask, cache=cache, layer_idx=layer_idx
        )
        x = residual + x

        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        return residual + x


class TextDecoder(nn.Module):
    def __init__(self, option: TextDecoderOption) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(option.vocab_size, option.hidden_size)
        self.layers = [
            TextDecoderLayer(option) for _ in range(option.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(option.hidden_size, eps=option.rms_norm_eps)

        self.mrope = build_mrope(
            head_dim=option.head_dim,
            base=option.rope_theta,
            mrope_section=option.mrope_section,
            interleaved=option.mrope_interleaved,
        )

    @override
    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        if inputs_embeds is not None:
            h = inputs_embeds
        elif input_ids is not None:
            h = self.embed_tokens(input_ids)
        else:
            message = "Either input_ids or inputs_embeds must be provided."
            raise ValueError(message)

        if position_ids is None:
            message = "position_ids must be provided for text decoding."
            raise ValueError(message)

        rope = apply_mrope(self.mrope, position_ids, dtype=h.dtype)

        mask = _attention_mask(h, cache)
        for i, layer in enumerate(self.layers):
            h = layer(h, rope, mask=mask, cache=cache, layer_idx=i)

        return cast("mx.array", self.norm(h))


def _attention_mask(h: mx.array, cache: KVCache | None) -> mx.array | None:
    seq_len = h.shape[1]
    if cache is not None and cache.offset > 0:
        if seq_len == 1:
            return None
        return _create_causal_mask_with_prefix(seq_len, cache.offset, h.dtype)
    return _create_causal_mask(seq_len, h.dtype)


def _masked_causal_block(seq_len: int, dtype: mx.Dtype) -> mx.array:
    masked = mx.full((seq_len, seq_len), float(mx.finfo(dtype).min), dtype=dtype)
    return mx.triu(masked, k=1)


def _create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    return _masked_causal_block(seq_len, dtype)[None, None, :, :]


def _create_causal_mask_with_prefix(
    seq_len: int,
    prefix_len: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    prefix = mx.zeros((seq_len, prefix_len), dtype=dtype)
    return mx.concatenate([prefix, _masked_causal_block(seq_len, dtype)], axis=1)[
        None, None, :, :
    ]
