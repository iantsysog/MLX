from typing import TYPE_CHECKING, cast, override

import mlx.core as mx
from mlx import nn

from yamlx.common.audio import log_mel_spectrogram
from yamlx.common.kv import KVCache, new_kv_cache
from yamlx.common.numeric import scalar_int

from .decoder import TextDecoder
from .encoder import AudioEncoder

if TYPE_CHECKING:
    import numpy as np

    from yamlx.speech.asr.qwen3.option import Qwen3ASROption

_TOKENS_RANK = 2
_POSITIONS_RANK = 3


class Qwen3ASRModel(nn.Module):
    def __init__(self, option: Qwen3ASROption, *, head_size: int | None = None) -> None:
        super().__init__()
        if option.encoder.output_dim != option.decoder.hidden_size:
            message = (
                f"encoder output_dim ({option.encoder.output_dim}) must "
                f"match decoder hidden_size ({option.decoder.hidden_size})"
            )
            raise ValueError(message)

        self.option = option
        self.audio_token_id = option.audio_token_id

        self.audio_tower = AudioEncoder(option.encoder)
        self.model = TextDecoder(option.decoder)

        output_size = head_size if head_size is not None else option.decoder.vocab_size
        self.lm_head = nn.Linear(option.decoder.hidden_size, output_size, bias=False)

        if head_size is None and option.decoder.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _embed_tokens(
        self, input_ids: mx.array, *, unchecked: bool = False
    ) -> mx.array:
        if not mx.issubdtype(input_ids.dtype, mx.integer):
            message = (
                "input_ids must use an integer dtype for embed_tokens; "
                f"got dtype {input_ids.dtype}."
            )
            raise ValueError(message)
        if not unchecked and input_ids.size:
            vocab_size = int(self.option.decoder.vocab_size)
            min_id = scalar_int(mx.min(input_ids))
            max_id = scalar_int(mx.max(input_ids))
            if min_id < 0 or max_id >= vocab_size:
                message = (
                    f"input_ids out of bounds for embed_tokens: "
                    f"min_token_id={min_id}, max_token_id={max_id}, "
                    f"vocab_size={vocab_size}"
                )
                raise ValueError(message)
        return cast("mx.array", self.model.embed_tokens(input_ids))

    def _forward(
        self,
        embeds: mx.array,
        position_ids: mx.array | None,
        cache: KVCache | None,
    ) -> mx.array:
        hidden = self.model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            cache=cache,
        )
        return self.lm_head(hidden)

    @override
    def __call__(
        self,
        input_ids: mx.array,
        input_features: mx.array | None = None,
        feature_lens: mx.array | None = None,
        position_ids: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        if input_features is not None:
            if feature_lens is None:
                message = "feature_lens is required when input_features is provided."
                raise ValueError(message)
            audio_features, _ = self.audio_tower(input_features, feature_lens)
            return self.forward_from_features(
                input_ids, audio_features, position_ids=position_ids, cache=cache
            )
        return self._forward(self._embed_tokens(input_ids), position_ids, cache)

    def encode_waveform(
        self, audio: np.ndarray, dtype: mx.Dtype
    ) -> tuple[mx.array, int]:
        mel = log_mel_spectrogram(audio, self.option.encoder.num_mel_bins)
        mel_batch = mx.array(mel)[None].astype(dtype)
        feature_lens = mx.array([mel.shape[1]], dtype=mx.int32)
        audio_features, _ = self.audio_tower(mel_batch, feature_lens)
        mx.eval(audio_features)
        return audio_features, int(audio_features.shape[1])

    def forward_from_features(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        position_ids: mx.array | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        embeds = self._embed_tokens(input_ids)
        audio_mask = input_ids == self.audio_token_id
        embeds = self._inject_audio_features(embeds, audio_features, audio_mask)
        return self._forward(embeds, position_ids, cache)

    @staticmethod
    def _inject_audio_features(
        embeds: mx.array,
        audio_features: mx.array,
        audio_mask: mx.array,
    ) -> mx.array:
        if (
            embeds.shape[0] != audio_features.shape[0]
            or embeds.shape[0] != audio_mask.shape[0]
            or embeds.shape[1] != audio_mask.shape[1]
        ):
            message = (
                f"Audio injection shape mismatch: embeds={embeds.shape}, "
                f"audio_features={audio_features.shape}, "
                f"audio_mask={audio_mask.shape}"
            )
            raise ValueError(message)
        if embeds.shape[2] != audio_features.shape[2]:
            message = (
                f"Audio injection channel mismatch: "
                f"embed_dim={embeds.shape[2]}, "
                f"audio_feature_dim={audio_features.shape[2]}"
            )
            raise ValueError(message)

        mask_int = audio_mask.astype(mx.int32)
        audio_counts = mx.sum(mask_int, axis=1)
        max_count = scalar_int(mx.max(audio_counts))
        if max_count == 0:
            return embeds
        if max_count > int(audio_features.shape[1]):
            message = (
                f"Too many audio placeholders: {max_count} placeholders, "
                f"but only {audio_features.shape[1]} audio feature tokens are available."
            )
            raise ValueError(message)

        cum_idx = mx.maximum(mx.cumsum(mask_int, axis=1) - 1, 0)
        gathered = mx.take_along_axis(audio_features, cum_idx[:, :, None], axis=1)
        return mx.where(audio_mask[:, :, None], gathered, embeds)

    def prefill(
        self,
        input_ids: mx.array,
        audio_features: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array:
        logits = self.forward_from_features(
            input_ids, audio_features, position_ids=position_ids, cache=cache
        )
        return logits[:, -1:, :]

    def step(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array:
        return self._forward(
            self._embed_tokens(input_ids, unchecked=True), position_ids, cache
        )

    def step_many(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        cache: KVCache,
    ) -> mx.array:
        if input_ids.ndim != _TOKENS_RANK:
            message = (
                f"input_ids for step_many must be rank-2 (B, T), "
                f"got shape={input_ids.shape}"
            )
            raise ValueError(message)
        if position_ids.ndim != _POSITIONS_RANK:
            message = (
                f"position_ids for step_many must be rank-3 (B, 3, T), "
                f"got shape={position_ids.shape}"
            )
            raise ValueError(message)
        if input_ids.shape[0] != position_ids.shape[0]:
            message = (
                f"batch mismatch in step_many: input_ids={input_ids.shape}, "
                f"position_ids={position_ids.shape}"
            )
            raise ValueError(message)
        if input_ids.shape[1] != position_ids.shape[2]:
            message = (
                f"sequence mismatch in step_many: "
                f"input_ids={input_ids.shape}, "
                f"position_ids={position_ids.shape}"
            )
            raise ValueError(message)
        return self.step(input_ids, position_ids, cache)

    def create_cache(self) -> KVCache:
        return new_kv_cache(self.option.decoder.num_hidden_layers)
