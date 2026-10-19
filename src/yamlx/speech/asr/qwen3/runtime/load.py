import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from yamlx.common.errors import (
    IncompleteDownloadError,
    ModelNotCachedError,
    WeightsValidationError,
)
from yamlx.speech.asr.qwen3.model.model import Qwen3ASRModel
from yamlx.speech.asr.qwen3.option import Qwen3ASROption
from yamlx.speech.asr.qwen3.presets import MODEL_DTYPE_NAMES, QUANTIZATION_CHOICES

if TYPE_CHECKING:
    from collections.abc import Sequence

_THINKER_PREFIX = "thinker."

_BITS_PER_WORD = 32
_DEFAULT_GROUP_SIZE = 64
_SUPPORTED_BITS = (2, 4, 8)
_SUPPORTED_GROUP_SIZES = (32, 64, 128)
_HASH_CHUNK_SIZE = 1024 * 1024

_CONV_WEIGHT_RANK = 4
_MATRIX_RANK = 2

MODEL_DTYPES: dict[str, mx.Dtype] = dict(
    zip(MODEL_DTYPE_NAMES, (mx.float16, mx.float32, mx.bfloat16), strict=True)
)
_QUANTIZED_FLOAT_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


def _dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def load_model(
    source: str | Path,
    *,
    dtype: mx.Dtype = mx.float16,
    quantization: str | None = None,
    expected_sha256: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Qwen3ASRModel, Qwen3ASROption]:
    return _load_checkpoint(
        source,
        dtype=dtype,
        quantization=quantization,
        expected_sha256=expected_sha256,
        cache_dir=cache_dir,
    )


def load_forced_aligner(
    source: str | Path,
    *,
    dtype: mx.Dtype = mx.float16,
    quantization: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Qwen3ASRModel, Qwen3ASROption]:
    return _load_checkpoint(
        source,
        dtype=dtype,
        quantization=quantization,
        cache_dir=cache_dir,
        aligner=True,
    )


def _load_checkpoint(
    source: str | Path,
    *,
    dtype: mx.Dtype = mx.float16,
    quantization: str | None = None,
    expected_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    aligner: bool = False,
) -> tuple[Qwen3ASRModel, Qwen3ASROption]:
    if quantization is not None and quantization not in QUANTIZATION_CHOICES:
        raise ValueError(
            f"Unsupported quantization '{quantization}'. "
            f"Expected one of: {QUANTIZATION_CHOICES}."
        )

    model_path = resolve_model_path(source, cache_dir=cache_dir)
    option = Qwen3ASROption.from_json_file(model_path / "config.json")
    head_size: int | None = None
    if aligner:
        if option.classify_num is None:
            raise ValueError(
                "Forced-aligner checkpoints require "
                "thinker_config.classify_num in config.json."
            )
        head_size = option.classify_num

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise WeightsValidationError(str(model_path), "No safetensors files found.")

    _verify_sha256(shards, expected_sha256)
    weights = {
        key.removeprefix(_THINKER_PREFIX): value
        for key, value in _load_safetensors(shards, str(model_path)).items()
    }
    checkpoint_had_lm_head_scales = "lm_head.scales" in weights
    _materialize_tied_lm_head(weights, tie=option.decoder.tie_word_embeddings)

    model = Qwen3ASRModel(option, head_size=head_size)
    _fix_conv_layouts(model, weights)
    if any(key.endswith(".scales") for key in weights):
        if quantization is not None:
            raise ValueError(
                "Checkpoint is already quantized. Omit the quantization "
                "argument to load it as-is."
            )
        _quantize_model_for_checkpoint(model, weights, str(model_path))

    _validate_weights(model, weights, option.dtype, str(model_path))
    model.load_weights(list(weights.items()))
    _cast_floating_params(model, dtype)
    if quantization is not None:
        _quantize_on_the_fly(model, quantization)
    _retie_lm_head(
        model,
        option,
        checkpoint_had_lm_head_scales=checkpoint_had_lm_head_scales,
    )

    mx.eval(model.parameters())
    model.eval()
    return model, option


def resolve_model_path(
    source: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    candidate = Path(source)
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate
    return _resolve_hf_snapshot(str(source), _cache_root(cache_dir))


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    hub = os.environ.get("HF_HUB_CACHE") or ""
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME") or ""
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_hf_snapshot(repo_id: str, cache_root: Path) -> Path:
    model_dir = cache_root / f"models--{repo_id.replace('/', '--')}"

    def _not_cached() -> NoReturn:
        incomplete = sorted(
            blob.name for blob in (model_dir / "blobs").glob("*.incomplete")
        )
        if incomplete:
            raise IncompleteDownloadError(repo_id, str(cache_root), incomplete[0])
        raise ModelNotCachedError(repo_id, str(cache_root))

    try:
        commit = (model_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        commit = ""
    snapshot = model_dir / "snapshots" / commit if commit else None
    if snapshot is None or not snapshot.is_dir():
        _not_cached()
    shards = sorted(snapshot.glob("*.safetensors"))
    if (
        not (snapshot / "config.json").is_file()
        or not shards
        or not all(shard.is_file() for shard in shards)
    ):
        _not_cached()
    return snapshot


def _load_safetensors(shards: Sequence[Path], model_path: str) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for shard in shards:
        try:
            loaded = mx.load(str(shard))
        except RuntimeError as error:
            raise WeightsValidationError(
                model_path, f"Cannot read safetensors file {shard}: {error}."
            ) from error
        if not isinstance(loaded, dict):
            raise WeightsValidationError(model_path, f"Unexpected contents in {shard}.")
        weights.update(loaded)
    return weights


def _verify_sha256(shards: Sequence[Path], expected: str | None) -> None:
    if expected is None:
        return
    if len(shards) != 1:
        raise ValueError(
            "expected_sha256 verification requires a single shard; "
            f"found {len(shards)} safetensors files."
        )
    digest = hashlib.sha256()
    with shards[0].open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    actual = digest.hexdigest()
    normalized = expected.strip().lower()
    if actual != normalized:
        raise ValueError(
            f"SHA-256 mismatch for {shards[0]}: expected {normalized}, got {actual}."
        )


def _materialize_tied_lm_head(weights: dict[str, mx.array], *, tie: bool) -> None:
    if (
        not tie
        or "lm_head.weight" in weights
        or "model.embed_tokens.weight" not in weights
    ):
        return
    for suffix in ("weight", "scales", "biases"):
        if (key := f"model.embed_tokens.{suffix}") in weights:
            weights[f"lm_head.{suffix}"] = weights[key]


def _flatten_params(model: Qwen3ASRModel) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def _fix_conv_layouts(model: Qwen3ASRModel, weights: dict[str, mx.array]) -> None:
    model_params = _flatten_params(model)
    for key, value in weights.items():
        if (
            "conv2d" not in key
            or not key.endswith(".weight")
            or value.ndim != _CONV_WEIGHT_RANK
            or key not in model_params
        ):
            continue
        expected = model_params[key].shape
        if value.shape != expected:
            transposed = value.transpose(0, 2, 3, 1)
            if transposed.shape == expected:
                weights[key] = transposed


def _flatten_leaf_modules(tree: object, prefix: str = "") -> dict[str, nn.Module]:
    flat: dict[str, nn.Module] = {}
    if isinstance(tree, dict):
        entries: list[tuple[str, object]] = [
            (str(key), value) for key, value in tree.items()
        ]
    elif isinstance(tree, list):
        entries = [(str(index), value) for index, value in enumerate(tree)]
    else:
        return flat
    for key, value in entries:
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, nn.Module):
            flat[path] = value
        else:
            flat.update(_flatten_leaf_modules(value, path))
    return flat


def _quantize_model_for_checkpoint(
    model: Qwen3ASRModel,
    weights: dict[str, mx.array],
    model_path: str,
) -> None:
    modules = _flatten_leaf_modules(model.leaf_modules())
    groups: dict[tuple[int, int], list[str]] = {}
    for key in weights:
        if not key.endswith(".scales"):
            continue
        module_path = key[: -len(".scales")]
        module = modules.get(module_path)
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            raise WeightsValidationError(
                model_path, f"Quantization scales for unknown module '{module_path}'."
            )
        params = _infer_quant_params(module_path, module, weights, model_path)
        groups.setdefault(params, []).append(module_path)

    for (bits, group_size), paths in sorted(groups.items()):
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            class_predicate=lambda path, _module, _paths=paths: path in _paths,
        )


def _infer_quant_params(
    module_path: str,
    module: nn.Linear | nn.Embedding,
    weights: dict[str, mx.array],
    model_path: str,
) -> tuple[int, int]:
    weight = weights.get(f"{module_path}.weight")
    scales = weights.get(f"{module_path}.scales")
    if weight is None or scales is None:
        raise WeightsValidationError(
            model_path, f"Missing packed weight or scales for '{module_path}'."
        )
    input_dim = int(module.weight.shape[1])
    packed_cols = int(weight.shape[1])
    scale_cols = int(scales.shape[1])
    shape_ok = (
        weight.ndim == _MATRIX_RANK
        and scales.ndim == _MATRIX_RANK
        and input_dim > 0
        and packed_cols > 0
        and scale_cols > 0
    )
    if (
        not shape_ok
        or packed_cols * _BITS_PER_WORD % input_dim != 0
        or input_dim % scale_cols != 0
    ):
        raise WeightsValidationError(
            model_path, f"Cannot infer quantization parameters for '{module_path}'."
        )
    bits = packed_cols * _BITS_PER_WORD // input_dim
    group_size = input_dim // scale_cols
    if bits not in _SUPPORTED_BITS or group_size not in _SUPPORTED_GROUP_SIZES:
        raise WeightsValidationError(
            model_path,
            f"Unsupported quantization for '{module_path}': "
            f"bits={bits}, group_size={group_size}.",
        )
    return bits, group_size


def _validate_weights(
    model: Qwen3ASRModel,
    weights: dict[str, mx.array],
    option_dtype: str,
    model_path: str,
) -> None:
    model_params = _flatten_params(model)
    if unexpected := sorted(set(weights) - set(model_params)):
        raise WeightsValidationError(model_path, f"Unexpected keys: {unexpected}.")
    if missing := sorted(set(model_params) - set(weights)):
        raise WeightsValidationError(model_path, f"Missing keys: {missing}.")
    for key in sorted(weights):
        expected_shape = tuple(model_params[key].shape)
        actual_shape = tuple(weights[key].shape)
        if actual_shape != expected_shape:
            raise WeightsValidationError(
                model_path,
                f"Shape mismatch for '{key}': "
                f"checkpoint {actual_shape}, model {expected_shape}.",
            )
    if any(key.endswith(".scales") for key in weights):
        _validate_quantized_dtypes(weights, model_path)
    else:
        _validate_plain_dtypes(weights, option_dtype, model_path)


def _validate_plain_dtypes(
    weights: dict[str, mx.array], option_dtype: str, model_path: str
) -> None:
    expected = MODEL_DTYPES.get(option_dtype)
    if expected is None:
        raise WeightsValidationError(
            model_path,
            f"Unsupported checkpoint dtype in config.json: {option_dtype!r}.",
        )
    for key in sorted(weights):
        if weights[key].dtype != expected:
            raise WeightsValidationError(
                model_path,
                f"'{key}' must be {_dtype_name(expected)}; "
                f"got {_dtype_name(weights[key].dtype)}.",
            )


def _validate_quantized_dtypes(weights: dict[str, mx.array], model_path: str) -> None:
    float_dtypes: set[mx.Dtype] = set()
    for key in sorted(weights):
        value = weights[key]
        module_path, _, leaf = key.rpartition(".")
        if leaf == "weight" and f"{module_path}.scales" in weights:
            if value.dtype != mx.uint32:
                raise WeightsValidationError(
                    model_path,
                    f"Packed weight '{key}' must be uint32; "
                    f"got {_dtype_name(value.dtype)}.",
                )
            continue
        if not mx.issubdtype(value.dtype, mx.floating):
            raise WeightsValidationError(
                model_path,
                f"'{key}' must be a floating tensor; got {_dtype_name(value.dtype)}.",
            )
        float_dtypes.add(value.dtype)
    if len(float_dtypes) > 1:
        names = sorted(_dtype_name(dtype) for dtype in float_dtypes)
        raise WeightsValidationError(
            model_path, f"Mixed floating dtypes in quantized checkpoint: {names}."
        )
    if float_dtypes and next(iter(float_dtypes)) not in _QUANTIZED_FLOAT_DTYPES:
        only = _dtype_name(next(iter(float_dtypes)))
        raise WeightsValidationError(
            model_path,
            f"Unsupported floating dtype in quantized checkpoint: {only}.",
        )


def _cast_floating_params(model: Qwen3ASRModel, dtype: mx.Dtype) -> None:
    params = _flatten_params(model)
    if not any(
        mx.issubdtype(value.dtype, mx.floating) and value.dtype != dtype
        for value in params.values()
    ):
        return
    model.load_weights([
        (
            key,
            value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value,
        )
        for key, value in params.items()
    ])


def _quantize_on_the_fly(model: Qwen3ASRModel, quantization: str) -> None:
    if quantization == "8-bit":
        nn.quantize(model, group_size=_DEFAULT_GROUP_SIZE, bits=8)
        return
    for bits, in_audio in ((4, False), (8, True)):
        nn.quantize(
            model,
            group_size=_DEFAULT_GROUP_SIZE,
            bits=bits,
            class_predicate=lambda path, module, in_audio=in_audio: (
                isinstance(module, (nn.Linear, nn.Embedding))
                and path.startswith("audio_tower") == in_audio
            ),
        )


def _retie_lm_head(
    model: Qwen3ASRModel,
    option: Qwen3ASROption,
    *,
    checkpoint_had_lm_head_scales: bool,
) -> None:
    if not option.decoder.tie_word_embeddings or checkpoint_had_lm_head_scales:
        return
    embedding = model.model.embed_tokens
    lm_head = model.lm_head
    if isinstance(embedding, nn.QuantizedEmbedding) and isinstance(
        lm_head, nn.QuantizedLinear
    ):
        lm_head.scales = embedding.scales
        lm_head.biases = embedding.biases
    lm_head.weight = embedding.weight
