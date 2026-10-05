from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

Tensor = torch.Tensor
CompactedCache = tuple[tuple[Tensor, Tensor, Tensor], ...]
FIXED_POINT_ATOL = 1e-7
FIXED_POINT_RTOL = 1e-7


@dataclass(frozen=True)
class TensorQuantizationStats:
    bits: int
    num_values: int
    num_scales: int
    payload_bytes: int
    metadata_bytes: int
    mse: float
    storage_is_simulated: bool
    compute_dtype: str

    @property
    def effective_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes


@dataclass(frozen=True)
class CacheQuantizationStats:
    key_bits: int
    value_bits: int
    group_size: int
    quantizer: str
    key_effective_bytes: int
    value_effective_bytes: int
    key_payload_bytes: int
    value_payload_bytes: int
    key_scale_bytes: int
    value_scale_bytes: int
    beta_bytes: int
    actual_tensor_bytes: int
    key_mse: float
    value_mse: float
    num_stored_entries: int
    min_layer_length: int
    max_layer_length: int
    storage_is_simulated: bool
    compute_dtype: str
    quantization_fixed_point_verified: bool
    quantization_fixed_point_max_abs_error: float
    quantization_fixed_point_tolerance: float

    @property
    def effective_packed_bytes(self) -> int:
        return self.key_effective_bytes + self.value_effective_bytes + self.beta_bytes

    @property
    def payload_bytes(self) -> int:
        return self.key_payload_bytes + self.value_payload_bytes

    @property
    def scale_metadata_bytes(self) -> int:
        return self.key_scale_bytes + self.value_scale_bytes

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "payload_bytes": self.payload_bytes,
            "scale_metadata_bytes": self.scale_metadata_bytes,
            "effective_packed_bytes": self.effective_packed_bytes,
            "logical_packed_occupancy_bytes": self.effective_packed_bytes,
            "resident_qdq_occupancy_bytes": self.actual_tensor_bytes,
        }


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _fake_quantize_last_dim(
    tensor: Tensor,
    bits: int,
    group_size: int,
) -> tuple[Tensor, TensorQuantizationStats]:
    if bits == 16:
        byte_count = _tensor_bytes(tensor)
        return tensor, TensorQuantizationStats(
            bits=bits,
            num_values=tensor.numel(),
            num_scales=0,
            payload_bytes=byte_count,
            metadata_bytes=0,
            mse=0.0,
            storage_is_simulated=False,
            compute_dtype=str(tensor.dtype),
        )
    if bits not in (8, 4):
        raise ValueError(f"Only BF16, INT8 and INT4 are supported, got {bits} bits.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    original_dtype = tensor.dtype
    original_shape = tensor.shape
    last_dim = original_shape[-1]
    padding = (-last_dim) % group_size
    work = tensor.float()
    if padding:
        work = torch.nn.functional.pad(work, (0, padding))

    grouped = work.reshape(*work.shape[:-1], -1, group_size)
    qmax = (1 << (bits - 1)) - 1
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(grouped / scales).clamp(-qmax, qmax)
    dequantized = (quantized * scales).reshape(*work.shape)
    if padding:
        dequantized = dequantized[..., :last_dim]
    dequantized = dequantized.reshape(original_shape).to(original_dtype)

    mse = torch.mean((dequantized.float() - tensor.float()) ** 2).item()
    num_scales = scales.numel()
    payload_bytes = math.ceil(tensor.numel() * bits / 8)
    metadata_bytes = num_scales * 2  # one BF16 scale per group
    return dequantized, TensorQuantizationStats(
        bits=bits,
        num_values=tensor.numel(),
        num_scales=num_scales,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        mse=mse,
        storage_is_simulated=True,
        compute_dtype=str(dequantized.dtype),
    )


def fake_quantize_key(
    keys: Tensor,
    bits: int,
    group_size: int,
    quantizer: str,
) -> tuple[Tensor, TensorQuantizationStats]:
    if quantizer == "per_vector_symmetric":
        return _fake_quantize_last_dim(keys, bits, group_size)
    if quantizer == "kivi_layout_symmetric":
        # KIVI's key layout groups along the token axis for each channel. This
        # remains symmetric QDQ, so it is a layout-compatible approximation,
        # not a claim to reproduce the complete KIVI algorithm.
        transposed = keys.transpose(-1, -2).contiguous()
        dequantized, stats = _fake_quantize_last_dim(transposed, bits, group_size)
        return dequantized.transpose(-1, -2).contiguous(), stats
    raise ValueError(f"Unknown KV quantizer: {quantizer}")


def fake_quantize_value(
    values: Tensor,
    bits: int,
    group_size: int,
    quantizer: str,
) -> tuple[Tensor, TensorQuantizationStats]:
    if quantizer in ("per_vector_symmetric", "kivi_layout_symmetric"):
        return _fake_quantize_last_dim(values, bits, group_size)
    raise ValueError(f"Unknown KV quantizer: {quantizer}")


def dense_cache_to_compacted(past_key_values: object) -> CompactedCache:
    if hasattr(past_key_values, "key_cache"):
        layers: Iterable[tuple[Tensor, Tensor]] = zip(
            past_key_values.key_cache,
            past_key_values.value_cache,
        )
    elif hasattr(past_key_values, "layers"):
        layers = (
            (layer.keys, layer.values)
            for layer in past_key_values.layers
            if getattr(layer, "is_initialized", True)
        )
    else:
        layers = past_key_values

    compacted = []
    for keys, values in layers:
        beta = torch.zeros(
            keys.shape[0],
            keys.shape[1],
            keys.shape[2],
            device=keys.device,
            dtype=keys.dtype,
        )
        compacted.append((keys, beta, values))
    if not compacted:
        raise ValueError("The model returned an empty KV cache.")
    return tuple(compacted)


def quantize_compacted_cache(
    cache: CompactedCache,
    key_bits: int,
    value_bits: int,
    group_size: int = 64,
    quantizer: str = "kivi_layout_symmetric",
) -> tuple[CompactedCache, CacheQuantizationStats]:
    quantized_layers = []
    key_stats: list[TensorQuantizationStats] = []
    value_stats: list[TensorQuantizationStats] = []
    beta_bytes = 0
    actual_tensor_bytes = 0
    num_stored_entries = 0
    layer_lengths = []
    fixed_point_max_abs_error = 0.0
    fixed_point_reference_max_abs = 0.0

    for keys, beta, values in cache:
        dequantized_keys, current_key_stats = fake_quantize_key(
            keys, key_bits, group_size, quantizer
        )
        dequantized_values, current_value_stats = fake_quantize_value(
            values, value_bits, group_size, quantizer
        )
        # The exact tensors below are passed to attention.  Requantizing them
        # verifies that no later path silently substituted the original BF16
        # cache and that the QDQ result is a quantization-grid fixed point.
        requantized_keys, _ = fake_quantize_key(
            dequantized_keys, key_bits, group_size, quantizer
        )
        requantized_values, _ = fake_quantize_value(
            dequantized_values, value_bits, group_size, quantizer
        )
        fixed_point_max_abs_error = max(
            fixed_point_max_abs_error,
            float(
                (requantized_keys.float() - dequantized_keys.float())
                .abs()
                .max()
                .item()
            ),
            float(
                (requantized_values.float() - dequantized_values.float())
                .abs()
                .max()
                .item()
            ),
        )
        fixed_point_reference_max_abs = max(
            fixed_point_reference_max_abs,
            float(dequantized_keys.float().abs().max().item()),
            float(dequantized_values.float().abs().max().item()),
        )
        quantized_layers.append((dequantized_keys, beta, dequantized_values))
        key_stats.append(current_key_stats)
        value_stats.append(current_value_stats)
        # Structurally zero bias (dense controls) does not
        # need storage in a packed implementation. QDQ tensors retain beta
        # because the vendored inference path expects one for every layer.
        if torch.count_nonzero(beta).item() > 0:
            beta_bytes += _tensor_bytes(beta)
        actual_tensor_bytes += (
            _tensor_bytes(dequantized_keys)
            + _tensor_bytes(beta)
            + _tensor_bytes(dequantized_values)
        )
        num_stored_entries += keys.shape[0] * keys.shape[1] * keys.shape[2]
        layer_lengths.append(int(keys.shape[2]))

    total_key_values = sum(stats.num_values for stats in key_stats)
    total_value_values = sum(stats.num_values for stats in value_stats)
    key_mse = (
        sum(stats.mse * stats.num_values for stats in key_stats) / total_key_values
        if total_key_values
        else 0.0
    )
    value_mse = (
        sum(stats.mse * stats.num_values for stats in value_stats) / total_value_values
        if total_value_values
        else 0.0
    )

    fixed_point_tolerance = (
        FIXED_POINT_ATOL + FIXED_POINT_RTOL * fixed_point_reference_max_abs
    )
    stats = CacheQuantizationStats(
        key_bits=key_bits,
        value_bits=value_bits,
        group_size=group_size,
        quantizer=quantizer,
        key_effective_bytes=sum(stats.effective_bytes for stats in key_stats),
        value_effective_bytes=sum(stats.effective_bytes for stats in value_stats),
        key_payload_bytes=sum(stats.payload_bytes for stats in key_stats),
        value_payload_bytes=sum(stats.payload_bytes for stats in value_stats),
        key_scale_bytes=sum(stats.metadata_bytes for stats in key_stats),
        value_scale_bytes=sum(stats.metadata_bytes for stats in value_stats),
        beta_bytes=beta_bytes,
        actual_tensor_bytes=actual_tensor_bytes,
        key_mse=key_mse,
        value_mse=value_mse,
        num_stored_entries=num_stored_entries,
        min_layer_length=min(layer_lengths),
        max_layer_length=max(layer_lengths),
        storage_is_simulated=(key_bits < 16 or value_bits < 16),
        compute_dtype=str(cache[0][0].dtype),
        quantization_fixed_point_verified=(
            fixed_point_max_abs_error <= fixed_point_tolerance
        ),
        quantization_fixed_point_max_abs_error=fixed_point_max_abs_error,
        quantization_fixed_point_tolerance=fixed_point_tolerance,
    )
    return tuple(quantized_layers), stats
