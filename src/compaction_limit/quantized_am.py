from __future__ import annotations

import math
from typing import Any

import torch

from .quantization import fake_quantize_key, fake_quantize_value

Tensor = torch.Tensor


def _quantize_head_key(
    keys: Tensor,
    bits: int,
    group_size: int,
    quantizer: str,
    prefix: Tensor | None = None,
    suffix: Tensor | None = None,
) -> Tensor:
    segments = []
    if prefix is not None and prefix.shape[0] > 0:
        segments.append(prefix)
    compact_start = sum(segment.shape[0] for segment in segments)
    segments.append(keys)
    if suffix is not None and suffix.shape[0] > 0:
        segments.append(suffix)
    combined = torch.cat(segments, dim=0)
    expanded = combined.unsqueeze(0).unsqueeze(0)
    quantized, _ = fake_quantize_key(expanded, bits, group_size, quantizer)
    return quantized[0, 0, compact_start : compact_start + keys.shape[0]]


def _quantize_head_value(
    values: Tensor,
    bits: int,
    group_size: int,
    quantizer: str,
) -> Tensor:
    expanded = values.unsqueeze(0).unsqueeze(0)
    quantized, _ = fake_quantize_value(expanded, bits, group_size, quantizer)
    return quantized[0, 0]


def _broadcast_attention_bias(
    attention_bias: Tensor | None,
    shape: torch.Size,
) -> Tensor | None:
    if attention_bias is None:
        return None
    try:
        return torch.broadcast_to(attention_bias.float(), shape)
    except RuntimeError as error:
        raise ValueError(
            f"attention_bias must be broadcastable to {tuple(shape)}, got "
            f"{tuple(attention_bias.shape)}"
        ) from error


def _bounded_nnls(
    matrix: Tensor,
    target: Tensor,
    iterations: int,
    lower: float,
    upper: float,
) -> Tensor:
    weights = _solve_values(matrix, target).clamp(lower, upper)
    if iterations <= 0:
        return weights
    spectral_norm = torch.linalg.matrix_norm(matrix, ord=2)
    step_size = 1.0 / spectral_norm.square().clamp_min(1e-12)
    for _ in range(iterations):
        gradient = matrix.T @ (matrix @ weights - target)
        weights = (weights - step_size * gradient).clamp(lower, upper)
    return weights


def _solve_values(design: Tensor, targets: Tensor) -> Tensor:
    if not torch.isfinite(design).all() or not torch.isfinite(targets).all():
        raise RuntimeError("Cannot fit compact values from non-finite inputs.")

    try:
        solution = torch.linalg.lstsq(design, targets).solution
        if torch.isfinite(solution).all():
            return solution
    except torch.linalg.LinAlgError:
        pass

    gram = design.T @ design
    gram = 0.5 * (gram + gram.T)
    rhs = design.T @ targets
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    scale = gram.abs().sum(dim=1).max().clamp_min(1.0)

    # CUDA's least-squares driver assumes full rank. Quantized attention can
    # produce duplicate columns, so retry the equivalent ridge system with a
    # regularizer scaled to the matrix magnitude instead of a fixed epsilon.
    for relative_ridge in (1e-6, 1e-5, 1e-4, 1e-3):
        solution, info = torch.linalg.solve_ex(
            gram + identity * (scale * relative_ridge),
            rhs,
            check_errors=False,
        )
        if bool(torch.all(info == 0)) and torch.isfinite(solution).all():
            return solution

    raise RuntimeError("Quantization-aware value fitting failed after ridge retries.")


def quantization_aware_refit(
    selected_keys: Tensor,
    full_keys: Tensor,
    full_values: Tensor,
    queries: Tensor,
    key_bits: int,
    value_bits: int,
    group_size: int,
    quantizer: str,
    refinement_iterations: int = 2,
    nnls_iterations: int = 2,
    attention_bias: Tensor | None = None,
    key_prefix: Tensor | None = None,
    key_suffix: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float | int]]:
    """Fit AM beta/values while evaluating the final quantized compact cache.

    Keys remain restricted to the selected original-key subset. The routine
    fake-quantizes those keys, refits beta against the full-precision attention
    mass, solves compact values, and performs residual corrections after value
    fake quantization. The returned keys and values are QDQ tensors and can be
    packed using the requested bit widths by a real storage backend.
    """

    if refinement_iterations < 0:
        raise ValueError("refinement_iterations must be non-negative.")
    if nnls_iterations < 0:
        raise ValueError("nnls_iterations must be non-negative.")

    original_dtype = full_keys.dtype
    dimension = full_keys.shape[-1]
    scale = 1.0 / math.sqrt(dimension)
    queries32 = queries.float()
    keys32 = full_keys.float()
    values32 = full_values.float()

    compact_keys = _quantize_head_key(
        selected_keys,
        key_bits,
        group_size,
        quantizer,
        prefix=key_prefix,
        suffix=key_suffix,
    )
    full_scores = (queries32 @ keys32.T) * scale
    bias32 = _broadcast_attention_bias(attention_bias, full_scores.shape)
    if bias32 is not None:
        full_scores = full_scores + bias32
    compact_scores = (queries32 @ compact_keys.float().T) * scale
    row_max = torch.maximum(
        full_scores.max(dim=1, keepdim=True).values,
        compact_scores.max(dim=1, keepdim=True).values,
    )
    full_exp = torch.exp(full_scores - row_max)
    mass_target = full_exp.sum(dim=1)

    mass_design = torch.exp(compact_scores - row_max)
    weights = _bounded_nnls(
        mass_design,
        mass_target,
        iterations=nnls_iterations,
        lower=math.exp(-3),
        upper=math.exp(3),
    )
    beta = torch.log(weights).to(original_dtype)

    targets = torch.softmax(full_scores, dim=-1) @ values32
    design = torch.softmax(compact_scores + beta.float(), dim=-1)
    latent_values = _solve_values(design, targets).to(original_dtype)

    best_values = _quantize_head_value(
        latent_values,
        value_bits,
        group_size,
        quantizer,
    )
    initial_mse = torch.mean((design @ best_values.float() - targets).square()).item()
    best_mse = initial_mse
    iterations_evaluated = 0
    iterations_accepted = 0

    for _ in range(refinement_iterations):
        iterations_evaluated += 1
        residual = targets - design @ best_values.float()
        correction = _solve_values(design, residual)
        candidate_latent = (best_values.float() + correction).to(original_dtype)
        candidate_values = _quantize_head_value(
            candidate_latent,
            value_bits,
            group_size,
            quantizer,
        )
        candidate_mse = torch.mean(
            (design @ candidate_values.float() - targets).square()
        ).item()
        if candidate_mse >= best_mse:
            break
        best_values = candidate_values
        best_mse = candidate_mse
        iterations_accepted += 1

    stats: dict[str, float | int] = {
        "key_bits": key_bits,
        "value_bits": value_bits,
        "group_size": group_size,
        "key_prefix_tokens": 0 if key_prefix is None else key_prefix.shape[0],
        "key_suffix_tokens": 0 if key_suffix is None else key_suffix.shape[0],
        "refinement_iterations_requested": refinement_iterations,
        "refinement_iterations_evaluated": iterations_evaluated,
        "refinement_iterations_accepted": iterations_accepted,
        "initial_quantized_attention_mse": initial_mse,
        "final_quantized_attention_mse": best_mse,
        "relative_mse_reduction": (
            (initial_mse - best_mse) / initial_mse if initial_mse > 0 else 0.0
        ),
    }
    return compact_keys, beta, best_values, stats


def make_quantization_aware_algorithm() -> type[Any]:
    """Create a vendored-AM-compatible algorithm class after vendor activation."""

    from compaction.algorithms.highest_attention_keys import (
        HighestAttentionKeysCompaction,
    )

    class QuantizationAwareHighestAttentionKeys(HighestAttentionKeysCompaction):
        def __init__(
            self,
            key_bits: int,
            value_bits: int,
            group_size: int,
            quantizer: str,
            refinement_iterations: int = 2,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.key_bits = key_bits
            self.value_bits = value_bits
            self.group_size = group_size
            self.quantizer = quantizer
            self.refinement_iterations = refinement_iterations
            self.last_stats: dict[str, float | int] = {}
            self.key_prefix: Tensor | None = None
            self.key_suffix: Tensor | None = None

        def name(self) -> str:
            return "QuantizationAwareHighestAttentionKeys"

        def set_key_quantization_context(
            self,
            prefix: Tensor,
            suffix: Tensor,
        ) -> None:
            self.key_prefix = prefix
            self.key_suffix = suffix

        def compute_compacted_cache(
            self,
            K: Tensor,
            V: Tensor,
            queries: Tensor,
            t: int,
            attention_bias: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor, list[int]]:
            selected_keys, _, selected_indices = self._select_keys_highest_attention(
                K,
                queries,
                t,
                attention_bias,
            )
            compact_keys, beta, compact_values, stats = quantization_aware_refit(
                selected_keys=selected_keys,
                full_keys=K,
                full_values=V,
                queries=queries,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                group_size=self.group_size,
                quantizer=self.quantizer,
                refinement_iterations=self.refinement_iterations,
                nnls_iterations=self.nnls_iters,
                attention_bias=attention_bias,
                key_prefix=self.key_prefix,
                key_suffix=self.key_suffix,
            )
            self.last_stats = stats
            return compact_keys, beta, compact_values, selected_indices

    return QuantizationAwareHighestAttentionKeys
