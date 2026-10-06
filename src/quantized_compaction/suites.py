from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .grid import FULL_KV_PAIRS, Condition

SYMMETRIC_PAIRS = ((16, 16), (8, 8), (4, 4))
DEFAULT_STAGE2_PAIRS = (*SYMMETRIC_PAIRS, (8, 4), (4, 8))
STAGE1_BUDGETS = (0.10, 0.05, 0.02)
STAGE1_FIXED_RATIOS = (0.10, 0.05, 0.02)
STAGE2_BUDGETS = (0.10, 0.05, 0.02)
STAGE3_RETAINED_RATIOS = (0.75, 0.50, 0.25, 0.10, 0.05)


def _condition(
    method: str,
    budget: float,
    precision_pair: tuple[int, int],
    composition_mode: str = "post",
) -> Condition:
    key_bits, value_bits = precision_pair
    entry_ratio = budget * 32.0 / (key_bits + value_bits)
    if entry_ratio > 1.0 + 1e-9:
        raise ValueError(
            f"Budget {budget} cannot be represented by {precision_pair}: "
            f"entry ratio would be {entry_ratio}."
        )
    if math.isclose(entry_ratio, 1.0):
        method = "dense"
        composition_mode = "post"
        entry_ratio = 1.0
    return Condition(
        method=method,
        entry_ratio=entry_ratio,
        key_bits=key_bits,
        value_bits=value_bits,
        ideal_budget_fraction=budget,
        composition_mode=composition_mode,
    )


def _fixed_ratio_condition(
    method: str,
    entry_ratio: float,
    precision_pair: tuple[int, int],
    composition_mode: str = "post",
) -> Condition:
    key_bits, value_bits = precision_pair
    return Condition(
        method=method,
        entry_ratio=entry_ratio,
        key_bits=key_bits,
        value_bits=value_bits,
        ideal_budget_fraction=entry_ratio * (key_bits + value_bits) / 32.0,
        composition_mode=composition_mode,
    )


def _dense_conditions() -> list[Condition]:
    return [
        Condition(
            method="dense",
            entry_ratio=1.0,
            key_bits=key_bits,
            value_bits=value_bits,
            ideal_budget_fraction=(key_bits + value_bits) / 32.0,
            composition_mode="post",
        )
        for key_bits, value_bits in FULL_KV_PAIRS
    ]


def _deduplicate(conditions: Iterable[Condition]) -> tuple[Condition, ...]:
    unique: dict[str, Condition] = {}
    for condition in conditions:
        unique[condition.condition_id] = condition
    return tuple(unique.values())


def build_stage1_suite() -> tuple[Condition, ...]:
    """AM-only cardinality/precision surface with quantization controls.

    Equal-byte diagonals answer how a fixed memory budget should be allocated
    between retained entries and bits per entry.  The fixed-ratio factorial
    independently exposes the cardinality-by-precision interaction.  Dense
    controls isolate quantization without compaction.
    """

    conditions: list[Condition] = [*_dense_conditions()]
    for budget in STAGE1_BUDGETS:
        conditions.extend(_condition("am", budget, pair) for pair in FULL_KV_PAIRS)
    for ratio in STAGE1_FIXED_RATIOS:
        for pair in FULL_KV_PAIRS:
            conditions.append(_fixed_ratio_condition("am", ratio, pair))
    return _deduplicate(conditions)


def build_stage2_suite(
    precision_pairs: Sequence[tuple[int, int]] = DEFAULT_STAGE2_PAIRS,
) -> tuple[Condition, ...]:
    """Composition-order surface on Stage 1's stratified precision shortlist."""

    pairs = tuple(dict.fromkeys(precision_pairs))
    if not pairs:
        raise ValueError("Stage 2 requires at least one precision pair.")
    unsupported = set(pairs) - set(FULL_KV_PAIRS)
    if unsupported:
        raise ValueError(f"Unsupported Stage 2 precision pairs: {sorted(unsupported)}")
    conditions: list[Condition] = []
    for budget in STAGE2_BUDGETS:
        for pair in pairs:
            for mode in ("post", "pre", "aware"):
                conditions.append(_condition("am", budget, pair, mode))
    return _deduplicate(conditions)


def _axis(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("dataset", "tofu")),
        str(row.get("weight_precision", "bf16")),
    )


def _dense_baselines(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("method") == "dense"
            and row.get("key_bits") == 16
            and row.get("value_bits") == 16
        ):
            baselines[_axis(row)] = row
    if not baselines:
        raise ValueError("No dense K16V16 baseline was found.")
    return baselines


def _row_quality_key(
    row: dict[str, Any], baselines: dict[tuple[str, str], dict[str, Any]]
) -> tuple[float, float, float, float]:
    baseline = baselines.get(_axis(row))
    if baseline is None:
        return (math.inf, math.inf, math.inf, math.inf)
    accuracy = row.get("primary_accuracy", row.get("token_f1"))
    baseline_accuracy = baseline.get("primary_accuracy", baseline.get("token_f1"))
    accuracy_drop = (
        max(0.0, float(baseline_accuracy) - float(accuracy))
        if _finite(accuracy) and _finite(baseline_accuracy)
        else math.inf
    )
    nll_delta = (
        max(0.0, float(row["answer_nll"]) - float(baseline["answer_nll"]))
        if _finite(row.get("answer_nll")) and _finite(baseline.get("answer_nll"))
        else math.inf
    )
    length_ratio = (
        float(row["mean_generated_tokens"])
        / float(baseline["mean_generated_tokens"])
        if _finite(row.get("mean_generated_tokens"))
        and _finite(baseline.get("mean_generated_tokens"))
        and float(baseline["mean_generated_tokens"]) > 0
        else 1.0
    )
    length_penalty = max(0.0, length_ratio - 1.25)
    memory = float(
        row.get(
            "model_plus_logical_packed_kv_bytes",
            row.get("logical_packed_kv_bytes", math.inf),
        )
    )
    return (accuracy_drop, length_penalty, nll_delta, memory)


def _group_quality_key(
    rows: Sequence[dict[str, Any]],
    baselines: dict[tuple[str, str], dict[str, Any]],
) -> tuple[float, float, float, float]:
    if not rows:
        return (math.inf, math.inf, math.inf, math.inf)
    keys = [_row_quality_key(row, baselines) for row in rows]
    return tuple(max(key[index] for key in keys) for index in range(4))


def _is_primary_budget(row: dict[str, Any]) -> bool:
    return _finite(row.get("ideal_budget_fraction")) and any(
        math.isclose(float(row["ideal_budget_fraction"]), budget, abs_tol=1e-9)
        for budget in STAGE1_BUDGETS
    )


def select_stage2_precision_pairs(
    stage1_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Keep symmetric controls plus the two best asymmetric precision pairs."""

    baselines = _dense_baselines(stage1_rows)
    am_rows = [
        row
        for row in stage1_rows
        if row.get("method") == "am"
        and row.get("composition_mode", "post") == "post"
        and _is_primary_budget(row)
    ]

    def aggregate_key(pair: tuple[int, int]) -> tuple[float, float, float, float]:
        pair_rows = [
            row
            for row in am_rows
            if (int(row["key_bits"]), int(row["value_bits"])) == pair
        ]
        if not pair_rows:
            return (math.inf, math.inf, math.inf, math.inf)
        return _group_quality_key(pair_rows, baselines)

    asymmetric_pairs = [pair for pair in FULL_KV_PAIRS if pair[0] != pair[1]]
    best_asymmetric = sorted(asymmetric_pairs, key=aggregate_key)[:2]
    selected = (*SYMMETRIC_PAIRS, *best_asymmetric)
    return {
        "selection_applied": True,
        "selection_scope": "stage2_precision_pairs",
        "strategy": "all_symmetric_plus_two_best_asymmetric_pairs",
        "selected_precision_pairs": [list(pair) for pair in selected],
        "pair_worst_case_quality_keys": {
            f"k{pair[0]}v{pair[1]}": list(aggregate_key(pair))
            for pair in FULL_KV_PAIRS
        },
    }


def build_stage3_suite() -> tuple[Condition, ...]:
    """Build the fixed symmetric-precision model-weight interaction grid."""

    conditions: list[Condition] = []
    for pair in SYMMETRIC_PAIRS:
        conditions.append(
            Condition(
                method="dense",
                entry_ratio=1.0,
                key_bits=pair[0],
                value_bits=pair[1],
                ideal_budget_fraction=(pair[0] + pair[1]) / 32.0,
                composition_mode="post",
            )
        )
        conditions.extend(
            _fixed_ratio_condition("am", ratio, pair, "post")
            for ratio in STAGE3_RETAINED_RATIOS
        )
    deduplicated = _deduplicate(conditions)
    if len(deduplicated) != 18:
        raise AssertionError(
            f"Stage 3 must contain exactly 18 conditions, got {len(deduplicated)}."
        )
    return deduplicated


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def select_stage1(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return Stage 1 coverage diagnostics and the Stage 2 precision shortlist."""

    precision_selection = select_stage2_precision_pairs(rows)
    return {
        "selection_applied": True,
        "selection_scope": "stage2_precision_pairs",
        "stage2_precision_selection": precision_selection,
        "num_rows": len(rows),
        "num_conditions": len({row.get("condition_id") for row in rows}),
        "methods": sorted({str(row.get("method")) for row in rows}),
        "primary_budgets": list(STAGE1_BUDGETS),
        "observed_ideal_budget_fractions": sorted(
            {
                float(row["ideal_budget_fraction"])
                for row in rows
                if _finite(row.get("ideal_budget_fraction"))
            },
            reverse=True,
        ),
    }


def select_stage2(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return Stage 2 composition diagnostics."""

    return {
        "selection_applied": False,
        "num_rows": len(rows),
        "num_conditions": len({row.get("condition_id") for row in rows}),
        "composition_modes": sorted(
            {str(row.get("composition_mode")) for row in rows}
        ),
    }


def build_pipeline_plan() -> dict[str, Any]:
    stage1 = build_stage1_suite()
    stage2 = build_stage2_suite()
    stage3 = build_stage3_suite()
    return {
        "sequential": True,
        "funnel_selection": True,
        "stage1": {
            "purpose": "am_cardinality_by_kv_precision_interaction",
            "num_conditions": len(stage1),
            "queries_per_kv_head": 5000,
            "datasets": ["tofu", "quality"],
            "questions": {"tofu": 200, "quality": "all questions in selected articles"},
            "methods": ["am"],
            "controls": ["dense_precision"],
            "budgets": list(STAGE1_BUDGETS),
            "fixed_cardinality_ratios": list(STAGE1_FIXED_RATIOS),
            "precision_pairs": [list(pair) for pair in FULL_KV_PAIRS],
            "conditions": [condition.to_dict() for condition in stage1],
        },
        "stage2": {
            "purpose": "precision_by_composition_interaction_on_stratified_pairs",
            "selection_dependency": "Stage 1 precision allocation",
            "num_conditions": len(stage2),
            "queries_per_kv_head": 5000,
            "datasets": ["tofu", "quality"],
            "questions": {"tofu": 200, "quality": "all questions in selected articles"},
            "methods": ["am"],
            "budgets": list(STAGE2_BUDGETS),
            "default_precision_pairs": [list(pair) for pair in DEFAULT_STAGE2_PAIRS],
            "composition_modes": ["post", "pre", "aware"],
        },
        "stage3": {
            "purpose": "model_weight_by_kv_policy_memory_quality_tradeoff",
            "selection_dependency": None,
            "conditions_per_weight_precision": len(stage3),
            "retained_entry_ratios": [1.0, *STAGE3_RETAINED_RATIOS],
            "precision_pairs": [list(pair) for pair in SYMMETRIC_PAIRS],
            "composition_modes": ["post"],
            "queries_per_kv_head": 5000,
            "datasets": ["tofu", "quality"],
            "questions": {"tofu": 200, "quality": "all questions in selected articles"},
            "methods": ["am"],
            "weight_precisions": ["bf16", "int8", "nf4"],
            "conditions": [condition.to_dict() for condition in stage3],
        },
    }
