from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

SUPPORTED_BITS = (16, 8, 4)
SUPPORTED_METHODS = ("am",)
SUPPORTED_COMPOSITION_MODES = ("post", "pre", "aware")
CURATED_KV_PAIRS = (
    (16, 16),
    (8, 8),
    (4, 4),
    (8, 4),
    (4, 8),
    (16, 4),
    (4, 16),
)
FULL_KV_PAIRS = tuple(
    (key_bits, value_bits) for key_bits in SUPPORTED_BITS for value_bits in SUPPORTED_BITS
)


@dataclass(frozen=True)
class Condition:
    method: str
    entry_ratio: float
    key_bits: int
    value_bits: int
    ideal_budget_fraction: float
    composition_mode: str = "post"

    @property
    def condition_id(self) -> str:
        ratio_text = f"{self.entry_ratio:.6f}".rstrip("0").rstrip(".")
        budget_text = f"{self.ideal_budget_fraction:.6f}".rstrip("0").rstrip(".")
        return (
            f"{self.method}_r{ratio_text}_k{self.key_bits}v{self.value_bits}"
            f"_b{budget_text}_{self.composition_mode}"
        )

    def to_dict(self) -> dict:
        return {"condition_id": self.condition_id, **asdict(self)}


def parse_precision_pair(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("v", ":").replace("k", "")
    try:
        key_text, value_text = normalized.split(":", maxsplit=1)
        pair = (int(key_text), int(value_text))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid KV precision pair {value!r}; use forms such as '8:4' or 'k8v4'."
        ) from error
    return _validate_precision_pairs((pair,))[0]


def _validate_precision_pairs(
    precision_pairs: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        dict.fromkeys(
            (int(key_bits), int(value_bits)) for key_bits, value_bits in precision_pairs
        )
    )
    if not pairs:
        raise ValueError("At least one KV precision pair is required.")
    unsupported = {
        pair
        for pair in pairs
        if pair[0] not in SUPPORTED_BITS or pair[1] not in SUPPORTED_BITS
    }
    if unsupported:
        raise ValueError(
            f"Unsupported KV precision pairs {sorted(unsupported)}; each side must be one "
            f"of {SUPPORTED_BITS}."
        )
    return pairs


def _validate_methods(methods: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).lower() for value in methods))
    unsupported = set(normalized) - set(SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(
            f"Unsupported methods {sorted(unsupported)}; supported values are {SUPPORTED_METHODS}."
        )
    return normalized


def _validate_composition_modes(modes: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).lower() for value in modes))
    unsupported = set(normalized) - set(SUPPORTED_COMPOSITION_MODES)
    if unsupported:
        raise ValueError(
            f"Unsupported composition modes {sorted(unsupported)}; supported values "
            f"are {SUPPORTED_COMPOSITION_MODES}."
        )
    return normalized


def _add_dense_controls(
    conditions: dict[tuple[str, float, int, int, str], Condition],
    precision_pairs: Sequence[tuple[int, int]],
) -> None:
    for key_bits, value_bits in precision_pairs:
        condition = Condition(
            method="dense",
            entry_ratio=1.0,
            key_bits=key_bits,
            value_bits=value_bits,
            ideal_budget_fraction=(key_bits + value_bits) / 32.0,
            composition_mode="post",
        )
        conditions[("dense", 1.0, key_bits, value_bits, "post")] = condition


def build_equal_budget_grid(
    budget_fractions: Sequence[float],
    methods: Sequence[str] = SUPPORTED_METHODS,
    precision_pairs: Sequence[tuple[int, int]] = CURATED_KV_PAIRS,
    include_dense_precision_controls: bool = True,
    composition_modes: Sequence[str] = ("post",),
) -> tuple[Condition, ...]:
    """Create equal-payload-byte allocations across cardinality and K/V precision.

    A dense K16V16 cache is budget 1.0. A condition with retained-entry ratio
    ``r``, key precision ``k``, and value precision ``v`` has ideal payload
    budget ``r * (k + v) / 32``. Scale metadata and AM beta are accounted for
    after quantization rather than in the ideal matching equation.
    """

    pairs = _validate_precision_pairs(precision_pairs)
    normalized_methods = _validate_methods(methods)
    normalized_modes = _validate_composition_modes(composition_modes)
    conditions: dict[tuple[str, float, int, int, str], Condition] = {}

    if include_dense_precision_controls:
        _add_dense_controls(conditions, pairs)

    for budget in budget_fractions:
        budget = float(budget)
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"Budget fractions must lie in (0, 1], got {budget}")
        for key_bits, value_bits in pairs:
            entry_ratio = budget * 32.0 / (key_bits + value_bits)
            if entry_ratio > 1.0 + 1e-9:
                continue
            entry_ratio = min(1.0, entry_ratio)
            if entry_ratio == 1.0:
                condition = Condition(
                    method="dense",
                    entry_ratio=1.0,
                    key_bits=key_bits,
                    value_bits=value_bits,
                    ideal_budget_fraction=budget,
                    composition_mode="post",
                )
                conditions[("dense", 1.0, key_bits, value_bits, "post")] = condition
                continue
            for method in normalized_methods:
                for composition_mode in normalized_modes:
                    if method != "am" and composition_mode == "aware":
                        continue
                    condition = Condition(
                        method=method,
                        entry_ratio=entry_ratio,
                        key_bits=key_bits,
                        value_bits=value_bits,
                        ideal_budget_fraction=budget,
                        composition_mode=composition_mode,
                    )
                    conditions[
                        (method, entry_ratio, key_bits, value_bits, composition_mode)
                    ] = condition

    return tuple(
        sorted(
            conditions.values(),
            key=lambda condition: (
                -condition.ideal_budget_fraction,
                -condition.entry_ratio,
                -condition.key_bits,
                -condition.value_bits,
                condition.method,
                condition.composition_mode,
            ),
        )
    )


def build_factorial_grid(
    entry_ratios: Sequence[float],
    precision_pairs: Sequence[tuple[int, int]] = CURATED_KV_PAIRS,
    methods: Sequence[str] = SUPPORTED_METHODS,
    include_dense_precision_controls: bool = True,
    composition_modes: Sequence[str] = ("post",),
) -> tuple[Condition, ...]:
    pairs = _validate_precision_pairs(precision_pairs)
    normalized_methods = _validate_methods(methods)
    normalized_modes = _validate_composition_modes(composition_modes)
    conditions: dict[tuple[str, float, int, int, str], Condition] = {}

    if include_dense_precision_controls:
        _add_dense_controls(conditions, pairs)

    for ratio in entry_ratios:
        ratio = float(ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"Entry ratios must lie in (0, 1], got {ratio}")
        for key_bits, value_bits in pairs:
            budget = ratio * (key_bits + value_bits) / 32.0
            if ratio == 1.0:
                condition = Condition(
                    method="dense",
                    entry_ratio=ratio,
                    key_bits=key_bits,
                    value_bits=value_bits,
                    ideal_budget_fraction=budget,
                    composition_mode="post",
                )
                conditions[("dense", ratio, key_bits, value_bits, "post")] = condition
            else:
                for method in normalized_methods:
                    for composition_mode in normalized_modes:
                        if method != "am" and composition_mode == "aware":
                            continue
                        condition = Condition(
                            method=method,
                            entry_ratio=ratio,
                            key_bits=key_bits,
                            value_bits=value_bits,
                            ideal_budget_fraction=budget,
                            composition_mode=composition_mode,
                        )
                        conditions[
                            (method, ratio, key_bits, value_bits, composition_mode)
                        ] = condition

    return tuple(
        sorted(
            conditions.values(),
            key=lambda condition: (
                -condition.entry_ratio,
                -condition.key_bits,
                -condition.value_bits,
                condition.method,
                condition.composition_mode,
            ),
        )
    )
