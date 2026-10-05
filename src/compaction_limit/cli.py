from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import DEFAULT_BIOGRAPHIES_PATH, DEFAULT_PASSAGES_PATH
from .grid import CURATED_KV_PAIRS, parse_precision_pair
from .runner import RunConfig, build_grid, run_experiment


def _precision_pair(value: str) -> tuple[int, int]:
    try:
        return parse_precision_pair(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_grid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grid-mode",
        choices=("equal_budget", "factorial"),
        default="equal_budget",
    )
    parser.add_argument(
        "--budget-fractions",
        type=float,
        nargs="+",
        default=(0.10, 0.05, 0.02),
        help="Fractions of dense BF16 KV bytes for equal-budget diagonals.",
    )
    parser.add_argument(
        "--entry-ratios",
        type=float,
        nargs="+",
        default=(0.10, 0.05, 0.02),
        help="Cardinality ratios used by the factorial grid.",
    )
    parser.add_argument(
        "--kv-pairs",
        type=_precision_pair,
        nargs="+",
        default=CURATED_KV_PAIRS,
        metavar="KxVy",
        help=(
            "Independent key/value precisions, e.g. k16v16 k8v4. The default is "
            "the curated symmetric and asymmetric seven-pair grid."
        ),
    )
    parser.add_argument(
        "--methods",
        choices=("am",),
        nargs="+",
        default=("am",),
    )
    parser.add_argument(
        "--composition-modes",
        choices=("post", "pre", "aware"),
        nargs="+",
        default=("post",),
        help=(
            "post=Q(C(K,V)); pre=Q(C(Q(K,V))); aware=fake quantization inside "
            "AM fitting. Aware applies to AM only."
        ),
    )
    parser.add_argument(
        "--dense-precision-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an uncompacted control for every requested K/V precision pair.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compaction-limit",
        description="Map KV cache cardinality and precision limits on local TOFU data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Print the condition grid without loading a model.",
    )
    _add_grid_arguments(plan_parser)

    run_parser = subparsers.add_parser("run", help="Run the CUDA experiment.")
    _add_grid_arguments(run_parser)
    run_parser.add_argument("--model", default="Qwen/Qwen3-4B")
    run_parser.add_argument(
        "--weight-precision", choices=("bf16", "int8", "nf4"), default="bf16"
    )
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument(
        "--author-ids", type=int, nargs="+", default=tuple(range(10))
    )
    run_parser.add_argument(
        "--biographies-path", type=Path, default=DEFAULT_BIOGRAPHIES_PATH
    )
    run_parser.add_argument("--passages-path", type=Path, default=DEFAULT_PASSAGES_PATH)
    run_parser.add_argument(
        "--corpus-format", choices=("biography", "passages"), default="biography"
    )
    run_parser.add_argument(
        "--dataset", choices=("tofu", "ruler", "quality", "hotpotqa"), default="tofu"
    )
    run_parser.add_argument("--dataset-path", type=Path)
    run_parser.add_argument("--max-questions", type=int, default=200)
    run_parser.add_argument("--max-contexts", type=int)
    run_parser.add_argument("--max-generation-questions", type=int)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--group-size", type=int, default=64)
    run_parser.add_argument(
        "--kv-quantizer",
        choices=("kivi_layout_symmetric", "per_vector_symmetric"),
        default="kivi_layout_symmetric",
    )
    run_parser.add_argument(
        "--max-queries-per-head",
        type=int,
        default=5000,
        help="Maximum reference query vectors used per KV head.",
    )
    run_parser.add_argument("--aware-refinement-iterations", type=int, default=2)
    run_parser.add_argument("--max-new-tokens", type=int, default=64)
    run_parser.add_argument("--generation-batch-size", type=int, default=8)
    run_parser.add_argument(
        "--skip-nll", action="store_true", help="Do not score reference-answer NLL."
    )
    run_parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Do not generate answers or compute text metrics.",
    )
    run_parser.add_argument(
        "--save-detailed-compaction-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-layer/head selected indices and beta summaries.",
    )
    run_parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def _grid_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        grid_mode=args.grid_mode,
        budget_fractions=tuple(args.budget_fractions),
        entry_ratios=tuple(args.entry_ratios),
        precision_pairs=tuple(args.kv_pairs),
        methods=tuple(args.methods),
        composition_modes=tuple(args.composition_modes),
        include_dense_precision_controls=args.dense_precision_controls,
    )


def _run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        model=args.model,
        weight_precision=args.weight_precision,
        device=args.device,
        author_ids=tuple(args.author_ids),
        biographies_path=args.biographies_path,
        passages_path=args.passages_path,
        corpus_format=args.corpus_format,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        max_questions=args.max_questions,
        max_contexts=args.max_contexts,
        max_generation_questions=args.max_generation_questions,
        seed=args.seed,
        grid_mode=args.grid_mode,
        budget_fractions=tuple(args.budget_fractions),
        entry_ratios=tuple(args.entry_ratios),
        precision_pairs=tuple(args.kv_pairs),
        methods=tuple(args.methods),
        composition_modes=tuple(args.composition_modes),
        include_dense_precision_controls=args.dense_precision_controls,
        group_size=args.group_size,
        kv_quantizer=args.kv_quantizer,
        max_queries_per_head=args.max_queries_per_head,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        compute_nll=not args.skip_nll,
        generate=not args.skip_generation,
        save_detailed_compaction_stats=args.save_detailed_compaction_stats,
        aware_refinement_iterations=args.aware_refinement_iterations,
        output_root=args.output_root,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        config = _grid_config(args)
        print(
            json.dumps(
                [condition.to_dict() for condition in build_grid(config)], indent=2
            )
        )
        return
    run_experiment(_run_config(args))


if __name__ == "__main__":
    main()
