#!/usr/bin/env python
"""
Compare reasoning compaction schemes for original vs sequential objectives.

Schemes compared:
- original
- original + head budget
- sequential
- sequential + head budget
"""
import argparse
import json
import os
import subprocess
import sys
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import copy

import torch

# Ensure repo root is on sys.path for absolute imports when run from parent dirs.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from evaluation.run_reasoning_evaluation import ReasoningEvaluator
from evaluation.configs.utils import load_algorithm_config, load_query_config
from compaction.compaction_methods import get_compaction_method


def build_compaction_method(
    method_name: str,
    algorithm_config: str,
    target_size: float,
    precomputed_budget_path: Optional[str],
    max_ratio_per_head: float,
):
    method_config = load_algorithm_config(algorithm_config, target_size=target_size)
    if method_name not in method_config:
        raise ValueError(f"Method '{method_name}' not found. Available: {list(method_config.keys())}")
    method_kwargs = dict(method_config[method_name])
    if precomputed_budget_path is not None:
        method_kwargs['precomputed_budget_path'] = precomputed_budget_path
        method_kwargs['max_ratio_per_head'] = max_ratio_per_head
    return get_compaction_method(method_name, method_kwargs)


def _ratio_str(value: float) -> str:
    return f"{value:.4f}".rstrip('0').rstrip('.')


def _resolve_original_budget_path(args, model_short_name: str) -> str:
    if args.original_budget_path is not None:
        return args.original_budget_path

    candidate = (
        Path("head_budget_optimization")
        / "head_budgets"
        / model_short_name
        / "global_rms"
        / f"global_rms_t{_ratio_str(args.budget_target_ratio)}.json"
    )
    if candidate.exists():
        return str(candidate)

    fallback = (
        Path("head_budget_optimization")
        / "head_budgets"
        / model_short_name
        / "optimized_agnostic.json"
    )
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        "Could not resolve an original head budget file. Provide --original-budget-path "
        f"or create one under {candidate} or {fallback}."
    )


def _generate_sequential_budget_from_scratch(args, model_short_name: str) -> str:
    if args.sequential_budget_path is not None:
        return args.sequential_budget_path

    budget_output_dir = args.sequential_budget_output_dir
    if budget_output_dir is None:
        budget_output_dir = (
            Path("logs")
            / "budget_optimization"
            / model_short_name
            / f"sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        budget_output_dir = Path(budget_output_dir)

    budget_output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = budget_output_dir / "optimized_agnostic.json"

    if budget_path.exists() and not args.recompute_sequential_budget:
        return str(budget_path)

    cmd = [
        sys.executable,
        "-m",
        "head_budget_optimization.run",
        "--dataset-name",
        args.dataset_name,
        "--n-articles",
        str(args.budget_n_articles),
        "--start-article",
        str(args.budget_start_article),
        "--model-name",
        args.model_name,
        "--device",
        args.device,
        "--algorithm-config",
        args.algorithm_config,
        "--method",
        args.sequential_method,
        "--query-config",
        args.guided_query_config,
        "--target-ratio",
        str(args.budget_target_ratio),
        "--solve-ratios",
        str(args.budget_target_ratio),
        "--solver-method",
        "ratio-agnostic",
        "--output-dir",
        str(budget_output_dir),
        "--max-new-tokens",
        str(args.max_generated_tokens),
    ]

    if args.max_model_len is not None and args.max_model_len != -1:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    print("\n=== Generating sequential head budget from scratch ===")
    print("Command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)

    if not budget_path.exists():
        raise FileNotFoundError(
            f"Sequential budget generation finished but {budget_path} was not created."
        )

    return str(budget_path)


def _call_run_evaluation(evaluator: ReasoningEvaluator, **kwargs) -> Dict[str, Any]:
    """Call run_evaluation with only supported kwargs (for backward compatibility)."""
    sig = inspect.signature(evaluator.run_evaluation)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return evaluator.run_evaluation(**filtered)


def run_scheme(
    evaluator: ReasoningEvaluator,
    scheme: str,
    args,
    compaction_method,
    query_config,
    compaction_policy: str,
    method_name: str,
) -> Dict[str, Any]:
    return _call_run_evaluation(
        evaluator,
        dataset_name=args.dataset_name,
        mode="compaction",
        compaction_method=compaction_method,
        query_config=query_config,
        compaction_interval=args.compaction_interval,
        target_size=args.target_size,
        max_generated_tokens=args.max_generated_tokens,
        protected_tokens=args.protected_tokens,
        seed=args.seed,
        deterministic=True,
        n_problems=args.n_problems,
        start_problem=args.start_problem,
        log_dir=args.log_dir,
        experiment_name=f"{args.name}_{scheme}" if args.name else scheme,
        algorithm_config_file=args.algorithm_config,
        query_config_file=args.guided_query_config,
        compaction_policy=compaction_policy,
        num_workers=args.num_workers,
        method_name=method_name,
        query_config_name=args.guided_query_config,
        precomputed_budget_path=args.precomputed_budget_path,
        max_ratio_per_head=args.max_ratio_per_head,
    )


def main():
    parser = argparse.ArgumentParser(description="Sequential reasoning compaction comparison.")
    parser.add_argument("--dataset-name", type=str, default="aime2025")
    parser.add_argument("--n-problems", type=int, default=-1)
    parser.add_argument("--start-problem", type=int, default=0)

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the model max length used by vLLM/HF cache allocation. Use -1 to let vLLM use the model default.",
    )
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument(
        "--compaction-interval", "--max-seq-len", "--max-len", "--max_len",
        dest="compaction_interval",
        type=int,
        default=4096,
        help="Number of tokens to generate per phase before compacting the KV cache.",
    )
    parser.add_argument("--target-size", type=float, default=0.1)
    parser.add_argument("--max-generated-tokens", type=int, default=4096)
    parser.add_argument("--protected-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--method", type=str, default="AM-HighestAttnKeys-basic")
    parser.add_argument("--algorithm-config", type=str, default="default")
    parser.add_argument("--guided-query-config", type=str, default="ss-plus-context-prefill")
    parser.add_argument(
        "--sequential-method",
        type=str,
        default="AM-HighestAttnKeys-sequential",
        help="Method name for the sequential objective.",
    )
    parser.add_argument(
        "--compaction-policy",
        type=str,
        default="concatenate",
        choices=["concatenate", "recompact_all"],
        help="Compaction policy to evaluate for all schemes.",
    )
    parser.add_argument(
        "--original-budget-path",
        type=str,
        default=None,
        help="Path to the precomputed head budget proportions JSON file for the original objective.",
    )
    parser.add_argument(
        "--max-ratio-per-head",
        type=float,
        default=1.0,
        help=(
            "Maximum ratio per head when using precomputed budgets (default: 1.0). "
            "If budgets would assign a higher ratio, proportions are blended towards uniform."
        ),
    )
    parser.add_argument(
        "--budget-target-ratio",
        type=float,
        default=0.05,
        help="Target ratio used when bootstrapping the sequential head budget from scratch.",
    )
    parser.add_argument(
        "--budget-n-articles",
        type=int,
        default=1,
        help="Number of articles to use when bootstrapping the sequential head budget.",
    )
    parser.add_argument(
        "--budget-start-article",
        type=int,
        default=0,
        help="Starting article index for sequential budget bootstrapping.",
    )
    parser.add_argument(
        "--sequential-budget-path",
        type=str,
        default=None,
        help="Existing sequential head budget file to use instead of generating one.",
    )
    parser.add_argument(
        "--sequential-budget-output-dir",
        type=str,
        default=None,
        help="Directory to write the generated sequential head budget.",
    )
    parser.add_argument(
        "--recompute-sequential-budget",
        action="store_true",
        help="Force regeneration of the sequential head budget even if it already exists.",
    )

    parser.add_argument("--log-dir", type=str, default="logs/reasoning_evaluation")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[
            "original",
            "original_head_budget",
            "sequential",
            "sequential_head_budget",
        ],
        choices=[
            "original",
            "original_head_budget",
            "sequential",
            "sequential_head_budget",
        ],
        help="Select which schemes to run (useful for bash-level parallelism).",
    )

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device != "cuda":
        raise RuntimeError("CUDA is required")

    if args.max_model_len == -1:
        max_model_len = None
    elif args.max_model_len is not None:
        max_model_len = args.max_model_len
    else:
        max_model_len = args.compaction_interval + 2048
    evaluator = ReasoningEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_model_len=max_model_len,
    )

    guided_query_config = load_query_config(args.guided_query_config)
    model_short_name = args.model_name.split('/')[-1]
    original_budget_path = _resolve_original_budget_path(args, model_short_name)
    sequential_budget_path = _generate_sequential_budget_from_scratch(args, model_short_name)

    schemes = {
        "original": (args.method, None),
        "original_head_budget": (args.method, original_budget_path),
        "sequential": (args.sequential_method, None),
        "sequential_head_budget": (args.sequential_method, sequential_budget_path),
    }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "compaction_policy": args.compaction_policy,
        "schemes": {},
        "query_configs": {
            "guided": args.guided_query_config,
        },
        "methods": {
            "original": args.method,
            "sequential": args.sequential_method,
        },
        "budget_paths": {
            "original": original_budget_path,
            "sequential": sequential_budget_path,
        },
    }

    scheme_filter = set(args.schemes)
    filtered_schemes = [item for item in schemes.items() if item[0] in scheme_filter]

    for scheme, (method_name, budget_path) in filtered_schemes:
        print(f"\n=== Running scheme: {scheme} ===")
        compaction_method = build_compaction_method(
            method_name,
            args.algorithm_config,
            args.target_size,
            budget_path,
            args.max_ratio_per_head,
        )
        output = run_scheme(
            evaluator,
            scheme,
            args,
            compaction_method,
            guided_query_config,
            args.compaction_policy,
            method_name,
        )
        summary["schemes"][scheme] = copy.deepcopy(output.get("overall_stats", {}))

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    if len(filtered_schemes) == 1:
        summary_path = log_path / (
            f"reasoning_compaction_sequential_{filtered_schemes[0][0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    else:
        summary_path = log_path / f"reasoning_compaction_sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Comparison Summary ===")
    header = (
        f"{'scheme':26s} | {'acc':>7s} | {'gen_tok':>7s} | {'rea_tok':>7s} | {'ans_tok':>7s} | "
        f"{'time(s)':>7s} | {'#compact':>8s} | {'cmp_t(s)':>8s} | "
        f"{'peak_kv':>8s} | {'avg_kv':>8s}"
    )
    print(header)
    print("-" * len(header))
    for scheme, stats in summary["schemes"].items():
        accuracy = stats.get("accuracy", 0.0)
        avg_total_gen = stats.get("avg_total_generated_tokens", 0.0)
        avg_reasoning = stats.get("avg_reasoning_tokens", 0.0)
        avg_answer = stats.get("avg_answer_tokens", 0.0)
        avg_time = stats.get("avg_generation_time", 0.0)
        avg_compactions = stats.get("avg_num_compactions", 0.0)
        avg_compaction_time = stats.get("avg_compaction_time", 0.0)
        max_peak_kv_mb = stats.get("max_peak_kv_memory_mb", 0.0)
        avg_peak_kv_mb = stats.get("avg_peak_kv_memory_mb", 0.0)
        print(
            f"{scheme:26s} | {accuracy:7.2%} | {avg_total_gen:7.0f} | {avg_reasoning:7.0f} | {avg_answer:7.0f} | "
            f"{avg_time:7.2f} | {avg_compactions:8.1f} | {avg_compaction_time:8.2f} | "
            f"{max_peak_kv_mb:7.1f}M | {avg_peak_kv_mb:7.1f}M"
        )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
