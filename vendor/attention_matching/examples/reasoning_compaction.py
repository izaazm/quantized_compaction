#!/usr/bin/env python
"""
Compare compaction schemes for the reasoning pipeline.

Schemes compared:
- baseline: no compaction, generate full max_generated_tokens via vLLM
- recompact_all: compact entire cache (compacted + uncompacted) every compaction_interval tokens
- concatenate: compact only uncompacted context, appending to previous compacted cache
- hybrid: concatenate until hybrid_budget, then recompact all
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import copy

import torch

from evaluation.run_reasoning_evaluation import ReasoningEvaluator
from evaluation.configs.utils import load_algorithm_config, load_query_config
from compaction.compaction_methods import get_compaction_method


def build_compaction_method(method_name: str, algorithm_config: str, target_size: float,
                             precomputed_budget_path: str, max_ratio_per_head: float):
    method_config = load_algorithm_config(algorithm_config, target_size=target_size)
    if method_name not in method_config:
        raise ValueError(f"Method '{method_name}' not found. Available: {list(method_config.keys())}")
    method_kwargs = dict(method_config[method_name])
    if precomputed_budget_path is not None:
        method_kwargs['precomputed_budget_path'] = precomputed_budget_path
        method_kwargs['max_ratio_per_head'] = max_ratio_per_head
    return get_compaction_method(method_name, method_kwargs)


def run_scheme(evaluator: ReasoningEvaluator, scheme: str, args, compaction_method, query_config) -> Dict[str, Any]:
    if scheme == "baseline":
        return evaluator.run_evaluation(
            dataset_name=args.dataset_name,
            mode="baseline",
            compaction_interval=args.compaction_interval,
            max_generated_tokens=args.max_generated_tokens,
            seed=args.seed,
            deterministic=True,
            n_problems=args.n_problems,
            start_problem=args.start_problem,
            log_dir=args.log_dir,
            experiment_name=f"{args.name}_baseline" if args.name else "baseline",
            algorithm_config_file=args.algorithm_config,
            query_config_file=args.query_config,
        )

    if scheme == "recompact_all":
        return evaluator.run_evaluation(
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
            experiment_name=f"{args.name}_recompact" if args.name else "recompact_all",
            algorithm_config_file=args.algorithm_config,
            query_config_file=args.query_config,
            compaction_policy="recompact_all",
        )

    if scheme == "concatenate":
        return evaluator.run_evaluation(
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
            experiment_name=f"{args.name}_concat" if args.name else "concatenate",
            algorithm_config_file=args.algorithm_config,
            query_config_file=args.query_config,
            compaction_policy="concatenate",
        )

    if scheme == "hybrid":
        return evaluator.run_evaluation(
            dataset_name=args.dataset_name,
            mode="compaction",
            compaction_method=compaction_method,
            query_config=query_config,
            compaction_interval=args.compaction_interval,
            target_size=args.target_size,
            max_generated_tokens=args.max_generated_tokens,
            hybrid_compressed_budget=args.hybrid_compressed_budget,
            protected_tokens=args.protected_tokens,
            seed=args.seed,
            deterministic=True,
            n_problems=args.n_problems,
            start_problem=args.start_problem,
            log_dir=args.log_dir,
            experiment_name=f"{args.name}_hybrid" if args.name else "hybrid",
            algorithm_config_file=args.algorithm_config,
            query_config_file=args.query_config,
            compaction_policy="hybrid",
        )

    raise ValueError(f"Unknown scheme: {scheme}")


def main():
    parser = argparse.ArgumentParser(description="Compare compaction schemes.")
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
        help="Override the model max length used by vLLM/HF cache allocation. Use -1 to fetch from model config.",
    )

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
    parser.add_argument("--query-config", type=str, default="repeat")
    parser.add_argument(
        "--precomputed-budget-path",
        type=str,
        default=None,
        help="Path to precomputed head budget proportions JSON file (for nonuniform head budgets)",
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

    parser.add_argument("--log-dir", type=str, default="logs/reasoning_evaluation")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["baseline", "recompact_all", "concatenate", "hybrid"],
        default=None,
        help="Run a single scheme (useful for bash-level parallelism).",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["baseline", "recompact_all", "concatenate", "hybrid"],
        choices=["baseline", "recompact_all", "concatenate", "hybrid"],
    )

    parser.add_argument(
        "--hybrid-compressed-budget",
        type=int,
        default=2048,
        help="Hybrid budget for concatenated compressed cache before forcing full recompact",
    )

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device != "cuda":
        raise RuntimeError("CUDA is required")

    # Handle max_model_len: -1 means use model's default (pass None to vLLM)
    if args.max_model_len == -1:
        max_model_len = None  # vLLM will use model's default context length
    elif args.max_model_len is not None:
        max_model_len = args.max_model_len
    else:
        max_model_len = args.compaction_interval + 2048
    
    evaluator = ReasoningEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_model_len=max_model_len,
    )

    query_config = load_query_config(args.query_config)
    compaction_method = build_compaction_method(
        args.method,
        args.algorithm_config,
        args.target_size,
        args.precomputed_budget_path,
        args.max_ratio_per_head,
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "schemes": {},
    }

    schemes_to_run = [args.scheme] if args.scheme else args.schemes

    for scheme in schemes_to_run:
        print(f"\n=== Running scheme: {scheme} ===")
        output = run_scheme(evaluator, scheme, args, compaction_method, query_config)
        # Store an independent snapshot of overall_stats to avoid accidental shared
        # references across runs (which can make the printed summaries look identical).
        summary["schemes"][scheme] = copy.deepcopy(output.get("overall_stats", {}))

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    if len(schemes_to_run) == 1:
        summary_path = log_path / (
            f"compaction_scheme_compare_{schemes_to_run[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    else:
        summary_path = log_path / f"compaction_scheme_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Comparison Summary ===")
    header = (
        f"{'scheme':16s} | {'acc':>7s} | {'gen_tok':>7s} | {'rea_tok':>7s} | {'ans_tok':>7s} | "
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
            f"{scheme:16s} | {accuracy:7.2%} | {avg_total_gen:7.0f} | {avg_reasoning:7.0f} | {avg_answer:7.0f} | "
            f"{avg_time:7.2f} | {avg_compactions:8.1f} | {avg_compaction_time:8.2f} | "
            f"{max_peak_kv_mb:7.1f}M | {avg_peak_kv_mb:7.1f}M"
        )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
