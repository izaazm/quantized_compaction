from __future__ import annotations

import csv
import json
import math
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from . import __version__
from .am_backend import (
    audit_model_quantization,
    answer_diagnostics,
    clear_device_cache,
    compact_cache,
    count_generated_tokens,
    extract_compaction_queries,
    extract_context_cache,
    format_qa_prompts,
    generate_answers,
    load_model_and_tokenizer,
    seed_everything,
)
from .data import (
    DEFAULT_BIOGRAPHIES_PATH,
    DEFAULT_PASSAGES_PATH,
    QARecord,
    load_evaluation_corpora,
)
from .grid import (
    CURATED_KV_PAIRS,
    SUPPORTED_BITS,
    SUPPORTED_COMPOSITION_MODES,
    Condition,
    build_equal_budget_grid,
    build_factorial_grid,
)
from .metrics import (
    best_exact_match,
    best_rouge_l,
    best_token_f1,
    finite_mean,
    quality_score,
    ruler_score,
)
from .quantization import (
    dense_cache_to_compacted,
    quantize_compacted_cache,
)


@dataclass(frozen=True)
class RunConfig:
    model: str = "Qwen/Qwen3-4B"
    weight_precision: str = "bf16"
    device: str = "cuda"
    author_ids: tuple[int, ...] = tuple(range(10))
    biographies_path: Path = DEFAULT_BIOGRAPHIES_PATH
    passages_path: Path = DEFAULT_PASSAGES_PATH
    corpus_format: str = "biography"
    dataset_name: str = "tofu"
    dataset_path: Path | None = None
    max_questions: int | None = 200
    max_contexts: int | None = None
    max_generation_questions: int | None = None
    seed: int = 42
    grid_mode: str = "equal_budget"
    budget_fractions: tuple[float, ...] = (0.10, 0.05, 0.02)
    entry_ratios: tuple[float, ...] = (0.10, 0.05, 0.02)
    precision_pairs: tuple[tuple[int, int], ...] = CURATED_KV_PAIRS
    methods: tuple[str, ...] = ("am",)
    composition_modes: tuple[str, ...] = ("post",)
    include_dense_precision_controls: bool = True
    group_size: int = 64
    kv_quantizer: str = "kivi_layout_symmetric"
    max_queries_per_head: int = 5000
    max_new_tokens: int = 64
    generation_batch_size: int = 8
    compute_nll: bool = True
    generate: bool = True
    save_detailed_compaction_stats: bool = False
    aware_refinement_iterations: int = 2
    conditions: tuple[Condition, ...] | None = None
    output_root: Path = Path("outputs")


def build_grid(config: RunConfig) -> tuple[Condition, ...]:
    if config.conditions is not None:
        if not config.conditions:
            raise ValueError("An explicit condition suite cannot be empty.")
        return config.conditions
    if config.grid_mode == "equal_budget":
        return build_equal_budget_grid(
            budget_fractions=config.budget_fractions,
            methods=config.methods,
            precision_pairs=config.precision_pairs,
            include_dense_precision_controls=config.include_dense_precision_controls,
            composition_modes=config.composition_modes,
        )
    if config.grid_mode == "factorial":
        return build_factorial_grid(
            entry_ratios=config.entry_ratios,
            methods=config.methods,
            precision_pairs=config.precision_pairs,
            include_dense_precision_controls=config.include_dense_precision_controls,
            composition_modes=config.composition_modes,
        )
    raise ValueError(
        f"Unknown grid mode {config.grid_mode!r}; choose 'equal_budget' or 'factorial'."
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() <= 10_000:
            return tensor.tolist()
        float_tensor = tensor.float()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(float_tensor.min().item()),
            "max": float(float_tensor.max().item()),
            "mean": float(float_tensor.mean().item()),
        }
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, default=_json_default, allow_nan=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default, allow_nan=True) + "\n")


def _write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    _write_jsonl(output_dir / "summary.jsonl", summaries)
    if not summaries:
        return
    fieldnames = list(summaries[0])
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def _safe_slug(text: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in text
    ).strip("-")


def _create_output_dir(config: RunConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config.output_root) / (
        f"{timestamp}-{_safe_slug(config.model)}-{config.weight_precision}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions").mkdir()
    (output_dir / "compaction_stats").mkdir()
    (output_dir / "condition_stats").mkdir()
    return output_dir


def _condition_groups(
    conditions: Sequence[Condition],
) -> list[tuple[tuple[Any, ...], list[Condition]]]:
    grouped: dict[tuple[Any, ...], list[Condition]] = defaultdict(list)
    for condition in conditions:
        if condition.method == "dense":
            key = ("dense", 1.0, "post", None, None)
        elif condition.composition_mode == "post":
            key = (
                condition.method,
                condition.entry_ratio,
                condition.composition_mode,
                None,
                None,
            )
        else:
            key = (
                condition.method,
                condition.entry_ratio,
                condition.composition_mode,
                condition.key_bits,
                condition.value_bits,
            )
        grouped[key].append(condition)
    return list(grouped.items())


def _validate_conditions(conditions: Sequence[Condition]) -> None:
    condition_ids = [condition.condition_id for condition in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("Condition IDs must be unique.")
    for condition in conditions:
        if condition.method not in ("dense", "am"):
            raise ValueError(f"Unsupported condition method: {condition.method}")
        if condition.composition_mode not in SUPPORTED_COMPOSITION_MODES:
            raise ValueError(
                f"Unsupported composition mode: {condition.composition_mode}"
            )
        if condition.key_bits not in SUPPORTED_BITS:
            raise ValueError(f"Unsupported key precision: {condition.key_bits}")
        if condition.value_bits not in SUPPORTED_BITS:
            raise ValueError(f"Unsupported value precision: {condition.value_bits}")
        if not 0.0 < condition.entry_ratio <= 1.0:
            raise ValueError("Condition entry ratios must lie in (0, 1].")
        if condition.ideal_budget_fraction <= 0.0:
            raise ValueError("Condition ideal budgets must be positive.")
        if condition.method == "dense" and (
            condition.entry_ratio != 1.0 or condition.composition_mode != "post"
        ):
            raise ValueError("Dense conditions must use ratio 1 and post composition.")
        if condition.composition_mode == "aware" and condition.method != "am":
            raise ValueError("Aware composition is valid only for AM conditions.")


def _generate_in_batches(
    model: Any,
    tokenizer: Any,
    cache: tuple,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    original_seq_len: int,
) -> list[str]:
    predictions = []
    start = 0
    effective_batch_size = min(batch_size, len(prompts))
    while start < len(prompts):
        current_prompts = prompts[start : start + effective_batch_size]
        try:
            predictions.extend(
                generate_answers(
                    model=model,
                    tokenizer=tokenizer,
                    cache=cache,
                    prompts=current_prompts,
                    max_new_tokens=max_new_tokens,
                    original_seq_len=original_seq_len,
                )
            )
            start += len(current_prompts)
        except torch.cuda.OutOfMemoryError:
            clear_device_cache()
            if effective_batch_size == 1:
                raise
            reduced_batch_size = max(1, effective_batch_size // 2)
            print(
                "Generation batch exceeded available CUDA memory; reducing "
                f"batch size from {effective_batch_size} to {reduced_batch_size}.",
                flush=True,
            )
            effective_batch_size = reduced_batch_size
    return predictions


def _evaluate_condition(
    config: RunConfig,
    condition: Condition,
    model: Any,
    tokenizer: Any,
    cache: tuple,
    prompts: Sequence[str],
    questions: Sequence[QARecord],
    original_seq_len: int,
    context_id: str = "context",
    generate_question_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    predictions = [""] * len(questions)
    generated_flags = [False] * len(questions)
    if config.generate:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, question in enumerate(questions):
            if (
                generate_question_ids is not None
                and question.question_id not in generate_question_ids
            ):
                continue
            token_cap = question.max_new_tokens or config.max_new_tokens
            buckets[min(token_cap, config.max_new_tokens)].append(index)
        for token_cap, indices in buckets.items():
            bucket_predictions = _generate_in_batches(
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                prompts=[prompts[index] for index in indices],
                batch_size=config.generation_batch_size,
                max_new_tokens=token_cap,
                original_seq_len=original_seq_len,
            )
            for index, prediction in zip(indices, bucket_predictions):
                predictions[index] = prediction
                generated_flags[index] = True

    rows = []
    for question, prompt, prediction, was_generated in zip(
        questions, prompts, predictions, generated_flags
    ):
        if config.compute_nll:
            diagnostics = answer_diagnostics(
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                prompt=prompt,
                answer=question.nll_target,
                device=config.device,
                original_seq_len=original_seq_len,
            )
        else:
            diagnostics = {
                "answer_perplexity": float("nan"),
                "answer_nll": float("nan"),
                "answer_tokens": 0,
                "first_token_entropy": float("nan"),
                "first_token_margin": float("nan"),
                "reference_first_token_probability": float("nan"),
                "reference_first_token_rank": float("nan"),
                "post_answer_eos_probability": float("nan"),
                "post_answer_eos_rank": float("nan"),
                "post_answer_eos_logit_gap": float("nan"),
            }

        if was_generated:
            references = question.references
            em = best_exact_match(prediction, references)
            f1 = best_token_f1(prediction, references)
            rouge = best_rouge_l(prediction, references)
            current_ruler_score = (
                ruler_score(prediction, references, question.task)
                if question.dataset == "ruler"
                else float("nan")
            )
            current_quality_accuracy = (
                quality_score(prediction, question.answer)
                if question.dataset == "quality"
                else float("nan")
            )
            generated_tokens = count_generated_tokens(tokenizer, prediction)
        else:
            em, f1, rouge = float("nan"), float("nan"), float("nan")
            current_ruler_score = float("nan")
            current_quality_accuracy = float("nan")
            generated_tokens = 0

        rows.append(
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "entry_ratio": condition.entry_ratio,
                "key_bits": condition.key_bits,
                "value_bits": condition.value_bits,
                "ideal_budget_fraction": condition.ideal_budget_fraction,
                "composition_mode": condition.composition_mode,
                "question_id": question.question_id,
                "dataset": question.dataset,
                "context_id": context_id,
                "task": question.task,
                "author_idx": question.author_idx,
                "qa_idx": question.qa_idx,
                "question": question.question,
                "reference": question.answer,
                "reference_aliases": list(question.answer_aliases),
                "scoring_references": list(question.references),
                "teacher_forced_reference": question.nll_target,
                "prediction": prediction,
                **diagnostics,
                "generated_tokens": generated_tokens,
                "exact_match": em,
                "token_f1": f1,
                "rouge_l": rouge,
                "ruler_score": current_ruler_score,
                "quality_accuracy": current_quality_accuracy,
                "was_generated": was_generated,
            }
        )

    weighted_nll_numerator = sum(
        row["answer_nll"] * row["answer_tokens"]
        for row in rows
        if math.isfinite(row["answer_nll"])
    )
    weighted_nll_denominator = sum(
        row["answer_tokens"] for row in rows if math.isfinite(row["answer_nll"])
    )
    mean_nll = (
        weighted_nll_numerator / weighted_nll_denominator
        if weighted_nll_denominator
        else float("nan")
    )
    aggregate = {
        "answer_nll": mean_nll,
        "answer_perplexity": (
            math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan")
        ),
        "exact_match": finite_mean(row["exact_match"] for row in rows),
        "token_f1": finite_mean(row["token_f1"] for row in rows),
        "rouge_l": finite_mean(row["rouge_l"] for row in rows),
        "ruler_score": finite_mean(row["ruler_score"] for row in rows),
        "quality_accuracy": finite_mean(row["quality_accuracy"] for row in rows),
        "first_token_entropy": finite_mean(row["first_token_entropy"] for row in rows),
        "first_token_margin": finite_mean(row["first_token_margin"] for row in rows),
        "reference_first_token_probability": finite_mean(
            row["reference_first_token_probability"] for row in rows
        ),
        "reference_first_token_rank": finite_mean(
            row["reference_first_token_rank"] for row in rows
        ),
        "post_answer_eos_probability": finite_mean(
            row["post_answer_eos_probability"] for row in rows
        ),
        "post_answer_eos_rank": finite_mean(
            row["post_answer_eos_rank"] for row in rows
        ),
        "post_answer_eos_logit_gap": finite_mean(
            row["post_answer_eos_logit_gap"] for row in rows
        ),
        "mean_generated_tokens": (
            finite_mean(row["generated_tokens"] for row in rows)
            if any(generated_flags)
            else float("nan")
        ),
        "num_nll_questions": sum(
            1 for row in rows if math.isfinite(row["answer_nll"])
        ),
        "num_generated_questions": sum(generated_flags),
    }
    dataset = questions[0].dataset if questions else ""
    aggregate["primary_accuracy"] = {
        "ruler": aggregate["ruler_score"],
        "quality": aggregate["quality_accuracy"],
    }.get(dataset, aggregate["token_f1"])
    return rows, aggregate


def _aggregate_prediction_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    weighted_nll_numerator = sum(
        row["answer_nll"] * row["answer_tokens"]
        for row in rows
        if math.isfinite(row["answer_nll"])
    )
    weighted_nll_denominator = sum(
        row["answer_tokens"] for row in rows if math.isfinite(row["answer_nll"])
    )
    answer_nll = (
        weighted_nll_numerator / weighted_nll_denominator
        if weighted_nll_denominator
        else float("nan")
    )
    metric_names = (
        "exact_match",
        "token_f1",
        "rouge_l",
        "ruler_score",
        "quality_accuracy",
        "first_token_entropy",
        "first_token_margin",
        "reference_first_token_probability",
        "reference_first_token_rank",
        "post_answer_eos_probability",
        "post_answer_eos_rank",
        "post_answer_eos_logit_gap",
    )
    result = {
        "answer_nll": answer_nll,
        "answer_perplexity": (
            math.exp(answer_nll) if math.isfinite(answer_nll) else float("nan")
        ),
        **{
            name: finite_mean(row.get(name, float("nan")) for row in rows)
            for name in metric_names
        },
        "mean_generated_tokens": finite_mean(
            row["generated_tokens"] for row in rows if row.get("was_generated")
        ),
        "num_nll_questions": sum(
            math.isfinite(row["answer_nll"]) for row in rows
        ),
        "num_generated_questions": sum(bool(row.get("was_generated")) for row in rows),
    }
    datasets = {str(row.get("dataset")) for row in rows}
    result["primary_accuracy"] = (
        result["ruler_score"]
        if datasets == {"ruler"}
        else result["quality_accuracy"]
        if datasets == {"quality"}
        else result["token_f1"]
    )
    if datasets == {"ruler"}:
        grouped_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            task = str(row.get("task", "unknown"))
            report_group = "qa" if task.startswith("qa_") else task
            grouped_tasks[report_group].append(row)
        task_metrics: dict[str, dict[str, Any]] = {}
        for report_group, task_rows in sorted(grouped_tasks.items()):
            task_generated = [row for row in task_rows if row.get("was_generated")]
            nll_tokens = sum(
                int(row["answer_tokens"])
                for row in task_rows
                if math.isfinite(row["answer_nll"])
            )
            task_nll = (
                sum(
                    float(row["answer_nll"]) * int(row["answer_tokens"])
                    for row in task_rows
                    if math.isfinite(row["answer_nll"])
                )
                / nll_tokens
                if nll_tokens
                else float("nan")
            )
            task_score = finite_mean(
                row["ruler_score"] for row in task_generated
            )
            task_metrics[report_group] = {
                "source_tasks": sorted(
                    {str(row.get("task", "unknown")) for row in task_rows}
                ),
                "num_questions": len(task_rows),
                "num_generated_questions": len(task_generated),
                "ruler_score": task_score,
                "answer_nll": task_nll,
            }
            result[f"ruler_{report_group}_score"] = task_score
            result[f"ruler_{report_group}_answer_nll"] = task_nll
            result[f"ruler_{report_group}_num_questions"] = len(task_rows)
        result["ruler_task_metrics"] = task_metrics
    return result


def _mean(values: Iterable[float | int]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float("nan")


def run_experiment(config: RunConfig) -> Path:
    """Run an experiment and leave a terminal manifest even after a failure."""

    output_root = Path(config.output_root)
    existing_runs = set(output_root.iterdir()) if output_root.exists() else set()
    wrapper_start = time.perf_counter()
    try:
        return _run_experiment_impl(config)
    except Exception as error:
        if output_root.exists():
            new_runs = sorted(
                (
                    path
                    for path in output_root.iterdir()
                    if path not in existing_runs and (path / "manifest.json").exists()
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if new_runs:
                manifest_path = new_runs[0] / "manifest.json"
                try:
                    failed_manifest = json.loads(manifest_path.read_text())
                    timing = failed_manifest.setdefault("timing", {})
                    timing["wall_seconds"] = time.perf_counter() - wrapper_start
                    failed_manifest.update(
                        {
                            "status": "failed",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    _write_json(manifest_path, failed_manifest)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
        raise


def _run_experiment_impl(config: RunConfig) -> Path:
    run_start = time.perf_counter()
    if not config.compute_nll and not config.generate:
        raise ValueError("At least one of NLL evaluation or generation must be enabled.")
    if config.generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive.")
    if config.group_size <= 0 or config.max_queries_per_head <= 0:
        raise ValueError("group_size and max_queries_per_head must be positive.")
    if config.generate and config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive when generation is enabled.")
    if config.aware_refinement_iterations < 0:
        raise ValueError("aware_refinement_iterations must be non-negative.")

    data_loading_start = time.perf_counter()
    conditions = build_grid(config)
    _validate_conditions(conditions)
    corpora = load_evaluation_corpora(
        config.dataset_name,
        biographies_path=config.biographies_path,
        passages_path=config.passages_path,
        dataset_path=config.dataset_path,
        author_ids=config.author_ids,
        corpus_format=config.corpus_format,
        max_questions=config.max_questions,
        max_contexts=config.max_contexts,
        seed=config.seed,
    )
    questions = [question for corpus in corpora for question in corpus.questions]
    data_loading_seconds = time.perf_counter() - data_loading_start
    if not questions:
        raise ValueError(f"Dataset {config.dataset_name!r} produced no questions.")
    output_dir = _create_output_dir(config)
    seed_everything(config.seed)

    generation_ids: set[str] | None = None
    if config.generate and config.max_generation_questions is not None:
        ordered_ids = [question.question_id for question in questions]
        random_source = random.Random(config.seed + 10_003)
        random_source.shuffle(ordered_ids)
        generation_ids = set(ordered_ids[: config.max_generation_questions])

    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": asdict(config),
        "dataset": config.dataset_name,
        "num_contexts": len(corpora),
        "num_questions": len(questions),
        "num_generation_questions_planned": (
            len(generation_ids) if generation_ids is not None else len(questions)
        ),
        "conditions": [condition.to_dict() for condition in conditions],
        "compaction_configuration": {
            "compressed_method": "AM-HighestAttnKeys",
            "query_generation_method": "context_prefill",
            "alternative_compaction_methods_enabled": False,
            "score_method": "rms",
            "beta_fit": "bounded_nnls",
            "value_fit": "least_squares",
            "max_queries_per_kv_head": config.max_queries_per_head,
        },
        "timing": {
            "data_loading_seconds": data_loading_seconds,
            "model_loading_seconds": 0.0,
            "context_prefill_seconds": 0.0,
            "dense_accounting_seconds": 0.0,
            "query_generation_seconds": 0.0,
            "input_quantization_seconds": 0.0,
            "compaction_seconds_unique": 0.0,
            "kv_qdq_seconds": 0.0,
            "evaluation_seconds": 0.0,
        },
        "precision_note": (
            "INT8/INT4 KV uses QDQ: attention consumes BF16 tensors whose values are "
            "restricted to the requested quantization grid. logical_packed_kv_bytes "
            "is the packed occupancy estimate (payload + BF16 scales + nonzero AM beta); "
            "resident_qdq_kv_bytes is the larger allocation used by this simulator. "
            "INT8/NF4 model weights use real BitsAndBytes quantized linear modules."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)

    print(f"Loading {config.model} ({config.weight_precision} weights) ...", flush=True)
    model_loading_start = time.perf_counter()
    model, tokenizer, weight_footprint_bytes = load_model_and_tokenizer(
        config.model, config.device, config.weight_precision
    )
    weight_audit = audit_model_quantization(
        model, config.weight_precision, weight_footprint_bytes
    )
    if not weight_audit["quantized_execution_verified"]:
        raise RuntimeError(
            f"Model weight audit failed for {config.weight_precision}: {weight_audit}"
        )
    manifest["timing"]["model_loading_seconds"] = (
        time.perf_counter() - model_loading_start
    )
    manifest["model_weight_audit"] = weight_audit
    manifest["model_weight_footprint_bytes"] = weight_footprint_bytes
    _write_json(output_dir / "manifest.json", manifest)

    accumulated_rows: dict[str, list[dict[str, Any]]] = {
        condition.condition_id: [] for condition in conditions
    }
    accumulated_stats: dict[str, list[dict[str, Any]]] = {
        condition.condition_id: [] for condition in conditions
    }
    context_manifest: list[dict[str, Any]] = []

    for context_number, corpus in enumerate(corpora, start=1):
        print(
            f"Context {context_number}/{len(corpora)}: {corpus.context_id}", flush=True
        )
        prompts = format_qa_prompts(
            tokenizer,
            [question.question for question in corpus.questions],
            config.model,
            dataset_name=corpus.dataset_name,
            answer_prefixes=[question.answer_prefix for question in corpus.questions],
        )
        context_prefill_start = time.perf_counter()
        (
            seq_len,
            past_key_values,
            article_indices,
            formatted_context,
            source_token_length,
        ) = extract_context_cache(
            model=model,
            tokenizer=tokenizer,
            corpus_text=corpus.text,
            device=config.device,
            model_name=config.model,
        )
        context_prefill_seconds = time.perf_counter() - context_prefill_start
        manifest["timing"]["context_prefill_seconds"] += context_prefill_seconds
        full_cache = dense_cache_to_compacted(past_key_values)
        dense_accounting_start = time.perf_counter()
        _, dense_stats = quantize_compacted_cache(
            full_cache,
            key_bits=16,
            value_bits=16,
            group_size=config.group_size,
            quantizer=config.kv_quantizer,
        )
        dense_accounting_seconds = time.perf_counter() - dense_accounting_start
        manifest["timing"]["dense_accounting_seconds"] += dense_accounting_seconds
        dense_bf16_bytes = dense_stats.effective_packed_bytes
        safe_context_id = _safe_slug(corpus.context_id) or f"context-{context_number}"
        context_stats_dir = output_dir / "compaction_stats" / safe_context_id
        context_stats_dir.mkdir(exist_ok=True)
        context_manifest.append(
            {
                "context_id": corpus.context_id,
                "dataset": corpus.dataset_name,
                "num_questions": len(corpus.questions),
                "formatted_context_tokens": seq_len,
                "source_token_length": source_token_length,
                "article_tokens": len(article_indices),
                "dense_bf16_kv_bytes": dense_bf16_bytes,
                "context_prefill_seconds": context_prefill_seconds,
                "dense_accounting_seconds": dense_accounting_seconds,
                "input_quantization_seconds": 0.0,
                "compaction_seconds_unique": 0.0,
                "kv_qdq_seconds": 0.0,
                "evaluation_seconds": 0.0,
            }
        )
        manifest["contexts"] = context_manifest
        _write_json(output_dir / "manifest.json", manifest)

        precomputed_queries = None
        precomputed_query_stats = None
        if any(
            condition.method == "am"
            for condition in conditions
        ):
            query_start = time.perf_counter()
            precomputed_queries, precomputed_query_stats = extract_compaction_queries(
                past_key_values_for_queries=past_key_values,
                reference_cache=full_cache,
                article_indices=article_indices,
                formatted_context=formatted_context,
                model=model,
                tokenizer=tokenizer,
                max_queries_per_head=config.max_queries_per_head,
            )
            precomputed_query_stats = {
                **precomputed_query_stats,
                "reused_across_compactions": True,
                "seconds": time.perf_counter() - query_start,
            }
            context_manifest[-1]["query_generation"] = precomputed_query_stats
            context_manifest[-1]["query_bank_shape"] = list(precomputed_queries.shape)
            context_manifest[-1]["query_generation_verified"] = (
                precomputed_queries.shape[-2] > 0
            )
            manifest["timing"]["query_generation_seconds"] += precomputed_query_stats[
                "seconds"
            ]
            _write_json(output_dir / "manifest.json", manifest)

        for group_key, group_conditions in _condition_groups(conditions):
            method, entry_ratio, composition_mode, _, _ = group_key
            representative = group_conditions[0]
            compaction_seconds = 0.0
            target_article_tokens = len(article_indices)
            input_quantization_stats = None
            input_quantization_seconds = 0.0
            if method == "dense":
                cardinality_cache = full_cache
                compaction_stats: dict[str, Any] = {"method": "dense"}
            else:
                compaction_source: Any = past_key_values
                past_key_values_for_queries = None
                if composition_mode == "pre":
                    input_quantization_start = time.perf_counter()
                    compaction_source, input_quantization_stats = quantize_compacted_cache(
                        full_cache,
                        key_bits=representative.key_bits,
                        value_bits=representative.value_bits,
                        group_size=config.group_size,
                        quantizer=config.kv_quantizer,
                    )
                    input_quantization_seconds = (
                        time.perf_counter() - input_quantization_start
                    )
                    manifest["timing"][
                        "input_quantization_seconds"
                    ] += input_quantization_seconds
                    context_manifest[-1][
                        "input_quantization_seconds"
                    ] += input_quantization_seconds
                    past_key_values_for_queries = past_key_values
                start = time.perf_counter()
                cardinality_cache, compaction_stats, target_article_tokens = compact_cache(
                    method=method,
                    entry_ratio=entry_ratio,
                    past_key_values=compaction_source,
                    seq_len=seq_len,
                    article_indices=article_indices,
                    formatted_context=formatted_context,
                    model=model,
                    tokenizer=tokenizer,
                    max_queries_per_head=config.max_queries_per_head,
                    save_detailed_stats=config.save_detailed_compaction_stats,
                    composition_mode=composition_mode,
                    key_bits=representative.key_bits,
                    value_bits=representative.value_bits,
                    group_size=config.group_size,
                    quantizer=config.kv_quantizer,
                    aware_refinement_iterations=config.aware_refinement_iterations,
                    past_key_values_for_queries=past_key_values_for_queries,
                    precomputed_queries=precomputed_queries,
                    precomputed_query_stats=precomputed_query_stats,
                )
                compaction_seconds = time.perf_counter() - start
                manifest["timing"][
                    "compaction_seconds_unique"
                ] += compaction_seconds
                context_manifest[-1][
                    "compaction_seconds_unique"
                ] += compaction_seconds
                if composition_mode == "pre":
                    del compaction_source
                    clear_device_cache()

            ratio_slug = f"{entry_ratio:.6f}".rstrip("0").rstrip(".")
            precision_slug = (
                f"_k{representative.key_bits}v{representative.value_bits}"
                if composition_mode != "post"
                else ""
            )
            compaction_stats_path = context_stats_dir / (
                f"{method}_r{ratio_slug}_{composition_mode}{precision_slug}.json"
            )
            _write_json(
                compaction_stats_path,
                {
                    "context_id": corpus.context_id,
                    "method": method,
                    "entry_ratio": entry_ratio,
                    "composition_mode": composition_mode,
                    "input_quantization": (
                        input_quantization_stats.to_dict()
                        if input_quantization_stats is not None
                        else None
                    ),
                    "target_article_tokens": target_article_tokens,
                    "input_quantization_seconds": input_quantization_seconds,
                    "compaction_seconds": compaction_seconds,
                    "stats": compaction_stats,
                },
            )

            for condition in group_conditions:
                print(
                    f"Evaluating {corpus.context_id} / {condition.condition_id} ...",
                    flush=True,
                )
                qdq_start = time.perf_counter()
                qdq_cache, cache_stats = quantize_compacted_cache(
                    cardinality_cache,
                    key_bits=condition.key_bits,
                    value_bits=condition.value_bits,
                    group_size=config.group_size,
                    quantizer=config.kv_quantizer,
                )
                qdq_seconds = time.perf_counter() - qdq_start
                manifest["timing"]["kv_qdq_seconds"] += qdq_seconds
                context_manifest[-1]["kv_qdq_seconds"] += qdq_seconds
                if not cache_stats.quantization_fixed_point_verified:
                    raise RuntimeError(
                        f"KV QDQ fixed-point audit failed for {condition.condition_id}: "
                        f"max error "
                        f"{cache_stats.quantization_fixed_point_max_abs_error} exceeds "
                        f"tolerance {cache_stats.quantization_fixed_point_tolerance}"
                    )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    cuda_before = int(torch.cuda.memory_allocated())
                else:
                    cuda_before = 0
                evaluation_start = time.perf_counter()
                rows, _ = _evaluate_condition(
                    config=config,
                    condition=condition,
                    model=model,
                    tokenizer=tokenizer,
                    cache=qdq_cache,
                    prompts=prompts,
                    questions=corpus.questions,
                    original_seq_len=seq_len,
                    context_id=corpus.context_id,
                    generate_question_ids=generation_ids,
                )
                evaluation_seconds = time.perf_counter() - evaluation_start
                manifest["timing"]["evaluation_seconds"] += evaluation_seconds
                context_manifest[-1]["evaluation_seconds"] += evaluation_seconds
                cuda_peak = (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0
                )
                accumulated_rows[condition.condition_id].extend(rows)
                context_stat = {
                    "context_id": corpus.context_id,
                    "formatted_context_tokens": seq_len,
                    "article_tokens": len(article_indices),
                    "target_article_tokens": target_article_tokens,
                    "realized_article_entry_ratio": (
                        target_article_tokens / len(article_indices)
                        if article_indices
                        else 0.0
                    ),
                    "dense_bf16_kv_bytes": dense_bf16_bytes,
                    "input_quantization_seconds": input_quantization_seconds,
                    "compaction_seconds": compaction_seconds,
                    "kv_qdq_seconds": qdq_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "cuda_allocated_before_evaluation_bytes": cuda_before,
                    "cuda_peak_allocated_bytes": cuda_peak,
                    "input_key_mse": (
                        input_quantization_stats.key_mse
                        if input_quantization_stats is not None
                        else float("nan")
                    ),
                    "input_value_mse": (
                        input_quantization_stats.value_mse
                        if input_quantization_stats is not None
                        else float("nan")
                    ),
                    "input_quantization": (
                        input_quantization_stats.to_dict()
                        if input_quantization_stats is not None
                        else None
                    ),
                    **cache_stats.to_dict(),
                }
                accumulated_stats[condition.condition_id].append(context_stat)
                _write_jsonl(
                    output_dir / "predictions" / f"{condition.condition_id}.jsonl",
                    accumulated_rows[condition.condition_id],
                )
                _write_json(
                    output_dir / "condition_stats" / f"{condition.condition_id}.json",
                    {
                        "status": "running",
                        "condition": condition.to_dict(),
                        "model_weight_audit": weight_audit,
                        "contexts_completed": accumulated_stats[condition.condition_id],
                        "predictions_file": str(
                            Path("predictions") / f"{condition.condition_id}.jsonl"
                        ),
                    },
                )
                del qdq_cache
                clear_device_cache()

            if method not in ("dense",):
                del cardinality_cache
                clear_device_cache()

        del full_cache, past_key_values
        clear_device_cache()
        manifest["contexts"] = context_manifest
        _write_json(output_dir / "manifest.json", manifest)

    summaries: list[dict[str, Any]] = []
    for condition in conditions:
        rows = accumulated_rows[condition.condition_id]
        stats = accumulated_stats[condition.condition_id]
        aggregate = _aggregate_prediction_rows(rows)
        mean_packed_bytes = _mean(item["effective_packed_bytes"] for item in stats)
        packed_bytes = max(item["effective_packed_bytes"] for item in stats)
        mean_resident_bytes = _mean(item["actual_tensor_bytes"] for item in stats)
        resident_bytes = max(item["actual_tensor_bytes"] for item in stats)
        dense_bytes = _mean(item["dense_bf16_kv_bytes"] for item in stats)
        summary = {
            **condition.to_dict(),
            "dataset": config.dataset_name,
            "model": config.model,
            "compaction_algorithm": (
                "AM-HighestAttnKeys"
                if condition.method == "am"
                else "uncompacted_dense"
            ),
            "query_generation_method": "context_prefill",
            "weight_precision": config.weight_precision,
            "weight_nominal_bits": weight_audit["nominal_storage_bits"],
            "model_weight_quantization_verified": weight_audit[
                "quantized_execution_verified"
            ],
            "query_generation_verified": all(
                context.get("query_generation_verified", True)
                for context in context_manifest
            ),
            "min_queries_per_kv_head": min(
                (
                    context["query_bank_shape"][-2]
                    for context in context_manifest
                    if "query_bank_shape" in context
                ),
                default=0,
            ),
            "max_queries_per_kv_head": max(
                (
                    context["query_bank_shape"][-2]
                    for context in context_manifest
                    if "query_bank_shape" in context
                ),
                default=0,
            ),
            "quantized_linear_8bit_modules": weight_audit[
                "quantized_linear_8bit_modules"
            ],
            "quantized_linear_4bit_modules": weight_audit[
                "quantized_linear_4bit_modules"
            ],
            "num_contexts": len(stats),
            "num_questions": len(rows),
            "mean_formatted_context_tokens": _mean(
                item["formatted_context_tokens"] for item in stats
            ),
            "max_formatted_context_tokens": max(
                item["formatted_context_tokens"] for item in stats
            ),
            "mean_article_tokens": _mean(item["article_tokens"] for item in stats),
            "mean_target_article_tokens": _mean(
                item["target_article_tokens"] for item in stats
            ),
            "realized_article_entry_ratio": _mean(
                item["realized_article_entry_ratio"] for item in stats
            ),
            "mean_num_stored_entries": _mean(
                item["num_stored_entries"] for item in stats
            ),
            "min_layer_length": min(item["min_layer_length"] for item in stats),
            "max_layer_length": max(item["max_layer_length"] for item in stats),
            "key_payload_bytes": _mean(item["key_payload_bytes"] for item in stats),
            "value_payload_bytes": _mean(
                item["value_payload_bytes"] for item in stats
            ),
            "scale_metadata_bytes": _mean(
                item["scale_metadata_bytes"] for item in stats
            ),
            "key_scale_metadata_bytes": _mean(
                item["key_scale_bytes"] for item in stats
            ),
            "value_scale_metadata_bytes": _mean(
                item["value_scale_bytes"] for item in stats
            ),
            "beta_bytes": _mean(item["beta_bytes"] for item in stats),
            "mean_logical_packed_kv_bytes": mean_packed_bytes,
            "logical_packed_kv_bytes": packed_bytes,
            "effective_packed_kv_bytes": packed_bytes,
            "max_logical_packed_kv_bytes": packed_bytes,
            "mean_resident_qdq_kv_bytes": mean_resident_bytes,
            "resident_qdq_kv_bytes": resident_bytes,
            "actual_qdq_tensor_bytes": resident_bytes,
            "effective_budget_vs_dense_bf16": mean_packed_bytes / dense_bytes,
            "kv_storage_is_simulated": any(
                item["storage_is_simulated"] for item in stats
            ),
            "kv_packed_storage_materialized": not any(
                item["storage_is_simulated"] for item in stats
            ),
            "kv_attention_compute_dtype": stats[0]["compute_dtype"],
            "kv_quantization_verified": all(
                item["quantization_fixed_point_verified"] for item in stats
            ),
            "kv_attention_uses_requested_qdq_values": all(
                item["quantization_fixed_point_verified"] for item in stats
            ),
            "kv_quantization_fixed_point_max_abs_error": max(
                item["quantization_fixed_point_max_abs_error"] for item in stats
            ),
            "kv_quantization_fixed_point_tolerance": max(
                item["quantization_fixed_point_tolerance"] for item in stats
            ),
            "key_mse": _mean(item["key_mse"] for item in stats),
            "value_mse": _mean(item["value_mse"] for item in stats),
            "input_key_mse": finite_mean(item["input_key_mse"] for item in stats),
            "input_value_mse": finite_mean(
                item["input_value_mse"] for item in stats
            ),
            "input_quantization_applied": any(
                item["input_quantization"] is not None for item in stats
            ),
            "input_quantization_verified": all(
                item["input_quantization"] is None
                or item["input_quantization"][
                    "quantization_fixed_point_verified"
                ]
                for item in stats
            ),
            "weight_footprint_bytes": weight_footprint_bytes,
            "model_parameter_storage_bytes": weight_audit["parameter_storage_bytes"],
            "model_buffer_storage_bytes": weight_audit["buffer_storage_bytes"],
            "model_plus_logical_packed_kv_bytes": weight_footprint_bytes + packed_bytes,
            "estimated_weight_plus_packed_kv_bytes": weight_footprint_bytes
            + packed_bytes,
            "model_plus_resident_qdq_kv_bytes": weight_footprint_bytes
            + resident_bytes,
            "mean_cuda_allocated_before_evaluation_bytes": _mean(
                item["cuda_allocated_before_evaluation_bytes"] for item in stats
            ),
            "max_cuda_peak_allocated_bytes": max(
                item["cuda_peak_allocated_bytes"] for item in stats
            ),
            "compaction_seconds": sum(item["compaction_seconds"] for item in stats),
            "input_quantization_seconds": sum(
                item["input_quantization_seconds"] for item in stats
            ),
            "kv_qdq_seconds": sum(item["kv_qdq_seconds"] for item in stats),
            "evaluation_seconds": sum(
                item["evaluation_seconds"] for item in stats
            ),
            **aggregate,
            "generated_length_vs_dense_bf16": float("nan"),
        }
        summaries.append(summary)

    dense = next(
        (
            row
            for row in summaries
            if row["method"] == "dense"
            and row["key_bits"] == 16
            and row["value_bits"] == 16
        ),
        None,
    )
    if dense is not None and math.isfinite(dense["mean_generated_tokens"]):
        for row in summaries:
            if math.isfinite(row["mean_generated_tokens"]):
                row["generated_length_vs_dense_bf16"] = (
                    row["mean_generated_tokens"] / dense["mean_generated_tokens"]
                    if dense["mean_generated_tokens"] > 0
                    else float("nan")
                )
    _write_summary(output_dir, summaries)
    for condition in conditions:
        condition_stats_path = (
            output_dir / "condition_stats" / f"{condition.condition_id}.json"
        )
        saved = json.loads(condition_stats_path.read_text())
        saved.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "aggregate": next(
                    row for row in summaries if row["condition_id"] == condition.condition_id
                ),
            }
        )
        _write_json(condition_stats_path, saved)

    wall_seconds = time.perf_counter() - run_start
    attributed_seconds = sum(
        float(value) for value in manifest["timing"].values()
    )
    manifest["timing"].update(
        {
            "wall_seconds": wall_seconds,
            "other_overhead_seconds": max(0.0, wall_seconds - attributed_seconds),
        }
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "num_conditions_completed": len(summaries),
            "contexts": context_manifest,
        }
    )
    _write_json(output_dir / "manifest.json", manifest)
    print(
        f"Complete in {wall_seconds:.1f}s. Results: {output_dir}", flush=True
    )
    return output_dir
