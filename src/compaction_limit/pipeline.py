from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .data import DEFAULT_BIOGRAPHIES_PATH, DEFAULT_PASSAGES_PATH
from .runner import RunConfig, run_experiment
from .suites import (
    build_pipeline_plan,
    build_stage1_suite,
    build_stage2_suite,
    build_stage3_funnel,
    select_stage1,
    select_stage2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULER_PATH = PROJECT_ROOT / "data" / "benchmarks" / "ruler_16k.jsonl"
DEFAULT_QUALITY_PATH = (
    PROJECT_ROOT / "data" / "benchmarks" / "quality_dev.jsonl"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, default=_json_default, allow_nan=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def _parse_ruler_task_samples(values: Sequence[str]) -> dict[str, int]:
    samples: dict[str, int] = {}
    for value in values:
        try:
            task, count_text = value.split("=", maxsplit=1)
            count = int(count_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid RULER task sample {value!r}; expected task=count."
            ) from error
        if task not in ("fwe", "qa_1", "qa_2") or count <= 0:
            raise ValueError(
                "RULER task samples must be positive counts for fwe, qa_1, and qa_2."
            )
        samples[task] = count
    if set(samples) != {"fwe", "qa_1", "qa_2"}:
        raise ValueError("RULER task samples must specify fwe, qa_1, and qa_2.")
    return samples


def _audit_ruler_subset(path: Path, expected_samples: dict[str, int]) -> None:
    rows = _read_jsonl(path)
    observed = Counter(str(row.get("task")) for row in rows)
    if dict(observed) != expected_samples:
        raise ValueError(
            f"RULER subset {path} has task counts {dict(observed)}, expected "
            f"exactly {expected_samples}. Re-run benchmark preparation."
        )


def _condition_ids(config: RunConfig) -> list[str]:
    return [condition.condition_id for condition in config.conditions or ()]


def _config_payload(config: RunConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), default=_json_default))


def _find_complete_run(output_root: Path, config: RunConfig) -> Path | None:
    expected_ids = _condition_ids(config)
    expected_config = _config_payload(config)
    if not output_root.exists():
        return None
    for run_dir in sorted(output_root.iterdir(), reverse=True):
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.jsonl"
        if (
            not run_dir.is_dir()
            or not manifest_path.exists()
            or not summary_path.exists()
        ):
            continue
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        saved_ids = [
            condition.get("condition_id")
            for condition in manifest.get("conditions", [])
        ]
        if (
            manifest.get("status") == "complete"
            and manifest.get("config", {}) == expected_config
            and saved_ids == expected_ids
        ):
            return run_dir
    return None


def _run_or_resume(config: RunConfig, resume: bool) -> Path:
    if resume:
        complete = _find_complete_run(Path(config.output_root), config)
        if complete is not None:
            print(f"Reusing complete run: {complete}", flush=True)
            return complete
    return run_experiment(config)


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _dense_baselines(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    by_weight: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    bf16: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not (
            row.get("method") == "dense"
            and row.get("key_bits") == 16
            and row.get("value_bits") == 16
        ):
            continue
        stage = str(row.get("stage", "unknown"))
        phase = str(row.get("phase") or "")
        dataset = str(row.get("dataset", "tofu"))
        weight = str(row.get("weight_precision", "unknown"))
        by_weight[(stage, phase, dataset, weight)] = row
        if weight == "bf16":
            bf16[(stage, phase, dataset)] = row
    return by_weight, bf16


def _add_comparison_columns(rows: list[dict[str, Any]]) -> None:
    by_weight, bf16 = _dense_baselines(rows)
    bf16_by_condition = {
        (
            str(row.get("stage", "unknown")),
            str(row.get("phase") or ""),
            str(row.get("dataset", "tofu")),
            str(row.get("condition_id", "unknown")),
        ): row
        for row in rows
        if row.get("weight_precision") == "bf16"
    }

    for row in rows:
        stage = str(row.get("stage", "unknown"))
        phase = str(row.get("phase") or "")
        dataset = str(row.get("dataset", "tofu"))
        weight = str(row.get("weight_precision", "unknown"))
        dense = by_weight.get((stage, phase, dataset, weight))
        bf16_dense = bf16.get((stage, phase, dataset))
        same_kv_bf16 = bf16_by_condition.get(
            (stage, phase, dataset, str(row.get("condition_id", "unknown")))
        )
        row["answer_nll_delta_vs_same_weight_dense"] = float("nan")
        row["primary_accuracy_drop_vs_same_weight_dense"] = float("nan")
        row["answer_nll_delta_vs_bf16_dense"] = float("nan")
        row["primary_accuracy_drop_vs_bf16_dense"] = float("nan")
        row["answer_nll_delta_vs_same_kv_bf16"] = float("nan")
        row["primary_accuracy_drop_vs_same_kv_bf16"] = float("nan")
        row["total_memory_fraction_vs_same_weight_dense"] = float("nan")
        row["total_memory_fraction_vs_bf16_dense"] = float("nan")
        row["kv_memory_fraction_vs_bf16_dense"] = float("nan")
        row["model_memory_fraction_vs_bf16_dense"] = float("nan")
        row["total_memory_saved_bytes_vs_bf16_dense"] = float("nan")
        for ruler_group in ("fwe", "qa"):
            row[f"ruler_{ruler_group}_score_drop_vs_same_weight_dense"] = float(
                "nan"
            )
            row[f"ruler_{ruler_group}_answer_nll_delta_vs_same_weight_dense"] = (
                float("nan")
            )
        if dense is not None:
            if _finite(row.get("answer_nll")) and _finite(dense.get("answer_nll")):
                row["answer_nll_delta_vs_same_weight_dense"] = (
                    float(row["answer_nll"]) - float(dense["answer_nll"])
                )
            if _finite(row.get("primary_accuracy")) and _finite(
                dense.get("primary_accuracy")
            ):
                row["primary_accuracy_drop_vs_same_weight_dense"] = (
                    float(dense["primary_accuracy"])
                    - float(row["primary_accuracy"])
                )
            total = row.get("model_plus_logical_packed_kv_bytes")
            dense_total = dense.get("model_plus_logical_packed_kv_bytes")
            if _finite(total) and _finite(dense_total) and float(dense_total) > 0:
                row["total_memory_fraction_vs_same_weight_dense"] = float(
                    total
                ) / float(dense_total)
            for ruler_group in ("fwe", "qa"):
                score_name = f"ruler_{ruler_group}_score"
                nll_name = f"ruler_{ruler_group}_answer_nll"
                if _finite(row.get(score_name)) and _finite(dense.get(score_name)):
                    row[
                        f"ruler_{ruler_group}_score_drop_vs_same_weight_dense"
                    ] = float(dense[score_name]) - float(row[score_name])
                if _finite(row.get(nll_name)) and _finite(dense.get(nll_name)):
                    row[
                        f"ruler_{ruler_group}_answer_nll_delta_vs_same_weight_dense"
                    ] = float(row[nll_name]) - float(dense[nll_name])
        if bf16_dense is not None:
            total = row.get("model_plus_logical_packed_kv_bytes")
            baseline_total = bf16_dense.get("model_plus_logical_packed_kv_bytes")
            kv = row.get("logical_packed_kv_bytes")
            baseline_kv = bf16_dense.get("logical_packed_kv_bytes")
            model = row.get("weight_footprint_bytes")
            baseline_model = bf16_dense.get("weight_footprint_bytes")
            if _finite(row.get("answer_nll")) and _finite(
                bf16_dense.get("answer_nll")
            ):
                row["answer_nll_delta_vs_bf16_dense"] = float(
                    row["answer_nll"]
                ) - float(bf16_dense["answer_nll"])
            if _finite(row.get("primary_accuracy")) and _finite(
                bf16_dense.get("primary_accuracy")
            ):
                row["primary_accuracy_drop_vs_bf16_dense"] = float(
                    bf16_dense["primary_accuracy"]
                ) - float(row["primary_accuracy"])
            if _finite(total) and _finite(baseline_total) and float(baseline_total) > 0:
                row["total_memory_fraction_vs_bf16_dense"] = float(total) / float(
                    baseline_total
                )
                row["total_memory_saved_bytes_vs_bf16_dense"] = float(
                    baseline_total
                ) - float(total)
            if _finite(kv) and _finite(baseline_kv) and float(baseline_kv) > 0:
                row["kv_memory_fraction_vs_bf16_dense"] = float(kv) / float(
                    baseline_kv
                )
            if (
                _finite(model)
                and _finite(baseline_model)
                and float(baseline_model) > 0
            ):
                row["model_memory_fraction_vs_bf16_dense"] = float(model) / float(
                    baseline_model
                )

        if same_kv_bf16 is not None:
            if _finite(row.get("answer_nll")) and _finite(
                same_kv_bf16.get("answer_nll")
            ):
                row["answer_nll_delta_vs_same_kv_bf16"] = float(
                    row["answer_nll"]
                ) - float(same_kv_bf16["answer_nll"])
            if _finite(row.get("primary_accuracy")) and _finite(
                same_kv_bf16.get("primary_accuracy")
            ):
                row["primary_accuracy_drop_vs_same_kv_bf16"] = float(
                    same_kv_bf16["primary_accuracy"]
                ) - float(row["primary_accuracy"])


def aggregate_pipeline(pipeline_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    runs = []
    for summary_path in sorted(pipeline_dir.rglob("summary.jsonl")):
        run_dir = summary_path.parent
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        relative = run_dir.relative_to(pipeline_dir)
        if "runs" not in relative.parts or not relative.parts[0].startswith("stage"):
            continue
        stage = relative.parts[0]
        phase = None
        run_rows = _read_jsonl(summary_path)
        run_timing = manifest.get("timing", {})
        timing_columns = {
            f"run_{name}": value for name, value in run_timing.items()
        }
        for row in run_rows:
            rows.append(
                {
                    "stage": stage,
                    "phase": phase,
                    "run_dir": str(relative),
                    **timing_columns,
                    **row,
                }
            )
        runs.append(
            {
                "stage": stage,
                "phase": phase,
                "run_dir": str(relative),
                "status": manifest.get("status", "missing"),
                "dataset": manifest.get("config", {}).get("dataset_name"),
                "weight_precision": manifest.get("config", {}).get("weight_precision"),
                "conditions_completed": len(run_rows),
                "timing": run_timing,
            }
        )

    _add_comparison_columns(rows)
    jsonl_path = pipeline_dir / "combined_summary.jsonl"
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=True) + "\n")
    csv_path = pipeline_dir / "combined_summary.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")
    return {
        "num_rows": len(rows),
        "runs": runs,
        "completed_run_wall_seconds": sum(
            float(run.get("timing", {}).get("wall_seconds", 0.0))
            for run in runs
            if run["status"] == "complete"
        ),
    }


def _resolved_plan(
    stage1_conditions: Iterable[Any],
    stage2_conditions: Iterable[Any],
    stage3_conditions: Iterable[Any],
    weight_precisions: Sequence[str],
    datasets: Sequence[str] = ("tofu", "quality"),
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "funnel_selection": True,
        "stage1": {
            "selection_dependency": None,
            "datasets": list(datasets),
            "conditions": [condition.to_dict() for condition in stage1_conditions],
        },
        "stage2": {
            "selection_dependency": "Stage 1 stratified precision allocation",
            "datasets": list(datasets),
            "conditions": [condition.to_dict() for condition in stage2_conditions],
        },
        "stage3": {
            "selection_dependency": "best-performing Stage 2 family",
            "datasets": list(datasets),
            "weight_precisions": list(weight_precisions),
            "conditions": [condition.to_dict() for condition in stage3_conditions],
        },
    }
    return plan


def _stage_dataset_config(
    base_config: RunConfig,
    args: argparse.Namespace,
    *,
    dataset: str,
    weight_precision: str,
    conditions: Sequence[Any],
    tofu_questions: int,
    ruler_questions: int,
    quality_articles: int,
    tofu_generation_questions: int,
    ruler_generation_questions: int,
    queries_per_head: int,
    generate: bool,
    output_root: Path,
) -> RunConfig:
    if dataset == "tofu":
        return replace(
            base_config,
            dataset_name="tofu",
            dataset_path=None,
            author_ids=tuple(args.author_ids),
            weight_precision=weight_precision,
            max_questions=tofu_questions,
            max_generation_questions=tofu_generation_questions,
            max_queries_per_head=queries_per_head,
            generation_batch_size=args.generation_batch_size,
            generate=generate,
            conditions=tuple(conditions),
            output_root=output_root / "tofu" / "runs",
        )
    if dataset == "ruler":
        return replace(
            base_config,
            dataset_name="ruler",
            dataset_path=args.ruler_path,
            author_ids=(),
            weight_precision=weight_precision,
            max_questions=ruler_questions,
            max_generation_questions=ruler_generation_questions,
            max_queries_per_head=queries_per_head,
            generation_batch_size=args.ruler_generation_batch_size,
            generate=generate,
            conditions=tuple(conditions),
            output_root=output_root / "ruler" / "runs",
        )
    if dataset == "quality":
        return replace(
            base_config,
            dataset_name="quality",
            dataset_path=args.quality_path,
            author_ids=(),
            weight_precision=weight_precision,
            max_questions=None,
            max_contexts=quality_articles,
            max_generation_questions=None,
            max_queries_per_head=queries_per_head,
            generation_batch_size=args.quality_generation_batch_size,
            generate=generate,
            conditions=tuple(conditions),
            output_root=output_root / "quality" / "runs",
        )
    raise ValueError(f"Unsupported Stage 1-3 dataset: {dataset}")


def _runtime_pipeline_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_pipeline_plan()
    task_samples = _parse_ruler_task_samples(args.ruler_task_samples)
    stage_settings = {
        "stage1": (
            args.screen_questions,
            args.screen_ruler_questions,
            args.screen_generation_questions,
            args.screen_ruler_generation_questions,
            args.screen_queries_per_head,
            args.screen_generate,
        ),
        "stage2": (
            args.confirm_questions,
            args.confirm_ruler_questions,
            args.confirm_generation_questions,
            args.confirm_ruler_generation_questions,
            args.confirm_queries_per_head,
            args.confirm_generate,
        ),
        "stage3": (
            args.final_questions,
            args.final_ruler_questions,
            args.final_generation_questions,
            args.final_ruler_generation_questions,
            args.final_queries_per_head,
            args.final_generate,
        ),
    }
    for stage, (
        tofu_questions,
        ruler_questions,
        tofu_generation,
        ruler_generation,
        queries,
        generate,
    ) in stage_settings.items():
        questions: dict[str, Any] = {}
        generation_questions: dict[str, Any] = {}
        for dataset in args.datasets:
            if dataset == "tofu":
                questions[dataset] = tofu_questions
                generation_questions[dataset] = tofu_generation if generate else 0
            elif dataset == "ruler":
                questions[dataset] = ruler_questions
                generation_questions[dataset] = ruler_generation if generate else 0
            else:
                questions[dataset] = "all questions"
                generation_questions[dataset] = "all questions" if generate else 0
        plan[stage]["datasets"] = list(args.datasets)
        plan[stage]["questions"] = questions
        plan[stage]["generation_questions"] = generation_questions
        if "ruler" in args.datasets:
            plan[stage]["ruler_task_samples"] = task_samples
        if "quality" in args.datasets:
            plan[stage]["quality_articles"] = args.quality_articles
        plan[stage]["queries_per_kv_head"] = queries

    plan["stage3"]["weight_precisions"] = list(args.stage3_weight_precisions)
    return plan


def run_pipeline(args: argparse.Namespace) -> Path:
    pipeline_start = time.perf_counter()
    if "ruler" in args.datasets:
        _audit_ruler_subset(
            Path(args.ruler_path),
            _parse_ruler_task_samples(args.ruler_task_samples),
        )
    pipeline_dir = Path(args.pipeline_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    config_payload = json.loads(
        json.dumps(
            {
                name: value
                for name, value in vars(args).items()
                if name not in ("resume", "plan_only")
            },
            default=_json_default,
        )
    )
    manifest_path = pipeline_dir / "pipeline_manifest.json"
    if manifest_path.exists() and any(pipeline_dir.rglob("manifest.json")):
        previous_manifest = _read_json(manifest_path)
        previous_config = {
            name: value
            for name, value in previous_manifest.get("config", {}).items()
            if name not in ("resume", "plan_only")
        }
        if previous_config != config_payload:
            raise ValueError(
                f"Pipeline directory {pipeline_dir} already contains runs from a "
                "different configuration; choose a new PIPELINE_NAME to avoid "
                "mixing incompatible summaries."
            )
    stage1_conditions = build_stage1_suite()
    _write_json(pipeline_dir / "plan.json", _runtime_pipeline_plan(args))

    manifest: dict[str, Any] = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "funnel_selection": True,
        "design": (
            "Every stage uses context-prefill AM-HighestAttnKeys. Stage 1 crosses "
            "equal-byte budgets and fixed retention ratios with K/V precision. "
            "Stage 2 measures composition order. Stage 3 sweeps the best Stage 2 "
            "family across model-weight precisions and retained-entry ratios."
        ),
    }
    _write_json(pipeline_dir / "pipeline_manifest.json", manifest)

    base_config = RunConfig(
        model=args.model,
        device=args.device,
        author_ids=tuple(args.author_ids),
        biographies_path=args.biographies_path,
        passages_path=args.passages_path,
        corpus_format=args.corpus_format,
        dataset_name="tofu",
        seed=args.seed,
        group_size=args.group_size,
        kv_quantizer=args.kv_quantizer,
        max_new_tokens=args.max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        compute_nll=True,
        save_detailed_compaction_stats=args.save_detailed_compaction_stats,
        aware_refinement_iterations=args.aware_refinement_iterations,
    )
    total_stages = 3
    stage_timings: dict[str, float] = {}
    active_stage = "stage1"
    active_stage_start = time.perf_counter()

    try:
        print(
            f"Stage 1/{total_stages}: AM cardinality/precision surface on "
            + ", ".join(args.datasets),
            flush=True,
        )
        stage1_runs: list[Path] = []
        for dataset in args.datasets:
            stage1_config = _stage_dataset_config(
                base_config,
                args,
                dataset=dataset,
                weight_precision="bf16",
                conditions=stage1_conditions,
                tofu_questions=args.screen_questions,
                ruler_questions=args.screen_ruler_questions,
                quality_articles=args.quality_articles,
                tofu_generation_questions=args.screen_generation_questions,
                ruler_generation_questions=args.screen_ruler_generation_questions,
                queries_per_head=args.screen_queries_per_head,
                generate=args.screen_generate,
                output_root=pipeline_dir / "stage1",
            )
            stage1_runs.append(_run_or_resume(stage1_config, args.resume))
            aggregate_pipeline(pipeline_dir)
        stage1_rows = [
            row
            for stage1_run in stage1_runs
            for row in _read_jsonl(stage1_run / "summary.jsonl")
        ]
        stage1_diagnostics = select_stage1(stage1_rows)
        stage2_precision_selection = stage1_diagnostics[
            "stage2_precision_selection"
        ]
        stage1_diagnostics["source_runs"] = [
            str(path.relative_to(pipeline_dir)) for path in stage1_runs
        ]
        _write_json(pipeline_dir / "stage1" / "diagnostics.json", stage1_diagnostics)
        aggregate_pipeline(pipeline_dir)
        stage_timings["stage1_seconds"] = time.perf_counter() - active_stage_start
        print(
            f"Stage 1 complete in {stage_timings['stage1_seconds']:.1f}s",
            flush=True,
        )

        active_stage = "stage2"
        active_stage_start = time.perf_counter()
        stage2_precision_pairs = tuple(
            tuple(pair)
            for pair in stage2_precision_selection["selected_precision_pairs"]
        )
        stage2_conditions = build_stage2_suite(stage2_precision_pairs)
        print(
            f"Stage 2/{total_stages}: composition-order surface on selected precisions",
            flush=True,
        )
        stage2_runs: list[Path] = []
        for dataset in args.datasets:
            stage2_config = _stage_dataset_config(
                base_config,
                args,
                dataset=dataset,
                weight_precision="bf16",
                conditions=stage2_conditions,
                tofu_questions=args.confirm_questions,
                ruler_questions=args.confirm_ruler_questions,
                quality_articles=args.quality_articles,
                tofu_generation_questions=args.confirm_generation_questions,
                ruler_generation_questions=args.confirm_ruler_generation_questions,
                queries_per_head=args.confirm_queries_per_head,
                generate=args.confirm_generate,
                output_root=pipeline_dir / "stage2",
            )
            stage2_runs.append(_run_or_resume(stage2_config, args.resume))
            aggregate_pipeline(pipeline_dir)
        stage2_rows = [
            row
            for stage2_run in stage2_runs
            for row in _read_jsonl(stage2_run / "summary.jsonl")
        ]
        stage2_diagnostics = select_stage2(stage2_rows)
        stage2_diagnostics["source_runs"] = [
            str(path.relative_to(pipeline_dir)) for path in stage2_runs
        ]
        _write_json(pipeline_dir / "stage2" / "diagnostics.json", stage2_diagnostics)
        aggregate_pipeline(pipeline_dir)
        stage_timings["stage2_seconds"] = time.perf_counter() - active_stage_start
        print(
            f"Stage 2 complete in {stage_timings['stage2_seconds']:.1f}s",
            flush=True,
        )

        active_stage = "stage3"
        active_stage_start = time.perf_counter()
        stage3_conditions, stage3_funnel = build_stage3_funnel(
            stage1_rows,
            stage2_rows,
            stage2_precision_pairs,
        )
        stage3_funnel["source_runs"] = [
            *[str(path.relative_to(pipeline_dir)) for path in stage1_runs],
            *[str(path.relative_to(pipeline_dir)) for path in stage2_runs],
        ]
        _write_json(pipeline_dir / "stage3" / "funnel.json", stage3_funnel)
        print(
            f"Stage 3/{total_stages}: retained-KV sweep across model weights",
            flush=True,
        )
        _write_json(
            pipeline_dir / "resolved_plan.json",
            _resolved_plan(
                stage1_conditions,
                stage2_conditions,
                stage3_conditions,
                args.stage3_weight_precisions,
                args.datasets,
            ),
        )
        stage3_runs = []
        for weight_precision in args.stage3_weight_precisions:
            for dataset in args.datasets:
                stage3_config = _stage_dataset_config(
                    base_config,
                    args,
                    dataset=dataset,
                    weight_precision=weight_precision,
                    conditions=stage3_conditions,
                    tofu_questions=args.final_questions,
                    ruler_questions=args.final_ruler_questions,
                    quality_articles=args.quality_articles,
                    tofu_generation_questions=args.final_generation_questions,
                    ruler_generation_questions=args.final_ruler_generation_questions,
                    queries_per_head=args.final_queries_per_head,
                    generate=args.final_generate,
                    output_root=pipeline_dir / "stage3",
                )
                stage3_runs.append(_run_or_resume(stage3_config, args.resume))
                aggregate_pipeline(pipeline_dir)

        stage_timings["stage3_seconds"] = time.perf_counter() - active_stage_start
        print(
            f"Stage 3 complete in {stage_timings['stage3_seconds']:.1f}s",
            flush=True,
        )

        aggregate = aggregate_pipeline(pipeline_dir)
        wall_seconds = time.perf_counter() - pipeline_start
        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "stage1_diagnostics": stage1_diagnostics,
                "stage1_runs": [
                    str(path.relative_to(pipeline_dir)) for path in stage1_runs
                ],
                "stage2_diagnostics": stage2_diagnostics,
                "stage2_runs": [
                    str(path.relative_to(pipeline_dir)) for path in stage2_runs
                ],
                "stage3_funnel": stage3_funnel,
                "stage3_runs": [
                    str(path.relative_to(pipeline_dir)) for path in stage3_runs
                ],
                "aggregate": aggregate,
                "timing": {
                    **stage_timings,
                    "wall_seconds": wall_seconds,
                },
            }
        )
    except Exception as error:
        stage_timings[f"{active_stage}_seconds_incomplete"] = (
            time.perf_counter() - active_stage_start
        )
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "aggregate": aggregate_pipeline(pipeline_dir),
                "timing": {
                    **stage_timings,
                    "wall_seconds": time.perf_counter() - pipeline_start,
                },
            }
        )
        _write_json(pipeline_dir / "pipeline_manifest.json", manifest)
        raise

    _write_json(pipeline_dir / "pipeline_manifest.json", manifest)
    print(
        f"Pipeline complete in {manifest['timing']['wall_seconds']:.1f}s: "
        f"{pipeline_dir}",
        flush=True,
    )
    return pipeline_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the stratified-funnel compaction-limit research pipeline."
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--pipeline-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--datasets",
        choices=("tofu", "ruler", "quality"),
        nargs="+",
        default=("tofu", "quality"),
    )
    parser.add_argument("--author-ids", type=int, nargs="+", default=tuple(range(10)))
    parser.add_argument(
        "--biographies-path", type=Path, default=DEFAULT_BIOGRAPHIES_PATH
    )
    parser.add_argument("--passages-path", type=Path, default=DEFAULT_PASSAGES_PATH)
    parser.add_argument(
        "--corpus-format", choices=("biography", "passages"), default="biography"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruler-path", type=Path, default=DEFAULT_RULER_PATH)
    parser.add_argument("--quality-path", type=Path, default=DEFAULT_QUALITY_PATH)
    parser.add_argument("--quality-articles", type=int, default=20)
    parser.add_argument(
        "--ruler-task-samples",
        nargs="+",
        default=("fwe=150", "qa_1=75", "qa_2=75"),
    )
    parser.add_argument("--screen-questions", type=int, default=200)
    parser.add_argument("--confirm-questions", type=int, default=200)
    parser.add_argument("--final-questions", type=int, default=200)
    parser.add_argument("--screen-ruler-questions", type=int, default=300)
    parser.add_argument("--confirm-ruler-questions", type=int, default=300)
    parser.add_argument("--final-ruler-questions", type=int, default=300)
    parser.add_argument("--screen-generation-questions", type=int, default=200)
    parser.add_argument("--confirm-generation-questions", type=int, default=200)
    parser.add_argument("--final-generation-questions", type=int, default=200)
    parser.add_argument("--screen-ruler-generation-questions", type=int, default=300)
    parser.add_argument("--confirm-ruler-generation-questions", type=int, default=300)
    parser.add_argument("--final-ruler-generation-questions", type=int, default=300)
    parser.add_argument("--screen-queries-per-head", type=int, default=5000)
    parser.add_argument("--confirm-queries-per-head", type=int, default=5000)
    parser.add_argument("--final-queries-per-head", type=int, default=5000)
    parser.add_argument(
        "--screen-generate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--confirm-generate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--final-generate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--ruler-generation-batch-size", type=int, default=4)
    parser.add_argument("--quality-generation-batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--kv-quantizer",
        choices=("kivi_layout_symmetric", "per_vector_symmetric"),
        default="kivi_layout_symmetric",
    )
    parser.add_argument("--aware-refinement-iterations", type=int, default=2)
    parser.add_argument(
        "--save-detailed-compaction-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--stage3-weight-precisions",
        choices=("bf16", "int8", "nf4"),
        nargs="+",
        default=("bf16", "int8", "nf4"),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "screen_questions",
        "confirm_questions",
        "final_questions",
        "screen_ruler_questions",
        "confirm_ruler_questions",
        "final_ruler_questions",
        "screen_generation_questions",
        "confirm_generation_questions",
        "final_generation_questions",
        "screen_ruler_generation_questions",
        "confirm_ruler_generation_questions",
        "final_ruler_generation_questions",
        "screen_queries_per_head",
        "confirm_queries_per_head",
        "final_queries_per_head",
        "max_new_tokens",
        "generation_batch_size",
        "ruler_generation_batch_size",
        "quality_articles",
        "quality_generation_batch_size",
        "group_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.aware_refinement_iterations < 0:
        raise ValueError("aware_refinement_iterations must be non-negative.")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.plan_only:
        print(json.dumps(_runtime_pipeline_plan(args), indent=2))
        return
    if args.pipeline_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.pipeline_dir = Path("outputs") / "pipelines" / timestamp
    run_pipeline(args)


if __name__ == "__main__":
    main()
