from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


RULER_REPOSITORY = "simonjegou/ruler"
DEFAULT_RULER_TASK_SAMPLES = {"fwe": 150, "qa_1": 75, "qa_2": 75}
HOTPOTQA_DEV_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
QUALITY_DEV_URL = (
    "https://raw.githubusercontent.com/nyu-mll/quality/"
    "05e85750d4c5444d2a0a4ad299f6df5f4df06068/data/v1.0.1/"
    "QuALITY.v1.0.1.htmlstripped.dev"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _balanced_sample(
    rows: Sequence[dict[str, Any]],
    count: int,
    seed: int,
    strata: Sequence[str],
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Sample counts must be positive.")
    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in strata)].append(row)
    for values in groups.values():
        rng.shuffle(values)

    sampled: list[dict[str, Any]] = []
    ordered_keys = sorted(groups, key=lambda key: tuple(str(value) for value in key))
    while len(sampled) < min(count, len(rows)):
        made_progress = False
        for key in ordered_keys:
            if groups[key]:
                sampled.append(groups[key].pop())
                made_progress = True
                if len(sampled) == min(count, len(rows)):
                    break
        if not made_progress:
            break
    return sampled


def prepare_ruler(
    output_path: Path,
    context_length: int,
    task_samples: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "RULER preparation requires the benchmark extra: "
            '`uv pip install -e ".[benchmarks]"`.'
        ) from error

    dataset = load_dataset(RULER_REPOSITORY, str(context_length), split="test")
    raw_rows = [
        {**dict(row), "_source_index": source_index}
        for source_index, row in enumerate(dataset)
    ]
    selected_tasks = tuple(dict.fromkeys(str(task) for task in task_samples))
    if not selected_tasks:
        raise ValueError("At least one RULER task must be selected.")
    if any(int(count) <= 0 for count in task_samples.values()):
        raise ValueError("RULER task sample counts must be positive.")
    available_tasks = {str(row.get("task")) for row in raw_rows}
    missing_tasks = set(selected_tasks) - available_tasks
    if missing_tasks:
        raise ValueError(
            f"RULER tasks not found in {RULER_REPOSITORY}/{context_length}: "
            f"{sorted(missing_tasks)}"
        )

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    available_per_task: dict[str, int] = {}
    selected_per_task: dict[str, int] = {}
    for task in selected_tasks:
        requested = int(task_samples[task])
        task_rows = [row for row in raw_rows if str(row.get("task")) == task]
        available_per_task[task] = len(task_rows)
        if len(task_rows) < requested:
            raise ValueError(
                f"RULER task {task!r} has {len(task_rows)} rows, fewer than the "
                f"requested {requested}."
            )
        rng.shuffle(task_rows)
        chosen = task_rows[:requested]
        sampled.extend(chosen)
        selected_per_task[task] = len(chosen)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for selected_index, row in enumerate(sampled):
            answer = row.get("answer", [])
            if isinstance(answer, str):
                answer = [answer]
            answer = [str(value) for value in answer if str(value)]
            if not answer:
                raise ValueError(
                    f"RULER row {row['_source_index']} has no reference answers."
                )
            task = str(row.get("task", "unknown"))
            payload = {
                "id": (
                    f"ruler_{context_length}_{task}_"
                    f"{row['_source_index']}"
                ),
                "context": row["context"],
                "question": row["question"],
                "answer": list(answer),
                "task": task,
                "answer_prefix": row.get("answer_prefix", ""),
                "teacher_forced_answer": (
                    ", ".join(answer) if task == "fwe" else answer[0]
                ),
                "max_new_tokens": int(row.get("max_new_tokens", 64)),
                "context_length": context_length,
                "source_index": int(row["_source_index"]),
                "selected_index": selected_index,
            }
            handle.write(json.dumps(payload) + "\n")
    return {
        "path": str(output_path),
        "repository": RULER_REPOSITORY,
        "configuration": str(context_length),
        "available_rows": len(raw_rows),
        "selected_rows": len(sampled),
        "tasks": list(selected_tasks),
        "requested_per_task": {
            task: int(task_samples[task]) for task in selected_tasks
        },
        "available_per_task": available_per_task,
        "selected_per_task": selected_per_task,
        "is_full_official_task_evaluation": all(
            selected_per_task[task] == available_per_task[task]
            for task in selected_tasks
        ),
    }


def prepare_hotpotqa(output_path: Path, count: int, seed: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".full.json")
    if not raw_path.exists():
        urllib.request.urlretrieve(HOTPOTQA_DEV_URL, raw_path)
    with raw_path.open() as handle:
        raw_rows = json.load(handle)
    if not isinstance(raw_rows, list):
        raise ValueError("The downloaded HotPotQA artifact is not a JSON list.")
    sampled = _balanced_sample(raw_rows, count, seed, strata=("type", "level"))
    _write_json(output_path, sampled)
    return {
        "path": str(output_path),
        "source_url": HOTPOTQA_DEV_URL,
        "available_rows": len(raw_rows),
        "selected_rows": len(sampled),
        "types": sorted({str(row.get("type", "unknown")) for row in sampled}),
        "levels": sorted({str(row.get("level", "unknown")) for row in sampled}),
    }


def prepare_quality(output_path: Path) -> dict[str, Any]:
    """Download the official QuALITY v1.0.1 HTML-stripped development set."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        urllib.request.urlretrieve(QUALITY_DEV_URL, output_path)
    rows = []
    with output_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a QuALITY object at {output_path}:{line_number}."
                )
            rows.append(row)
    return {
        "path": str(output_path),
        "source_url": QUALITY_DEV_URL,
        "rows": len(rows),
        "unique_articles": len({str(row["article_id"]) for row in rows}),
        "questions": sum(len(row.get("questions", [])) for row in rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RULER-16K, QuALITY, and HotPotQA evaluation data."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruler-context-length", type=int, default=16_384)
    parser.add_argument(
        "--ruler-task-samples",
        nargs="+",
        default=tuple(
            f"{task}={count}" for task, count in DEFAULT_RULER_TASK_SAMPLES.items()
        ),
        metavar="TASK=COUNT",
    )
    parser.add_argument("--hotpotqa-questions", type=int, default=100)
    parser.add_argument(
        "--skip-ruler", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip-hotpotqa", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip-quality", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ruler_task_samples: dict[str, int] = {}
    for value in args.ruler_task_samples:
        try:
            task, count_text = value.split("=", maxsplit=1)
            count = int(count_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid --ruler-task-samples value {value!r}; use TASK=COUNT."
            ) from error
        if not task or count <= 0:
            raise ValueError(
                f"Invalid --ruler-task-samples value {value!r}; use TASK=COUNT."
            )
        ruler_task_samples[task] = count
    output_dir = args.output_dir
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
    }
    if not args.skip_ruler:
        report["ruler"] = prepare_ruler(
            output_dir / "ruler_16k.jsonl",
            context_length=args.ruler_context_length,
            task_samples=ruler_task_samples,
            seed=args.seed,
        )
    if not args.skip_hotpotqa:
        report["hotpotqa"] = prepare_hotpotqa(
            output_dir / "hotpotqa_dev_distractor_subset.json",
            count=args.hotpotqa_questions,
            seed=args.seed,
        )
    if not args.skip_quality:
        report["quality"] = prepare_quality(output_dir / "quality_dev.jsonl")
    _write_json(output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
