from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed_lines = 0
    try:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if isinstance(value, dict):
                    rows.append(value)
                else:
                    malformed_lines += 1
    except OSError:
        pass
    return rows, malformed_lines


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=True) + "\n")


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def aggregate_sweep(sweep_dir: Path, status: str) -> dict[str, Any]:
    sweep_dir = Path(sweep_dir)
    runs_dir = sweep_dir / "runs"
    combined_rows: list[dict[str, Any]] = []
    run_index = []
    malformed_lines = 0

    run_dirs = sorted(path for path in runs_dir.glob("*") if path.is_dir())
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        rows, malformed = _read_jsonl(run_dir / "summary.jsonl")
        malformed_lines += malformed
        relative_run_dir = str(run_dir.relative_to(sweep_dir))
        for row in rows:
            combined_rows.append({"run_dir": relative_run_dir, **row})
        run_index.append(
            {
                "run_dir": relative_run_dir,
                "status": manifest.get("status", "missing"),
                "model": manifest.get("config", {}).get("model"),
                "weight_precision": manifest.get("config", {}).get("weight_precision"),
                "conditions_planned": len(manifest.get("conditions", [])),
                "conditions_completed": len(rows),
            }
        )

    _write_jsonl(sweep_dir / "combined_summary.jsonl", combined_rows)
    if combined_rows:
        with (sweep_dir / "combined_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=_fieldnames(combined_rows),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(combined_rows)
    else:
        (sweep_dir / "combined_summary.csv").write_text("")

    manifest = {
        "status": status,
        "aggregated_at": datetime.now(timezone.utc).isoformat(),
        "num_runs": len(run_index),
        "num_summary_rows": len(combined_rows),
        "malformed_summary_lines_skipped": malformed_lines,
        "runs": run_index,
        "artifact_index": {
            "combined_summary_jsonl": "combined_summary.jsonl",
            "combined_summary_csv": "combined_summary.csv",
            "settings": "settings.env",
            "plan": "plan.json",
            "environment": "environment.log",
            "sweep_log": "logs/sweep.log",
            "per_run_artifacts": "runs/",
        },
    }
    _write_json(sweep_dir / "sweep_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate a compaction-limit sweep.")
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--status", choices=("complete", "failed", "running"), required=True
    )
    args = parser.parse_args(argv)
    manifest = aggregate_sweep(args.sweep_dir, args.status)
    print(
        f"Saved {manifest['num_summary_rows']} summary rows from "
        f"{manifest['num_runs']} run(s) to {args.sweep_dir}"
    )


if __name__ == "__main__":
    main()
