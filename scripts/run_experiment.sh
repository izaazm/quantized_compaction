#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Edit variables below or override them from the command line, for example:
#   MODEL_NAME=Qwen/Qwen3-4B AUTHOR_IDS="0 1" MAX_QUESTIONS=40 \
#     ./scripts/run_experiment.sh
#
# The default full sweep evaluates BF16, INT8 and NF4 model weights. Use
# WEIGHT_PRECISIONS=bf16 for a smaller local pilot. Set PLAN_ONLY=true to print
# the compression grid without loading a model.

# Model and runtime.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
WEIGHT_PRECISIONS="${WEIGHT_PRECISIONS:-${WEIGHT_PRECISION:-bf16 int8 nf4}}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

# Local TOFU corpus. Values containing several items are space-separated.
AUTHOR_IDS="${AUTHOR_IDS:-0 1 2 3 4 5 6 7 8 9}"
BIOGRAPHIES_PATH="${BIOGRAPHIES_PATH:-$ROOT_DIR/data/tofu_author_biographies.json}"
PASSAGES_PATH="${PASSAGES_PATH:-$ROOT_DIR/data/tofu_qa_passages.json}"
CORPUS_FORMAT="${CORPUS_FORMAT:-biography}" # biography or passages
MAX_QUESTIONS="${MAX_QUESTIONS:-200}"
SEED="${SEED:-42}"

# Compression surface. The default uses the shared 10%, 5%, and 2% payload
# budgets. The four asymmetric pairs test
# which side benefits more from precision without expanding to a full 3x3 grid.
GRID_MODE="${GRID_MODE:-equal_budget}" # equal_budget or factorial
BUDGET_FRACTIONS="${BUDGET_FRACTIONS:-0.10 0.05 0.02}"
ENTRY_RATIOS="${ENTRY_RATIOS:-0.10 0.05 0.02}"
KV_PAIRS="${KV_PAIRS:-k16v16 k8v8 k4v4 k8v4 k4v8 k16v4 k4v16}"
METHODS="${METHODS:-am}"
COMPOSITION_MODES="${COMPOSITION_MODES:-post}"
INCLUDE_DENSE_PRECISION_CONTROLS="${INCLUDE_DENSE_PRECISION_CONTROLS:-true}"

# KV quantize/dequantize simulation. Attention consumes BF16 tensors restricted
# to the requested INT8/INT4 grid; logical packed and resident QDQ bytes are
# reported separately.
KV_QUANTIZER="${KV_QUANTIZER:-kivi_layout_symmetric}"
GROUP_SIZE="${GROUP_SIZE:-64}"

# Attention Matching query extraction. This long-context experiment uses
# context-prefill queries from the corpus and does not use TOFU QA for fitting.
MAX_QUERIES_PER_HEAD="${MAX_QUERIES_PER_HEAD:-5000}"
AWARE_REFINEMENT_ITERATIONS="${AWARE_REFINEMENT_ITERATIONS:-2}"
SAVE_DETAILED_COMPACTION_STATS="${SAVE_DETAILED_COMPACTION_STATS:-false}"

# Evaluation. Set either flag to false to skip that part of evaluation.
COMPUTE_NLL="${COMPUTE_NLL:-true}"
GENERATE="${GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"

# Each invocation gets one sweep directory containing settings, environment,
# logs, per-weight runs, and combined summaries.
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
SWEEP_NAME="${SWEEP_NAME:-$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_NAME//\//-}-$$}"
PLAN_ONLY="${PLAN_ONLY:-false}"

if [[ ! -x "$PYTHON_BIN" ]]; then
	printf 'Python environment not found at %s\n' "$PYTHON_BIN" >&2
	printf 'Run ./setup_env.sh or set PYTHON_BIN explicitly.\n' >&2
	exit 1
fi

if [[ "$COMPUTE_NLL" != "true" && "$GENERATE" != "true" ]]; then
	printf 'At least one of COMPUTE_NLL or GENERATE must be true.\n' >&2
	exit 1
fi

read -r -a AUTHOR_ID_ARGS <<< "$AUTHOR_IDS"
read -r -a BUDGET_FRACTION_ARGS <<< "$BUDGET_FRACTIONS"
read -r -a ENTRY_RATIO_ARGS <<< "$ENTRY_RATIOS"
read -r -a KV_PAIR_ARGS <<< "$KV_PAIRS"
read -r -a METHOD_ARGS <<< "$METHODS"
read -r -a COMPOSITION_MODE_ARGS <<< "$COMPOSITION_MODES"
read -r -a WEIGHT_PRECISION_ARGS <<< "$WEIGHT_PRECISIONS"

GRID_ARGS=(
	"--grid-mode" "$GRID_MODE"
	"--budget-fractions" "${BUDGET_FRACTION_ARGS[@]}"
	"--entry-ratios" "${ENTRY_RATIO_ARGS[@]}"
	"--kv-pairs" "${KV_PAIR_ARGS[@]}"
	"--methods" "${METHOD_ARGS[@]}"
	"--composition-modes" "${COMPOSITION_MODE_ARGS[@]}"
)

if [[ "$INCLUDE_DENSE_PRECISION_CONTROLS" != "true" ]]; then
	GRID_ARGS+=("--no-dense-precision-controls")
fi

cd "$ROOT_DIR"

if [[ "$PLAN_ONLY" == "true" ]]; then
	"$PYTHON_BIN" -m compaction_limit.cli plan "${GRID_ARGS[@]}"
	exit 0
fi

SWEEP_DIR="$OUTPUT_ROOT/sweeps/$SWEEP_NAME"
mkdir -p "$SWEEP_DIR/logs" "$SWEEP_DIR/runs"

{
	printf 'MODEL_NAME=%q\n' "$MODEL_NAME"
	printf 'WEIGHT_PRECISIONS=%q\n' "$WEIGHT_PRECISIONS"
	printf 'DEVICE=%q\n' "$DEVICE"
	printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
	printf 'AUTHOR_IDS=%q\n' "$AUTHOR_IDS"
	printf 'BIOGRAPHIES_PATH=%q\n' "$BIOGRAPHIES_PATH"
	printf 'PASSAGES_PATH=%q\n' "$PASSAGES_PATH"
	printf 'CORPUS_FORMAT=%q\n' "$CORPUS_FORMAT"
	printf 'MAX_QUESTIONS=%q\n' "$MAX_QUESTIONS"
	printf 'SEED=%q\n' "$SEED"
	printf 'GRID_MODE=%q\n' "$GRID_MODE"
	printf 'BUDGET_FRACTIONS=%q\n' "$BUDGET_FRACTIONS"
	printf 'ENTRY_RATIOS=%q\n' "$ENTRY_RATIOS"
	printf 'KV_PAIRS=%q\n' "$KV_PAIRS"
	printf 'METHODS=%q\n' "$METHODS"
	printf 'COMPOSITION_MODES=%q\n' "$COMPOSITION_MODES"
	printf 'INCLUDE_DENSE_PRECISION_CONTROLS=%q\n' "$INCLUDE_DENSE_PRECISION_CONTROLS"
	printf 'KV_QUANTIZER=%q\n' "$KV_QUANTIZER"
	printf 'GROUP_SIZE=%q\n' "$GROUP_SIZE"
	printf 'MAX_QUERIES_PER_HEAD=%q\n' "$MAX_QUERIES_PER_HEAD"
	printf 'AWARE_REFINEMENT_ITERATIONS=%q\n' "$AWARE_REFINEMENT_ITERATIONS"
	printf 'SAVE_DETAILED_COMPACTION_STATS=%q\n' "$SAVE_DETAILED_COMPACTION_STATS"
	printf 'COMPUTE_NLL=%q\n' "$COMPUTE_NLL"
	printf 'GENERATE=%q\n' "$GENERATE"
	printf 'MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
	printf 'GENERATION_BATCH_SIZE=%q\n' "$GENERATION_BATCH_SIZE"
	printf 'OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
	printf 'SWEEP_NAME=%q\n' "$SWEEP_NAME"
} > "$SWEEP_DIR/settings.env"

"$PYTHON_BIN" -m compaction_limit.cli plan "${GRID_ARGS[@]}" > "$SWEEP_DIR/plan.json"

{
	date -u
	"$PYTHON_BIN" -VV
	uname -a
	"$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_version", torch.version.cuda)'
	"$PYTHON_BIN" -c 'import importlib.metadata as m; print("installed_packages"); print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"])))'
	if command -v nvidia-smi >/dev/null 2>&1; then
		nvidia-smi
	else
		printf 'nvidia-smi unavailable\n'
	fi
} > "$SWEEP_DIR/environment.log" 2>&1

printf 'running\n' > "$SWEEP_DIR/status.txt"

finish_sweep() {
	local exit_code=$?
	local final_status="failed"
	trap - EXIT
	if [[ "$exit_code" -eq 0 ]]; then
		final_status="complete"
	fi
	printf '%s\n' "$final_status" > "$SWEEP_DIR/status.txt"
	"$PYTHON_BIN" -m compaction_limit.aggregate \
		--sweep-dir "$SWEEP_DIR" --status "$final_status" || true
	printf 'Sweep %s. Artifacts: %s\n' "$final_status" "$SWEEP_DIR"
	exit "$exit_code"
}
trap finish_sweep EXIT

exec > >(tee -a "$SWEEP_DIR/logs/sweep.log") 2>&1

COMMON_ARGS=(
	"${GRID_ARGS[@]}"
	"--model" "$MODEL_NAME"
	"--device" "$DEVICE"
	"--author-ids" "${AUTHOR_ID_ARGS[@]}"
	"--biographies-path" "$BIOGRAPHIES_PATH"
	"--passages-path" "$PASSAGES_PATH"
	"--corpus-format" "$CORPUS_FORMAT"
	"--max-questions" "$MAX_QUESTIONS"
	"--seed" "$SEED"
	"--group-size" "$GROUP_SIZE"
	"--kv-quantizer" "$KV_QUANTIZER"
	"--max-queries-per-head" "$MAX_QUERIES_PER_HEAD"
	"--aware-refinement-iterations" "$AWARE_REFINEMENT_ITERATIONS"
	"--max-new-tokens" "$MAX_NEW_TOKENS"
	"--generation-batch-size" "$GENERATION_BATCH_SIZE"
	"--output-root" "$SWEEP_DIR/runs"
)

if [[ "$COMPUTE_NLL" != "true" ]]; then
	COMMON_ARGS+=("--skip-nll")
fi

if [[ "$GENERATE" != "true" ]]; then
	COMMON_ARGS+=("--skip-generation")
fi

if [[ "$SAVE_DETAILED_COMPACTION_STATS" != "true" ]]; then
	COMMON_ARGS+=("--no-save-detailed-compaction-stats")
fi

for weight_precision in "${WEIGHT_PRECISION_ARGS[@]}"; do
	printf 'Starting %s weights for %s\n' "$weight_precision" "$MODEL_NAME"
	"$PYTHON_BIN" -m compaction_limit.cli run \
		"${COMMON_ARGS[@]}" --weight-precision "$weight_precision"
done
