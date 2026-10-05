#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

AUTHOR_IDS="${AUTHOR_IDS:-0 1 2 3 4 5 6 7 8 9}"
DATASETS="${DATASETS:-tofu ruler quality}"
BIOGRAPHIES_PATH="${BIOGRAPHIES_PATH:-$ROOT_DIR/data/tofu_author_biographies.json}"
PASSAGES_PATH="${PASSAGES_PATH:-$ROOT_DIR/data/tofu_qa_passages.json}"
RULER_PATH="${RULER_PATH:-$ROOT_DIR/data/benchmarks/ruler_16k.jsonl}"
RULER_TASK_SAMPLES="${RULER_TASK_SAMPLES:-fwe=150 qa_1=75 qa_2=75}"
QUALITY_PATH="${QUALITY_PATH:-$ROOT_DIR/data/benchmarks/quality_dev.jsonl}"
QUALITY_ARTICLES="${QUALITY_ARTICLES:-50}"
PREPARE_BENCHMARKS="${PREPARE_BENCHMARKS:-true}"
CORPUS_FORMAT="${CORPUS_FORMAT:-biography}"
SEED="${SEED:-42}"

SCREEN_QUESTIONS="${SCREEN_QUESTIONS:-200}"
CONFIRM_QUESTIONS="${CONFIRM_QUESTIONS:-200}"
FINAL_QUESTIONS="${FINAL_QUESTIONS:-200}"
SCREEN_RULER_QUESTIONS="${SCREEN_RULER_QUESTIONS:-300}"
CONFIRM_RULER_QUESTIONS="${CONFIRM_RULER_QUESTIONS:-300}"
FINAL_RULER_QUESTIONS="${FINAL_RULER_QUESTIONS:-300}"
SCREEN_GENERATION_QUESTIONS="${SCREEN_GENERATION_QUESTIONS:-200}"
CONFIRM_GENERATION_QUESTIONS="${CONFIRM_GENERATION_QUESTIONS:-200}"
FINAL_GENERATION_QUESTIONS="${FINAL_GENERATION_QUESTIONS:-200}"
SCREEN_RULER_GENERATION_QUESTIONS="${SCREEN_RULER_GENERATION_QUESTIONS:-300}"
CONFIRM_RULER_GENERATION_QUESTIONS="${CONFIRM_RULER_GENERATION_QUESTIONS:-300}"
FINAL_RULER_GENERATION_QUESTIONS="${FINAL_RULER_GENERATION_QUESTIONS:-300}"
SCREEN_QUERIES_PER_HEAD="${SCREEN_QUERIES_PER_HEAD:-5000}"
CONFIRM_QUERIES_PER_HEAD="${CONFIRM_QUERIES_PER_HEAD:-5000}"
FINAL_QUERIES_PER_HEAD="${FINAL_QUERIES_PER_HEAD:-5000}"
SCREEN_GENERATE="${SCREEN_GENERATE:-true}"
CONFIRM_GENERATE="${CONFIRM_GENERATE:-true}"
FINAL_GENERATE="${FINAL_GENERATE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-32}"
RULER_GENERATION_BATCH_SIZE="${RULER_GENERATION_BATCH_SIZE:-16}"
QUALITY_GENERATION_BATCH_SIZE="${QUALITY_GENERATION_BATCH_SIZE:-32}"

KV_QUANTIZER="${KV_QUANTIZER:-kivi_layout_symmetric}"
GROUP_SIZE="${GROUP_SIZE:-64}"
AWARE_REFINEMENT_ITERATIONS="${AWARE_REFINEMENT_ITERATIONS:-2}"
SAVE_DETAILED_COMPACTION_STATS="${SAVE_DETAILED_COMPACTION_STATS:-false}"
STAGE3_WEIGHT_PRECISIONS="${STAGE3_WEIGHT_PRECISIONS:-bf16 int8 nf4}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
PIPELINE_NAME="${PIPELINE_NAME:-$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_NAME//\//-}-$$}"
PIPELINE_DIR="$OUTPUT_ROOT/pipelines/$PIPELINE_NAME"
RESUME="${RESUME:-true}"
PLAN_ONLY="${PLAN_ONLY:-false}"

if [[ ! -x "$PYTHON_BIN" ]]; then
	printf 'Python environment not found at %s\n' "$PYTHON_BIN" >&2
	printf 'Run ./setup_env.sh or set PYTHON_BIN explicitly.\n' >&2
	exit 1
fi

read -r -a AUTHOR_ID_ARGS <<< "$AUTHOR_IDS"
read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a RULER_TASK_SAMPLE_ARGS <<< "$RULER_TASK_SAMPLES"
read -r -a WEIGHT_PRECISION_ARGS <<< "$STAGE3_WEIGHT_PRECISIONS"

cd "$ROOT_DIR"

ARGS=(
	"--pipeline-dir" "$PIPELINE_DIR"
	"--model" "$MODEL_NAME"
	"--device" "$DEVICE"
	"--datasets" "${DATASET_ARGS[@]}"
	"--author-ids" "${AUTHOR_ID_ARGS[@]}"
	"--biographies-path" "$BIOGRAPHIES_PATH"
	"--passages-path" "$PASSAGES_PATH"
	"--ruler-path" "$RULER_PATH"
	"--ruler-task-samples" "${RULER_TASK_SAMPLE_ARGS[@]}"
	"--quality-path" "$QUALITY_PATH"
	"--quality-articles" "$QUALITY_ARTICLES"
	"--corpus-format" "$CORPUS_FORMAT"
	"--seed" "$SEED"
	"--screen-questions" "$SCREEN_QUESTIONS"
	"--confirm-questions" "$CONFIRM_QUESTIONS"
	"--final-questions" "$FINAL_QUESTIONS"
	"--screen-ruler-questions" "$SCREEN_RULER_QUESTIONS"
	"--confirm-ruler-questions" "$CONFIRM_RULER_QUESTIONS"
	"--final-ruler-questions" "$FINAL_RULER_QUESTIONS"
	"--screen-generation-questions" "$SCREEN_GENERATION_QUESTIONS"
	"--confirm-generation-questions" "$CONFIRM_GENERATION_QUESTIONS"
	"--final-generation-questions" "$FINAL_GENERATION_QUESTIONS"
	"--screen-ruler-generation-questions" "$SCREEN_RULER_GENERATION_QUESTIONS"
	"--confirm-ruler-generation-questions" "$CONFIRM_RULER_GENERATION_QUESTIONS"
	"--final-ruler-generation-questions" "$FINAL_RULER_GENERATION_QUESTIONS"
	"--screen-queries-per-head" "$SCREEN_QUERIES_PER_HEAD"
	"--confirm-queries-per-head" "$CONFIRM_QUERIES_PER_HEAD"
	"--final-queries-per-head" "$FINAL_QUERIES_PER_HEAD"
	"--max-new-tokens" "$MAX_NEW_TOKENS"
	"--generation-batch-size" "$GENERATION_BATCH_SIZE"
	"--ruler-generation-batch-size" "$RULER_GENERATION_BATCH_SIZE"
	"--quality-generation-batch-size" "$QUALITY_GENERATION_BATCH_SIZE"
	"--kv-quantizer" "$KV_QUANTIZER"
	"--group-size" "$GROUP_SIZE"
	"--aware-refinement-iterations" "$AWARE_REFINEMENT_ITERATIONS"
	"--stage3-weight-precisions" "${WEIGHT_PRECISION_ARGS[@]}"
)

if [[ "$RESUME" != "true" ]]; then ARGS+=("--no-resume"); fi
if [[ "$SCREEN_GENERATE" != "true" ]]; then ARGS+=("--no-screen-generate"); fi
if [[ "$CONFIRM_GENERATE" != "true" ]]; then ARGS+=("--no-confirm-generate"); fi
if [[ "$FINAL_GENERATE" != "true" ]]; then ARGS+=("--no-final-generate"); fi
if [[ "$SAVE_DETAILED_COMPACTION_STATS" != "true" ]]; then
	ARGS+=("--no-save-detailed-compaction-stats")
fi

if [[ "$PLAN_ONLY" == "true" ]]; then
	"$PYTHON_BIN" -m quantized_compaction.pipeline --plan-only "${ARGS[@]}"
	exit 0
fi

mkdir -p "$PIPELINE_DIR/logs"

{
	printf 'MODEL_NAME=%q\n' "$MODEL_NAME"
	printf 'DEVICE=%q\n' "$DEVICE"
	printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
	printf 'DATASETS=%q\n' "$DATASETS"
	printf 'AUTHOR_IDS=%q\n' "$AUTHOR_IDS"
	printf 'BIOGRAPHIES_PATH=%q\n' "$BIOGRAPHIES_PATH"
	printf 'PASSAGES_PATH=%q\n' "$PASSAGES_PATH"
	printf 'RULER_PATH=%q\n' "$RULER_PATH"
	printf 'RULER_TASK_SAMPLES=%q\n' "$RULER_TASK_SAMPLES"
	printf 'QUALITY_PATH=%q\n' "$QUALITY_PATH"
	printf 'QUALITY_ARTICLES=%q\n' "$QUALITY_ARTICLES"
	printf 'CORPUS_FORMAT=%q\n' "$CORPUS_FORMAT"
	printf 'SEED=%q\n' "$SEED"
	printf 'SCREEN_QUESTIONS=%q\n' "$SCREEN_QUESTIONS"
	printf 'CONFIRM_QUESTIONS=%q\n' "$CONFIRM_QUESTIONS"
	printf 'FINAL_QUESTIONS=%q\n' "$FINAL_QUESTIONS"
	printf 'SCREEN_RULER_QUESTIONS=%q\n' "$SCREEN_RULER_QUESTIONS"
	printf 'CONFIRM_RULER_QUESTIONS=%q\n' "$CONFIRM_RULER_QUESTIONS"
	printf 'FINAL_RULER_QUESTIONS=%q\n' "$FINAL_RULER_QUESTIONS"
	printf 'SCREEN_GENERATION_QUESTIONS=%q\n' "$SCREEN_GENERATION_QUESTIONS"
	printf 'CONFIRM_GENERATION_QUESTIONS=%q\n' "$CONFIRM_GENERATION_QUESTIONS"
	printf 'FINAL_GENERATION_QUESTIONS=%q\n' "$FINAL_GENERATION_QUESTIONS"
	printf 'SCREEN_RULER_GENERATION_QUESTIONS=%q\n' "$SCREEN_RULER_GENERATION_QUESTIONS"
	printf 'CONFIRM_RULER_GENERATION_QUESTIONS=%q\n' "$CONFIRM_RULER_GENERATION_QUESTIONS"
	printf 'FINAL_RULER_GENERATION_QUESTIONS=%q\n' "$FINAL_RULER_GENERATION_QUESTIONS"
	printf 'SCREEN_QUERIES_PER_HEAD=%q\n' "$SCREEN_QUERIES_PER_HEAD"
	printf 'CONFIRM_QUERIES_PER_HEAD=%q\n' "$CONFIRM_QUERIES_PER_HEAD"
	printf 'FINAL_QUERIES_PER_HEAD=%q\n' "$FINAL_QUERIES_PER_HEAD"
	printf 'SCREEN_GENERATE=%q\n' "$SCREEN_GENERATE"
	printf 'CONFIRM_GENERATE=%q\n' "$CONFIRM_GENERATE"
	printf 'FINAL_GENERATE=%q\n' "$FINAL_GENERATE"
	printf 'MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
	printf 'GENERATION_BATCH_SIZE=%q\n' "$GENERATION_BATCH_SIZE"
	printf 'RULER_GENERATION_BATCH_SIZE=%q\n' "$RULER_GENERATION_BATCH_SIZE"
	printf 'QUALITY_GENERATION_BATCH_SIZE=%q\n' "$QUALITY_GENERATION_BATCH_SIZE"
	printf 'KV_QUANTIZER=%q\n' "$KV_QUANTIZER"
	printf 'GROUP_SIZE=%q\n' "$GROUP_SIZE"
	printf 'AWARE_REFINEMENT_ITERATIONS=%q\n' "$AWARE_REFINEMENT_ITERATIONS"
	printf 'SAVE_DETAILED_COMPACTION_STATS=%q\n' "$SAVE_DETAILED_COMPACTION_STATS"
	printf 'STAGE3_WEIGHT_PRECISIONS=%q\n' "$STAGE3_WEIGHT_PRECISIONS"
	printf 'OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
	printf 'PIPELINE_NAME=%q\n' "$PIPELINE_NAME"
	printf 'RESUME=%q\n' "$RESUME"
} > "$PIPELINE_DIR/settings.env"

printf 'running\n' > "$PIPELINE_DIR/status.txt"

finish_pipeline() {
	local exit_code=$?
	local final_status="failed"
	trap - EXIT
	if [[ "$exit_code" -eq 0 ]]; then
		final_status="complete"
	fi
	printf '%s\n' "$final_status" > "$PIPELINE_DIR/status.txt"
	printf 'Pipeline %s. Artifacts: %s\n' "$final_status" "$PIPELINE_DIR"
	exit "$exit_code"
}
trap finish_pipeline EXIT

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
} > "$PIPELINE_DIR/environment.log" 2>&1

exec > >(tee -a "$PIPELINE_DIR/logs/pipeline.log") 2>&1

if [[ "$PREPARE_BENCHMARKS" == "true" ]]; then
	NEED_RULER=false
	NEED_QUALITY=false
	for dataset in "${DATASET_ARGS[@]}"; do
		if [[ "$dataset" == "ruler" ]]; then NEED_RULER=true; fi
		if [[ "$dataset" == "quality" ]]; then NEED_QUALITY=true; fi
	done
	if [[ "$NEED_RULER" == "true" || "$NEED_QUALITY" == "true" ]]; then
		PREP_ARGS=(
			--output-dir "$(dirname "$RULER_PATH")"
			--ruler-task-samples "${RULER_TASK_SAMPLE_ARGS[@]}"
			--seed "$SEED"
			--skip-hotpotqa
		)
		if [[ "$NEED_RULER" != "true" ]]; then PREP_ARGS+=(--skip-ruler); fi
		if [[ "$NEED_QUALITY" != "true" ]]; then PREP_ARGS+=(--skip-quality); fi
		"$PYTHON_BIN" -m quantized_compaction.prepare_benchmarks "${PREP_ARGS[@]}"
	fi
fi

"$PYTHON_BIN" -m quantized_compaction.pipeline "${ARGS[@]}"
