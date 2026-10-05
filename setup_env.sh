#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/compaction_limit_uv_cache}"
export UV_CACHE_DIR="$TASK_UV_CACHE_DIR"

# Optional controls:
#   WITH_NF4=false  Skip BitsAndBytes when only BF16 weights are needed.
#   VERIFY=false    Skip the plan-only configuration check.
#   HF_LOGIN=true   Start an interactive Hugging Face login after installation.
#   PYTHON_VERSION  Override the default Python version (3.12).
WITH_NF4="${WITH_NF4:-true}"
VERIFY="${VERIFY:-true}"
HF_LOGIN="${HF_LOGIN:-false}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

run_uv() {
	if command -v uv >/dev/null 2>&1; then
		uv "$@"
	else
		python3 -m uv "$@"
	fi
}

if ! command -v uv >/dev/null 2>&1; then
	printf 'Installing uv...\n'
	python3 -m pip install --user uv
fi

cd "$ROOT_DIR"

if [[ -x .venv/bin/python ]]; then
	printf 'Reusing existing environment at %s/.venv...\n' "$ROOT_DIR"
else
	printf 'Creating Python %s environment at %s/.venv...\n' "$PYTHON_VERSION" "$ROOT_DIR"
	run_uv venv --python "$PYTHON_VERSION" .venv
fi

if [[ "$WITH_NF4" == "true" ]]; then
	INSTALL_TARGET=".[quantized,benchmarks]"
else
	INSTALL_TARGET=".[benchmarks]"
fi

printf 'Installing %s in editable mode...\n' "$INSTALL_TARGET"
run_uv pip install --python .venv/bin/python -e "$INSTALL_TARGET"

if [[ "$VERIFY" == "true" ]]; then
	printf 'Checking the sequential experiment pipeline...\n'
	PLAN_ONLY=true PYTHON_BIN="$ROOT_DIR/.venv/bin/python" \
		./scripts/run_all_experiments.sh >/dev/null
fi

if [[ "$HF_LOGIN" == "true" ]]; then
	.venv/bin/hf auth login
fi

printf '\nEnvironment setup complete. Activate it with:\n\n'
printf '    source %s/.venv/bin/activate\n\n' "$ROOT_DIR"
printf 'Preview the experiment with:\n\n'
printf '    PLAN_ONLY=true ./scripts/run_all_experiments.sh\n\n'
