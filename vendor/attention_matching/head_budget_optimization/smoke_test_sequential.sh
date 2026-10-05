#!/bin/bash
# Smoke test: run head budget computation for sequential objective
# This script runs a minimal head budget optimization with sequential compaction
# to verify the integration works end-to-end.

set -e

echo "=========================================="
echo "HEAD BUDGET OPTIMIZATION - SEQUENTIAL SMOKE TEST"
echo "=========================================="
echo ""

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
N_ARTICLES="${N_ARTICLES:-1}"
N_EVAL_POINTS="${N_EVAL_POINTS:-3}"
TARGET_RATIO="${TARGET_RATIO:-0.05}"
SOLVE_RATIOS="${SOLVE_RATIOS:-0.02,0.05}"
METHOD="${METHOD:-AM-HighestAttnKeys-sequential}"
ALGORITHM_CONFIG="${ALGORITHM_CONFIG:-default}"
QUERY_CONFIG="${QUERY_CONFIG:-repeat}"
DATASET_NAME="${DATASET_NAME:-quality}"
DEVICE="${DEVICE:-cuda}"

# Derived values
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="logs/budget_optimization/${MODEL_SHORT_NAME}/sequential_smoke_${TIMESTAMP}"

echo "Configuration:"
echo "  Model: $MODEL_NAME ($MODEL_SHORT_NAME)"
echo "  Method: $METHOD"
echo "  Algorithm Config: $ALGORITHM_CONFIG"
echo "  Query Config: $QUERY_CONFIG"
echo "  Dataset: $DATASET_NAME"
echo "  N Articles: $N_ARTICLES"
echo "  N Eval Points: $N_EVAL_POINTS"
echo "  Target Ratio (baseline): $TARGET_RATIO"
echo "  Solve Ratios: $SOLVE_RATIOS"
echo "  Device: $DEVICE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# Run head budget optimization
python -m head_budget_optimization.run \
  --model-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --target-ratio "$TARGET_RATIO" \
  --n-articles "$N_ARTICLES" \
  --n-eval-points "$N_EVAL_POINTS" \
  --max-ratio 1.0 \
  --dataset-name "$DATASET_NAME" \
  --algorithm-config "$ALGORITHM_CONFIG" \
  --method "$METHOD" \
  --query-config "$QUERY_CONFIG" \
  --solve-ratios "$SOLVE_RATIOS" \
  --solver-method ratio-agnostic \
  --step-size 0.001 \
  --max-model-len 16384 \

echo ""
echo "=========================================="
echo "Smoke test completed successfully!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
