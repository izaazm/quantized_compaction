# Quantized Compaction

This project measures memory and quality when Attention Matching KV-cache
compaction is combined with KV quantization and BF16, INT8, or NF4 model
weights on TOFU, RULER-16K, and QuALITY. Stage 1 compares retained KV size and
key/value precision, Stage 2 compares quantization order, and Stage 3 evaluates
the best configuration across model precisions and denser retention ratios.

## Directory

- `src/quantized_compaction/`: experiment pipeline, compaction, quantization,
  evaluation, and result logging.
- `scripts/`: experiment launchers.
- `data/`: benchmark data.
- `vendor/attention_matching/`: Attention Matching implementation.
- `outputs/`: experiment results.

## Run

```bash
./setup_env.sh
source .venv/bin/activate
./scripts/run_all_experiments.sh
```
