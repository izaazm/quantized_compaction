# Quantized Compaction Agent Notes

## Purpose

This repository measures the memory and quality tradeoff between Attention
Matching KV-cache compaction, independent key/value quantization, and
model-weight quantization. Keep the implementation focused on this experiment.
Do not add unrelated compaction methods, automatic Pareto selection,
visualization, or a Stage 4 unless the user explicitly requests them.

The compaction method is the lightweight `AM-HighestAttnKeys` path with
context-prefill queries. For retained-entry ratio `r`, key precision `k`, and
value precision `v`, the ideal KV byte budget is:

```text
ideal_budget_fraction = r * (k + v) / 32
```

## Pipeline

The experiment ends after Stage 3.

Stage 1 measures the cardinality-by-KV-precision surface under BF16 model
weights. It contains nine dense K/V precision controls, every K/V precision
pair at equal 10%, 5%, and 2% ideal byte budgets, and every K/V precision pair
at fixed 10%, 5%, and 2% retained-entry ratios. Duplicate condition IDs are
removed. Stage 1 selects the three symmetric pairs K16V16, K8V8, and K4V4 plus
the two best asymmetric pairs for Stage 2. The asymmetric pairs are selected
only by measured performance across every configured dataset and budget, with
memory as the final tiebreaker.

Stage 2 measures composition order. It evaluates the five Stage 1 precision
pairs at the 10%, 5%, and 2% ideal byte budgets with `post`, `pre`, and `aware`
composition. It does not rerun dense controls, and it does not select or
promote a family into Stage 3.

Stage 3 is independent of Stages 1 and 2. It evaluates K16V16, K8V8, and K4V4
with `post` composition at retained-entry ratios 1.0, 0.75, 0.50, 0.25, 0.10,
and 0.05 under BF16, INT8, and NF4 model weights. The r=1.0 conditions use no
compaction. Stage 3 evaluates 18 conditions per model precision and 54
configurations in total.

## Datasets

The pipeline supports TOFU, RULER-16K, and QuALITY. The launcher currently
defaults to all three through `DATASETS="tofu ruler quality"`.

TOFU uses the local biographies and QA passages. Its full configuration
evaluates 200 questions.

RULER uses the prepared 16K subset with 150 `fwe`, 75 `qa_1` SQuAD, and 75
`qa_2` HotPotQA examples. The two QA sources remain identifiable in raw rows
and are combined into the reported QA metric.

QuALITY uses the official v1.0.1 HTML-stripped development set. The single
`QUALITY_ARTICLES` setting applies to all three stages. Every question attached
to each selected article is evaluated. Questions from the same article reuse
one prefetched and compacted context cache, and generation is batched within
that article.

Benchmark preparation is handled automatically by
`scripts/run_all_experiments.sh` unless `PREPARE_BENCHMARKS=false` is set.

## Quantization

KV INT8 and INT4 use QDQ simulation. Attention consumes BF16 tensors restricted
to the requested quantization grid. The runner audits the final QDQ cache and
fails if it is not fixed on that grid.

The composition modes are:

- `post`: compact first, then quantize.
- `pre`: quantize the original cache before compaction, then quantize the
  compacted cache.
- `aware`: fake-quantize selected keys during fitting and apply final QDQ.

Model INT8 and NF4 use BitsAndBytes quantized linear modules rather than fake
weight quantization. The runner verifies the loaded module type and records the
actual model weight footprint.

The result rows record requested key and value bits, quantization error and
verification fields, logical packed KV bytes, physical resident QDQ bytes,
model weight bytes, combined model-plus-logical-KV bytes, CUDA peak allocation,
NLL, perplexity, generated-answer metrics, dataset accuracy, counts, and
timing. Raw prediction rows retain the dataset, context, task, question,
reference, and generated answer.

Do not add Pareto analysis to the experiment runner. The saved measurements
are the inputs for later memory-versus-performance analysis.

## Important implementation files

- `src/quantized_compaction/suites.py` defines the Stage 1 and Stage 2 grids and
  the Stage 1-to-2 funnel and independent Stage 3 grid.
- `src/quantized_compaction/pipeline.py` runs Stages 1-3 sequentially, resumes only
  exact completed configurations, and writes combined summaries.
- `src/quantized_compaction/runner.py` loads models and data, runs compaction and
  evaluation, audits quantization, measures memory, and writes results.
- `src/quantized_compaction/am_backend.py` integrates the vendored Attention
  Matching implementation and prepares evaluation prompts.
- `src/quantized_compaction/quantized_am.py` implements quantization-aware fitting.
- `src/quantized_compaction/quantization.py` implements KV fake quantization and
  packed-memory accounting.
- `src/quantized_compaction/data.py` loads TOFU, RULER, and QuALITY corpora.
- `src/quantized_compaction/metrics.py` implements generated-answer metrics.
- `src/quantized_compaction/prepare_benchmarks.py` prepares RULER and QuALITY data.
- `scripts/run_all_experiments.sh` is the main Stage 1-3 launcher.
- `scripts/run_experiment.sh` runs a standalone experiment.
- `vendor/attention_matching/` contains the vendored compaction dependency.
- `data/` contains TOFU inputs and prepared benchmark data.
- `outputs/pipelines/<name>/` contains pipeline artifacts.

## Setup and execution

Set up the environment with:

```bash
./setup_env.sh
source .venv/bin/activate
```

Preview the resolved plan without loading a model with:

```bash
PLAN_ONLY=true ./scripts/run_all_experiments.sh
```

Run the full default experiment with TOFU, RULER, and QuALITY with:

```bash
PIPELINE_NAME=qwen4b-stage123 ./scripts/run_all_experiments.sh
```

Select datasets with a space-separated `DATASETS` value. For example:

```bash
DATASETS="tofu ruler" PIPELINE_NAME=qwen4b-tofu-ruler \
./scripts/run_all_experiments.sh
```

Set the same QuALITY article count for every stage with:

```bash
DATASETS=quality QUALITY_ARTICLES=50 PIPELINE_NAME=qwen4b-quality50 \
./scripts/run_all_experiments.sh
```

The launcher defaults to 5,000 context-prefill queries per KV head for every
stage. `RESUME=true` reuses only runs whose complete saved configuration and
condition list exactly match the requested run. Always use a new
`PIPELINE_NAME` when changing a configuration unless resuming the same run.

## Output layout

```text
outputs/pipelines/<pipeline-name>/
  settings.env
  environment.log
  plan.json
  resolved_plan.json
  pipeline_manifest.json
  combined_summary.jsonl
  combined_summary.csv
  stage1/diagnostics.json
  stage2/diagnostics.json
  stage3/design.json
  stage1/<dataset>/runs/<run>/...
  stage2/<dataset>/runs/<run>/...
  stage3/<dataset>/runs/<weight-run>/...
```

## Working rules

Keep changes minimal and scoped to the request. Do not invent experiment
families, baselines, datasets, stages, or analysis. Preserve downloaded outputs
and unrelated user files. Use `apply_patch` for source and documentation edits.
Run relevant checks only after the requested implementation is complete.
