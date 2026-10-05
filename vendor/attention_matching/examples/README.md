# Compaction Examples

This directory contains example scripts demonstrating various applications of the KV cache compaction methods.

## 0. Prep

To easily set up your environment, you can run the provided installation script from the `compaction` root directory:

```bash
cd ..
./setup_env.sh
source .venv/bin/activate
```

## 1. Modular Compaction on TOFU Dataset (`tofu_compaction_modular.py`)

This script provides a minimal reproduction of the modular KV cache compaction method evaluated on the TOFU dataset. It serves to test the composability of compacted KV caches when splitting the dataset across multiple caches, similar to the training-based cartridges experiments.

### Overview

The script performs the following steps:
1. **Monolithic Compaction**: Compresses the entire KV cache of $N$ authors into a single compacted cache of size $R$.
2. **Modular Compaction**: 
   - Splits the authors into Subset A and Subset B.
   - Compresses the KV cache of Subset A into a cache of size $R/2$.
   - Compresses the KV cache of Subset B into a cache of size $R/2$.
3. **Composition**: The compacted caches for Subset A and Subset B are composed by concatenating their compacted key, value, and beta tensors along the sequence dimension, resulting in a joint cache of size $R$.
4. **Fine-grained Evaluation**: All caches (Monolithic, Cache A, Cache B, and Composed A+B) are evaluated using `TOFUQAGenerateDataset` on:
   - Subset A questions
   - Subset B questions
   - All questions

### Usage

```bash
# Use Biography to self study
python -m examples.tofu_compaction_modular \
  --model llama3b --num-authors 2 --num-tokens 512 \
  --vllm --vllm-gpu-util 0.4 --vllm-max-model-len 4096 \
  --corpus-format biography --query-config ss-plus-repeat\
  --save-json --output-dir ./results_compaction_modular
```

### Arguments

- `--num-authors`: Total number of authors to load from the TOFU dataset (default: 10).
- `--num-tokens`: Total target budget ($R$) for the compacted cache (default: 64). For modular caches, the budget is split in half.
- `--model`: Hugging Face model identifier or a shorthand (`llama`, `llama3b`, `olmo`, `qwen`) (default: `qwen`).
- `--query-config`: The configuration for query generation during compaction (e.g., `repeat`, `ss-plus-repeat`).
- `--vllm`: Flag to instantiate a `vLLM` model to generate prompts and answers when using self-study configurations. Note: This requires additional GPU memory alongside the Hugging Face model.
- `--vllm-gpu-util`: GPU memory utilization for vLLM (default: `0.4`). Increase this if vLLM engine fails to start due to `No available memory for the cache blocks`.
- `--vllm-max-model-len`: Max model length for vLLM (default: `4096`). Decrease this to massively reduce the amount of KV cache memory vLLM attempts to pre-allocate.
- `--batch-size`: Batch size for generating answers during evaluation.

## 3. Sequential Modular Compaction (`tofu_compaction_sequential.py`)

This script implements a **sequential compaction pipeline** for modular KV caches. Unlike standard modular compaction where each author is compressed in isolation, this pipeline allows Author B to be compacted while **conditioned** on the already-compacted memory of Author A.

### Overview

The sequential pipeline performs:
1. **Phase 1: Compacting Author A**: Author A is compressed to $R/2$ tokens. The last $N$ tokens are preserved as an uncompacted "tail" to provide causal bridging.
2. **Global RoPE Shifting**: Before compacting Author B, its prefilled uncompacted KV cache is globally shifted forward by Author A's sequence length to ensure perfect temporal continuity.
3. **Phase 2: Query Generation for Author B**: 
   - **Independent**: Generates queries using only Author B's context.
   - **Guided**: Generates queries for Author B using Author A's compacted cache (and the uncompacted tail) as a prefix, capturing the interaction between the two modules.
4. **Phase 3: Sequential Compaction of B**: Author B is compressed to $R/2$ tokens using an exact sequential optimization objective. The solver explicitly accepts the past context ($P$) and the uncompacted tail ($S$) to compute the true exact global attention target, ensuring the OLS weighting perfectly matches inference conditions.
5. **Phase 4: Evaluation**: Compares the "Independent" and "Guided" strategies across Author A, Author B, Full (A+B), and Joint (Composition) performance.

### Usage

```bash
# Compare Guided vs Independent query generation
python -m examples.tofu_compaction_sequential \
  --model llama3b --num-authors 2 \
  --num-tokens 512 --guided-ss --query-config ss-plus-repeat \
  --vllm --vllm-gpu-util 0.4 --vllm-max-model-len 4096 \
  --corpus-format biography --uncompacted-tail-len 20 \
  --save-json --output-dir ./results_compaction_sequential_ss_plus_repeat_20
```

Results and model responses are saved to `./results_compaction_sequential/` by default.

## 4. Progressive Sequential Compaction (`tofu_compaction_progressive.py`)

This script implements a **progressive sequential compaction pipeline** across $N$ authors. It builds upon the sequential strategy but streams authors one by one, using the full biography of the *next* author as the uncompacted context tail ($S$) to flawlessly bridge the boundary between cartridges.

### Overview

The progressive pipeline performs:
1. **Dynamic Streaming**: Evaluates and compacts a stream of $N$ authors dynamically. At Phase $i$, it maintains the compacted cache of previous authors ($C_0 \dots C_{i-1}$) and loads Author $A_i$ and the next author $A_{i+1}$.
2. **Context-Aware Query Alignment**: Generates queries for Author $A_i$ (e.g., using self-study). The self-study queries are dynamically shifted in RoPE phase to mathematically align as if they were physically appended *after* the full text of $A_{i+1}$.
3. **Exact Boundary Bridging**: Compacts Author $A_i$ using the exact optimization objective where the *entire* next author ($A_{i+1}$) serves as the uncompacted context tail ($S$). This natively makes $A_i$ aware of $A_{i+1}$ during its compression.
4. **Progressive Evaluation**: Provides intermediate evaluations showing the "Before" and "After" accuracy on all currently seen authors at each phase to rigorously track factual retention.
5. **Final Preservation**: The final author in the stream is explicitly left uncompacted to maintain the most recent memory with perfect fidelity.

### Usage

```bash
# Run progressive compaction across 3 authors
python -m examples.tofu_compaction_progressive \
  --model llama3b --num-authors 4 --num-tokens-per-author 256 \
  --query-source next_bio_plus_ss --vllm \
  --vllm-gpu-util 0.4 --vllm-max-model-len 6144 \
  --corpus-format biography --on-policy-ss \
  --save-json --output-dir ./results_compaction_progressive
```

### Key Arguments

- `--num-authors`: Total number of authors to process sequentially in the stream (default: 3).
- `--num-tokens-per-author`: Compaction target budget ($R$) assigned independently to each author's module.
- `--query-source`: Determines the queries used to compact Author $i$. `next_bio` uses the full context of the next author. `next_bio_plus_ss` adds self-study queries about the current author appended after the next author.
- `--on-policy-ss`: Flag to enable on-policy self-study. When set, queries are generated using the compacted cache of all previous authors as context. By default, it is off-policy.

## 5. LLM-as-a-Judge Evaluation

```bash
python -m examples.llm_judge_eval \
    --input-dir ./results_compaction_sequential_ss_plus_repeat_0/
    --save-details
```
s
### Key Arguments

- `--num-tokens`: Total target budget ($R$). Split 50/50 between Author A and Author B.
- `--guided-ss`: Flag to enable "Guided" self-study query generation for the second module.
- `--uncompacted-tail-len`: Number of tokens to leave uncompacted at the end of each module to serve as a continuous context bridge (default: 20).
- `--corpus-format`: Format of the prefill text (`biography`, `qa`, or `answers`).
- `--output-dir`: Directory to save evaluation results and JSON responses.
