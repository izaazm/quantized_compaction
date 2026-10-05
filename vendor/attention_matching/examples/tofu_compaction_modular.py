import os
import argparse
import time
import json
import sys
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed as hf_set_seed

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Setup paths for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from evaluation.utils import (
    load_model_and_tokenizer,
    extract_full_kv_cache,
)
from evaluation.configs.utils import load_query_config
from compaction.compaction_methods.registry import get_compaction_method
from models.generate import generate_with_compacted_cache_batch

from cartridges.cartridges.data.tofu.evals import TOFUQAGenerateDataset
from cartridges.cartridges.data.tofu.utils import load_tofu_authors, authors_to_corpus_text


_BIOGRAPHY_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cartridges", "data_synthesis", "tofu_author_biographies.json"
)


def _load_biography_index(json_path: str) -> dict:
    json_path = os.path.abspath(json_path)
    if not os.path.exists(json_path):
        print(f"  [biography] JSON not found at {json_path}, falling back to 'qa' format.")
        return {}
    import json
    with open(json_path) as f:
        data = json.load(f)
    return {entry["author_idx"]: entry for entry in data}


def build_corpus(
    authors,
    corpus_format: str,
    biography_index: dict,
) -> str:
    parts = []
    for author in authors:
        if corpus_format == "qa":
            lines = [
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in author.qa_pairs
            ]
            parts.append("\n\n".join(lines))

        elif corpus_format == "biography":
            entry = biography_index.get(author.index)
            if entry is not None:
                parts.append(entry["biography"])
            else:
                # Graceful fallback: use structured Q&A
                print(
                    f"  [biography] No biography for author {author.index} "
                    f"— falling back to 'qa' format."
                )
                lines = [
                    f"Q: {qa['question']}\nA: {qa['answer']}"
                    for qa in author.qa_pairs
                ]
                parts.append("\n\n".join(lines))

        else:  # "answers" — original behaviour
            parts.append(author.to_corpus_text())

    return "\n\n".join(parts)


def evaluate_compacted_cache_on_dataset(
    cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    model,
    tokenizer,
    original_seq_len: int,
    dataset_type: str = "tofu", # "tofu" or "composition"
    num_authors: int = 1,
    author_offset: int = 0,
    label: str = "eval",
    batch_size: int = 32,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    save_json: bool = False,
    output_dir: str = ".",
) -> pd.DataFrame:
    """Generic evaluation function for compacted caches on TOFU or Composition datasets."""
    
    if dataset_type == "composition":
        json_path = os.path.join(root_dir, "cartridges", "data_synthesis", "tofu_composition_eval.json")
        if not os.path.exists(json_path):
            print(f"  [composition] Dataset not found at {json_path}, skipping.")
            return pd.DataFrame()
        with open(json_path) as f:
            data = json.load(f)

        items = [item for item in data["items"] if item["author_a_idx"] == 0 and item["author_b_idx"] == 1]
        if not items: return pd.DataFrame()
        
        eval_items = [{"prompt": it["question"], "answer": it["answer"], "convo_id": it["question_id"]} for it in items]
        dummy_dataset = TOFUQAGenerateDataset(
            config=TOFUQAGenerateDataset.Config(num_authors=1, author_offset=0, seed=seed),
            tokenizer=tokenizer, seed=seed
        )
    else:
        dataset = TOFUQAGenerateDataset(config=TOFUQAGenerateDataset.Config(num_authors=num_authors, author_offset=author_offset, seed=seed), tokenizer=tokenizer, seed=seed)
        eval_items = [{"prompt": it.prompt, "answer": it.answer, "convo_id": it.convo_id, "metadata": it.metadata} for it in dataset]
        dummy_dataset = dataset

    print(f"  [{label}] Evaluating {len(eval_items)} questions...")
    results = []
    
    for batch_start in tqdm(range(0, len(eval_items), batch_size), desc=f"Generating ({label})", leave=False):
        batch_end = min(batch_start + batch_size, len(eval_items))
        batch_items = eval_items[batch_start:batch_end]
        
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True)
            for it in batch_items
        ]
        
        answers = generate_with_compacted_cache_batch(
            model=model, tokenizer=tokenizer, prompts=prompts, compacted_cache=cache,
            max_new_tokens=max_new_tokens, original_seq_len=original_seq_len,
            temperature=temperature
        )
        
        for i, pred in enumerate(answers):
            item = batch_items[i]
            score_val, extras = dummy_dataset.score(pred=pred, answer=item["answer"], convo_id=item["convo_id"])
            
            res = {
                "index": batch_start + i,
                "convo_id": item["convo_id"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "pred": pred,
                **(score_val if isinstance(score_val, dict) else {"score": score_val}),
                **extras,
            }
            if "metadata" in item:
                res.update({"author_index": item["metadata"]["author_index"], "qa_index": item["metadata"]["qa_index"]})
            results.append(res)
            
    df = pd.DataFrame(results)
    if save_json and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        json_path = os.path.join(output_dir, f"results_{safe_label}.json")
        df.to_json(json_path, orient="records", indent=2)
        print(f"  Saved detailed results to {json_path}")
        
    return df


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_to_cache(
    cache: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (cache * cos) + (rotate_half(cache) * sin)


def compute_rope_correction(
    model: Any,
    current_positions: torch.Tensor,
    target_positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    use_local_rope: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model

    if use_local_rope and hasattr(base_model, 'rotary_emb_local'):
        rotary_emb = base_model.rotary_emb_local
    else:
        rotary_emb = base_model.rotary_emb

    if current_positions.dim() == 1:
        current_positions = current_positions.unsqueeze(0)
    if target_positions.dim() == 1:
        target_positions = target_positions.unsqueeze(0)

    position_diff = target_positions - current_positions
    dummy = torch.zeros(1, 1, rotary_emb.inv_freq.shape[0] * 2, device=device, dtype=dtype)
    cos_diff, sin_diff = rotary_emb(dummy, position_diff.to(device))
    return cos_diff.to(dtype), sin_diff.to(dtype)


def compose_compacted_caches_with_rope(
    cache_a: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    cache_b: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    model: Any,
    seq_len_a: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Compose two compacted caches by applying a uniform RoPE phase shift to B's keys.
    
    Each key in B was originally at position p with RoPE phase R(p). To place it at
    global position p + seq_len_a, we apply R(seq_len_a) uniformly to all keys.
    This works because R(-p) * R(p + seq_len_a) = R(seq_len_a).
    """
    device = cache_a[0][0].device
    dtype = cache_a[0][0].dtype
    
    L_B = cache_b[0][0].shape[2]
    
    # Uniform shift: every key in B shifts by seq_len_a positions
    current_positions = torch.zeros(L_B, device=device, dtype=torch.long)
    target_positions = torch.full((L_B,), seq_len_a, device=device, dtype=torch.long)
    cos_diff, sin_diff = compute_rope_correction(model, current_positions, target_positions, device, dtype)
    
    composed = []
    for layer_a, layer_b in zip(cache_a, cache_b):
        C1_A, beta_A, C2_A = layer_a
        C1_B, beta_B, C2_B = layer_b
        
        C1_B_shifted = apply_rotary_pos_emb_to_cache(C1_B, cos_diff, sin_diff)
        
        C1 = torch.cat([C1_A, C1_B_shifted], dim=2)
        beta = torch.cat([beta_A, beta_B], dim=2)
        C2 = torch.cat([C2_A, C2_B], dim=2)
        
        composed.append((C1, beta, C2))

    return tuple(composed)


def compose_modules_with_rope(
    modules: List[Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], int]],
    model: Any,
) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], int]:
    """Compose N independently compacted modules with RoPE correction.
    
    Each module is a (compacted_cache, original_seq_len) pair. Modules are placed
    in sequence: module 0 at positions [0, seq_len_0), module 1 at [seq_len_0, 
    seq_len_0 + seq_len_1), etc. A uniform RoPE shift is applied to each module's
    keys to align them to their global position.
    
    Parameters
    ----------
    modules : list of (compacted_cache, original_seq_len) pairs
        Each compacted_cache is a tuple of (C1, beta, C2) per layer.
        original_seq_len is the pre-compaction sequence length (stored as metadata).
    model : Any
        Model used to compute RoPE embeddings.
        
    Returns
    -------
    composed_cache : tuple of (C1, beta, C2) per layer
    total_seq_len : int
        Sum of all original_seq_lens (use as original_seq_len for evaluation).
    """
    if len(modules) == 0:
        raise ValueError("Need at least one module to compose")
    if len(modules) == 1:
        return modules[0]
    
    device = modules[0][0][0][0].device
    dtype = modules[0][0][0][0].dtype
    num_layers = len(modules[0][0])
    
    # Start with the first module (no shift needed — it's at position 0)
    result_layers = [list(layer) for layer in modules[0][0]]  # make mutable
    cumulative_offset = modules[0][1]  # seq_len of first module
    
    for cache, seq_len in modules[1:]:
        L = cache[0][0].shape[2]  # compacted length of this module
        
        # Compute uniform RoPE shift by cumulative_offset
        current_positions = torch.zeros(L, device=device, dtype=torch.long)
        target_positions = torch.full((L,), cumulative_offset, device=device, dtype=torch.long)
        cos_diff, sin_diff = compute_rope_correction(
            model, current_positions, target_positions, device, dtype
        )
        
        for l in range(num_layers):
            C1, beta, C2 = cache[l]
            C1_shifted = apply_rotary_pos_emb_to_cache(C1, cos_diff, sin_diff)
            
            result_layers[l][0] = torch.cat([result_layers[l][0], C1_shifted], dim=2)
            result_layers[l][1] = torch.cat([result_layers[l][1], beta], dim=2)
            result_layers[l][2] = torch.cat([result_layers[l][2], C2], dim=2)
        
        cumulative_offset += seq_len
    
    composed = tuple((l[0], l[1], l[2]) for l in result_layers)
    return composed, cumulative_offset


def compose_compacted_caches(
    cache_a: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    cache_b: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Compose two compacted caches by concatenating their C1, beta, C2 along the sequence dimension (dim=2)."""
    composed = []
    for layer_a, layer_b in zip(cache_a, cache_b):
        C1_A, beta_A, C2_A = layer_a
        C1_B, beta_B, C2_B = layer_b
        
        C1 = torch.cat([C1_A, C1_B], dim=2)
        beta = torch.cat([beta_A, beta_B], dim=2)
        C2 = torch.cat([C2_A, C2_B], dim=2)
        
        composed.append((C1, beta, C2))

    return tuple(composed)


def print_summary(summary: Dict[str, pd.DataFrame], args):
    """Print comparison table."""
    all_score_cols = set()
    for df in summary.values():
        all_score_cols.update(col for col in df.columns if "rouge" in col.lower() or col == "score")
    score_cols = sorted(all_score_cols)
    if not score_cols:
        print("\n  No score columns found in results.")
        return

    name_width = max(len(name) for name in summary) + 2
    col_width = 12
    header = f"{'Cartridge':<{name_width}}" + "".join(f"{col:>{col_width}}" for col in score_cols) + f"{'n_questions':>{col_width}}"
    separator = "-" * len(header)

    print("\n" + separator)
    print("  MODULAR COMPACTION COMPARISON SUMMARY")
    print(f"  {args.num_authors} authors | R={args.num_tokens} tokens | model={args.model}")
    print(separator)
    print(header)
    print(separator)

    prev_group = None
    thin_sep = "·" * len(header)
    for name, df in summary.items():
        group = name.split("(")[0].strip() if "(" in name else name
        if prev_group is not None and group != prev_group:
            print(thin_sep)
        prev_group = group

        row = f"{name:<{name_width}}"
        for col in score_cols:
            if col in df.columns:
                row += f"{df[col].mean():>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}}"
        row += f"{len(df):>{col_width}}"
        print(row)

    print(separator)


def main():
    parser = argparse.ArgumentParser(description="Modular KV Cache Compaction on TOFU Dataset")
    parser.add_argument("--num-authors", type=int, default=2, help="Total number of authors in the experiment")
    parser.add_argument("--num-tokens", type=int, default=512, help="Total budget of compacted tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="qwen", help="HuggingFace model name or shorthand (llama, llama3b, olmo, qwen)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query-config", type=str, default="ss-plus-repeat", help="Query config file name")
    parser.add_argument("--vllm", action="store_true", help="Initialize vLLM model for self-study generation")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.4, help="GPU memory utilization for vLLM (increase if failing with no available memory)")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096, help="Max model length for vLLM (decrease to save KV cache memory)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generating answers")
    parser.add_argument("--save-json", action="store_true", help="Save the generated answers and scores to JSON files")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save JSON files if --save-json is used")
    parser.add_argument("--corpus-format", type=str, default="answers", choices=["answers", "qa", "biography"])
    parser.add_argument("--biography-json", type=str, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    
    model_name = args.model
    if model_name == "llama":
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif model_name == "llama3b":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "olmo":
        model_name = "allenai/OLMo-3-7B-Instruct"
    elif model_name == "qwen":
        model_name = "Qwen/Qwen3-4b"
        
    print(f"Loading {model_name} ...")
    model, tokenizer = load_model_and_tokenizer(model_name, device=args.device)
    
    vllm_model = None
    if args.vllm:
        print(f"Loading vLLM model for self-study generation ...")
        from vllm import LLM
        vllm_model = LLM(
            model=model_name, 
            gpu_memory_utilization=args.vllm_gpu_util, 
            max_model_len=args.vllm_max_model_len,
            enforce_eager=True, # Adding this to bypass torch.compile overhead shown in your logs
        )
        
    print(f"\nLoading {args.num_authors} TOFU authors...")
    all_authors = load_tofu_authors(num_authors=args.num_authors, seed=args.seed)
    half = args.num_authors // 2
    half_tokens = args.num_tokens // 2

    bio_json_path = args.biography_json or _BIOGRAPHY_JSON
    biography_index = _load_biography_index(bio_json_path) if args.corpus_format == "biography" else {}
    print(f"Corpus format: '{args.corpus_format}'" +
          (f" ({len(biography_index)} biographies loaded)" if biography_index else ""))

    mono_corpus = build_corpus(all_authors, args.corpus_format, biography_index)
    a_corpus    = build_corpus(all_authors[:half], args.corpus_format, biography_index)
    b_corpus    = build_corpus(all_authors[half:], args.corpus_format, biography_index)

    algorithm_kwargs = {
        "algorithm": "highest_attention_keys",
        "score_method": "rms",
        "nnls_iters": 2,
        "nnls_lower_bound": 0.05,
        "nnls_upper_bound": 20.0,
        "c2_method": "lsq",
    }
    compaction_method = get_compaction_method("AM-HighestAttnKeys", method_kwargs=algorithm_kwargs)
    query_config = load_query_config(args.query_config)
    
    summary = {}
    
    # --- Monolithic Compaction ---
    print(f"\n=== Monolithic Compaction (All {args.num_authors} authors) ===")
    seq_len_mono, past_kv_mono, idx_mono, fmt_ctx_mono, _ = extract_full_kv_cache(
        model, tokenizer, mono_corpus, args.device, model_name=model_name,
    )
    print(f"  Extracted full KV cache: {seq_len_mono} tokens. Compacting to {args.num_tokens} tokens...")
    t0 = time.time()
    compacted_mono, _ = compaction_method.compact_kv_cache(
        past_key_values=past_kv_mono,
        target_size=args.num_tokens,
        indices=idx_mono,
        query_config=query_config,
        model=model,
        tokenizer=tokenizer,
        formatted_context=fmt_ctx_mono,
        vllm_model=vllm_model,
        compute_stats=False,
    )
    dt = time.time() - t0
    print(f"  Compaction took {dt:.2f}s")
    
    print(f"\n--- Monolithic on all {args.num_authors} authors ---")
    summary["Mono (all)"] = evaluate_compacted_cache_on_dataset(
        compacted_mono, model, tokenizer, seq_len_mono, dataset_type="tofu", num_authors=args.num_authors, label="mono-all", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Monolithic on A-only ---")
    summary["Mono (A-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_mono, model, tokenizer, seq_len_mono, dataset_type="tofu", num_authors=half, author_offset=0, label="mono-A", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Monolithic on B-only ---")
    summary["Mono (B-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_mono, model, tokenizer, seq_len_mono, dataset_type="tofu", num_authors=args.num_authors - half, author_offset=half, label="mono-B", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Monolithic on Joint Questions (Authors 0 & 1) ---")
    summary["Mono (Joint)"] = evaluate_compacted_cache_on_dataset(
        compacted_mono, model, tokenizer, seq_len_mono, dataset_type="composition", label="mono-joint", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    
    # --- Modular Compaction ---
    # Phase A
    print(f"\n=== Modular Compaction: Phase A (Authors 0-{half-1}) ===")
    seq_len_a, past_kv_a, idx_a, fmt_ctx_a, _ = extract_full_kv_cache(
        model, tokenizer, a_corpus, args.device, model_name=model_name,
    )
    compacted_a, _ = compaction_method.compact_kv_cache(
        past_key_values=past_kv_a, target_size=half_tokens, indices=idx_a, query_config=query_config, model=model, tokenizer=tokenizer, formatted_context=fmt_ctx_a, vllm_model=vllm_model, compute_stats=False,
    )
    
    print(f"\n--- Cache A on A-only ---")
    summary["Cache A (A-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_a, model, tokenizer, seq_len_a, dataset_type="tofu", num_authors=half, author_offset=0, label="cart-A-on-A", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Cache A on B-only ---")
    summary["Cache A (B-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_a, model, tokenizer, seq_len_a, dataset_type="tofu", num_authors=args.num_authors - half, author_offset=half, label="cart-A-on-B", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Cache A on Joint Questions (Authors 0 & 1) ---")
    summary["Cache A (Joint)"] = evaluate_compacted_cache_on_dataset(
        compacted_a, model, tokenizer, seq_len_a, dataset_type="composition", label="cache-A-joint", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    
    # Phase B
    print(f"\n=== Modular Compaction: Phase B (Authors {half}-{args.num_authors-1}) ===")
    seq_len_b, past_kv_b, idx_b, fmt_ctx_b, _ = extract_full_kv_cache(
        model, tokenizer, b_corpus, args.device, model_name=model_name,
    )
    compacted_b, _ = compaction_method.compact_kv_cache(
        past_key_values=past_kv_b, target_size=args.num_tokens - half_tokens, indices=idx_b, query_config=query_config, model=model, tokenizer=tokenizer, formatted_context=fmt_ctx_b, vllm_model=vllm_model, compute_stats=False,
    )
    
    print(f"\n--- Cache B on B-only ---")
    summary["Cache B (B-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_b, model, tokenizer, seq_len_b, dataset_type="tofu", num_authors=args.num_authors - half, author_offset=half, label="cart-B-on-B", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Cache B on A-only ---")
    summary["Cache B (A-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_b, model, tokenizer, seq_len_b, dataset_type="tofu", num_authors=half, author_offset=0, label="cart-B-on-A", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Cache B on Joint Questions (Authors 0 & 1) ---")
    summary["Cache B (Joint)"] = evaluate_compacted_cache_on_dataset(
        compacted_b, model, tokenizer, seq_len_b, dataset_type="composition", label="cache-B-joint", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    
    # Composition
    print(f"\n=== Composition: A + B ===")
    compacted_composed = compose_compacted_caches(compacted_a, compacted_b)
    seq_len_composed = max(seq_len_a, seq_len_b)
    
    print(f"\n--- Composed A+B on all {args.num_authors} authors ---")
    summary["A+B (all)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed, model, tokenizer, seq_len_a + seq_len_b, dataset_type="tofu", num_authors=args.num_authors, label="composed-all", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Composed A+B on A-only ---")
    summary["A+B (A-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed, model, tokenizer, seq_len_a + seq_len_b, dataset_type="tofu", num_authors=half, author_offset=0, label="composed-A", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Composed A+B on B-only ---")
    summary["A+B (B-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed, model, tokenizer, seq_len_a + seq_len_b, dataset_type="tofu", num_authors=args.num_authors - half, author_offset=half, label="composed-B", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Composed A+B on Joint Questions (Authors 0 & 1) ---")
    summary["A+B (Joint)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed, model, tokenizer, seq_len_a + seq_len_b, dataset_type="composition", label="composed-joint", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )

    print(f"\n=== Composition with RoPE Correction: A + B ===")
    compacted_composed_rope = compose_compacted_caches_with_rope(compacted_a, compacted_b, model, seq_len_a=seq_len_a)

    print(f"\n--- Composed A+B RoPE on all {args.num_authors} authors ---")
    summary["A+B RoPE (all)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed_rope, model, tokenizer, seq_len_a + seq_len_b, dataset_type="tofu", num_authors=args.num_authors, label="composed-rope-all", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Composed A+B RoPE on A-only ---")
    summary["A+B RoPE (A-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed_rope, model, tokenizer, seq_len_a + seq_len_b, dataset_type="tofu", num_authors=half, author_offset=0, label="composed-rope-A", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Composed A+B RoPE on B-only ---")
    summary["A+B RoPE (B-only)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed_rope, model, tokenizer, seq_len_a + seq_len_b, dataset_type="tofu", num_authors=args.num_authors - half, author_offset=half, label="composed-rope-B", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    print(f"\n--- Composed A+B RoPE on Joint Questions (Authors 0 & 1) ---")
    summary["A+B RoPE (Joint)"] = evaluate_compacted_cache_on_dataset(
        compacted_composed_rope, model, tokenizer, seq_len_a + seq_len_b, dataset_type="composition", label="composed-rope-joint", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    
    print_summary(summary, args)
    

if __name__ == "__main__":
    main()
