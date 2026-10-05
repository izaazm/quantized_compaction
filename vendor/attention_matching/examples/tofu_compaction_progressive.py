import os
import argparse
import time
import json
import sys
from typing import Dict, List, Tuple, Any, Optional

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from models.generate import (
    generate_with_compacted_cache_batch,
    get_sliding_layer_info
)

from cartridges.cartridges.data.tofu.evals import TOFUQAGenerateDataset
from cartridges.cartridges.data.tofu.utils import load_tofu_authors, authors_to_corpus_text
from compaction.compaction_methods.base import FullCacheCompactionAlgorithm
from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction
from compaction.query_generation import QueryGenerator
from models.cache import CompactedPrefixCache, CompactedPrefixLayer


_BIOGRAPHY_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cartridges", "data_synthesis", "tofu_author_biographies.json"
)


def _load_biography_index(json_path: str) -> dict:
    json_path = os.path.abspath(json_path)
    if not os.path.exists(json_path):
        print(f"  [biography] JSON not found at {json_path}, falling back to 'qa' format.")
        return {}
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
                print(f"  [biography] No biography for author {author.index} — falling back to 'qa' format.")
                lines = [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in author.qa_pairs]
                parts.append("\n\n".join(lines))
        else:
            parts.append(author.to_corpus_text())
    return "\n\n".join(parts)


def evaluate_compacted_cache_on_dataset(
    cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    model,
    tokenizer,
    original_seq_len: int,
    dataset_type: str = "tofu",
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
        dummy_dataset = TOFUQAGenerateDataset(config=TOFUQAGenerateDataset.Config(num_authors=1, author_offset=0, seed=seed), tokenizer=tokenizer, seed=seed)
    else:
        dataset = TOFUQAGenerateDataset(config=TOFUQAGenerateDataset.Config(num_authors=num_authors, author_offset=author_offset, seed=seed), tokenizer=tokenizer, seed=seed)
        eval_items = [{"prompt": it.prompt, "answer": it.answer, "convo_id": it.convo_id, "metadata": it.metadata} for it in dataset]
        dummy_dataset = dataset

    print(f"  [{label}] Evaluating {len(eval_items)} questions...")
    results = []
    for batch_start in tqdm(range(0, len(eval_items), batch_size), desc=f"Generating ({label})", leave=False):
        batch_end = min(batch_start + batch_size, len(eval_items))
        batch_items = eval_items[batch_start:batch_end]
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True) for it in batch_items]
        answers = generate_with_compacted_cache_batch(
            model=model, tokenizer=tokenizer, prompts=prompts, 
            compacted_cache=cache, max_new_tokens=max_new_tokens, 
            original_seq_len=original_seq_len, temperature=temperature
        )
        for i, pred in enumerate(answers):
            item = batch_items[i]
            score_val, extras = dummy_dataset.score(pred=pred, answer=item["answer"], convo_id=item["convo_id"])
            res = {"index": batch_start + i, "convo_id": item["convo_id"], "prompt": item["prompt"], "answer": item["answer"], "pred": pred, **(score_val if isinstance(score_val, dict) else {"score": score_val}), **extras}
            if "metadata" in item: res.update({"author_index": item["metadata"]["author_index"], "qa_index": item["metadata"]["qa_index"]})
            results.append(res)
    df = pd.DataFrame(results)
    if save_json and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        json_path = os.path.join(output_dir, f"results_{safe_label}.json")
        df.to_json(json_path, orient="records", indent=2)
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


def shift_uncompacted_cache(uncompacted_cache, offset, model):
    if uncompacted_cache[0][0].shape[2] == 0:
        return uncompacted_cache
    device = uncompacted_cache[0][0].device
    dtype = uncompacted_cache[0][0].dtype
    seq_len = uncompacted_cache[0][0].shape[2]
    
    current_positions = torch.arange(0, seq_len, device=device)
    target_positions = torch.arange(offset, offset + seq_len, device=device)
    cos_diff, sin_diff = compute_rope_correction(model, current_positions, target_positions, device, dtype)
    
    shifted = []
    for K, V in uncompacted_cache:
        K_shifted = apply_rotary_pos_emb_to_cache(K, cos_diff, sin_diff)
        shifted.append((K_shifted, V))
    return tuple(shifted)


def shift_queries(queries: torch.Tensor, offset: int, model: Any) -> torch.Tensor:
    """Shifts the RoPE phase of independent queries by the given offset."""
    device = queries.device
    dtype = queries.dtype
    seq_len = queries.shape[2]
    
    current_positions = torch.arange(0, seq_len, device=device)
    target_positions = torch.arange(offset, offset + seq_len, device=device)
    cos_diff, sin_diff = compute_rope_correction(model, current_positions, target_positions, device, dtype)
    
    queries_shifted = torch.zeros_like(queries)
    for l in range(queries.shape[0]):
        q_l = queries[l].unsqueeze(0) # [1, num_heads, seq_len, head_dim]
        q_shifted = apply_rotary_pos_emb_to_cache(q_l, cos_diff, sin_diff)
        queries_shifted[l] = q_shifted.squeeze(0)
    return queries_shifted


def concat_uncompacted_caches(cache1, cache2):
    if cache1[0][0].shape[2] == 0: return cache2
    if cache2[0][0].shape[2] == 0: return cache1
    composed = []
    for (K1, V1), (K2, V2) in zip(cache1, cache2):
        composed.append((torch.cat([K1, K2], dim=2), torch.cat([V1, V2], dim=2)))
    return tuple(composed)


def ensure_compacted_format(cache):
    """Wraps an uncompacted (K, V) cache into the compacted (C1, beta, C2) format."""
    if cache is None:
        return None
    if len(cache[0]) == 3:
        return cache
    wrapped = []
    for layer in cache:
        K, V = layer
        beta = torch.zeros(K.shape[0], K.shape[1], K.shape[2], device=K.device, dtype=K.dtype)
        wrapped.append((K, beta, V))
    return tuple(wrapped)


def construct_combined_cache_tuple(compacted_prefix, uncompacted_suffix):
    if uncompacted_suffix[0][0].shape[2] == 0:
        return compacted_prefix
        
    compacted_prefix = ensure_compacted_format(compacted_prefix)
    num_layers = len(compacted_prefix)
    combined = []
    for l in range(num_layers):
        C1_A, beta_A, C2_A = compacted_prefix[l]
        K_B, V_B = uncompacted_suffix[l]
        
        C1 = torch.cat([C1_A, K_B], dim=2)
        beta_B = torch.zeros(beta_A.shape[0], beta_A.shape[1], K_B.shape[2], device=beta_A.device, dtype=beta_A.dtype)
        beta = torch.cat([beta_A, beta_B], dim=2)
        C2 = torch.cat([C2_A, V_B], dim=2)
        combined.append((C1, beta, C2))
    return tuple(combined)


def compose_compacted_caches(
    cache_a: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    cache_b: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    if cache_a is None: return cache_b
    if cache_b is None: return cache_a
    composed = []
    for layer_a, layer_b in zip(cache_a, cache_b):
        C1_A, beta_A, C2_A = layer_a
        C1_B, beta_B, C2_B = layer_b
        C1 = torch.cat([C1_A, C1_B], dim=2)
        beta = torch.cat([beta_A, beta_B], dim=2)
        C2 = torch.cat([C2_A, C2_B], dim=2)
        composed.append((C1, beta, C2))
    return tuple(composed)


class SequentialCompactionAlgorithm(HighestAttentionKeysCompaction):
    """
    Implements the sequential compaction objective:
    Compact current context U given past compacted context P.
    """
    
    def __init__(self, *args, k_medoids_iters=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_medoids_iters = k_medoids_iters
        
    def compute_compacted_cache_sequential(
        self,
        U_K: torch.Tensor,
        U_V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        P_K: Optional[torch.Tensor] = None,
        P_beta: Optional[torch.Tensor] = None,
        P_V: Optional[torch.Tensor] = None,
        attention_bias_u: torch.Tensor = None,
        S_K: Optional[torch.Tensor] = None,
        S_V: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        n, d = queries.shape
        device = U_K.device
        dtype = U_K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Compute M_U, M_P, M_S, V_U, V_P, V_S
        U_scores32 = ((queries @ U_K.T.to(queries.dtype)) * inv_sqrt_d).to(torch.float32)
        if attention_bias_u is not None:
            U_scores32 += attention_bias_u.to(torch.float32)
        
        M_U = torch.logsumexp(U_scores32, dim=-1)
        V_U = torch.softmax(U_scores32, dim=-1) @ U_V.to(torch.float32)

        has_P = P_K is not None and P_K.shape[0] > 0
        has_S = S_K is not None and S_K.shape[0] > 0

        if has_P:
            P_scores32 = ((queries @ P_K.T.to(queries.dtype)) * inv_sqrt_d + P_beta.to(queries.dtype)).to(torch.float32)
            M_P = torch.logsumexp(P_scores32, dim=-1)
            V_P = torch.softmax(P_scores32, dim=-1) @ P_V.to(torch.float32)
            
        if has_S:
            S_scores32 = ((queries @ S_K.T.to(queries.dtype)) * inv_sqrt_d).to(torch.float32)
            M_S = torch.logsumexp(S_scores32, dim=-1)
            V_S = torch.softmax(S_scores32, dim=-1) @ S_V.to(torch.float32)
            
        # Global LSE: D_true in log-space
        lse_list = [M_U]
        if has_P: lse_list.append(M_P)
        if has_S: lse_list.append(M_S)
        
        global_lse = torch.logsumexp(torch.stack(lse_list, dim=0), dim=0)
        
        # True global output Y
        weight_U_true = torch.exp(M_U - global_lse)
        Y = weight_U_true.unsqueeze(1) * V_U
        
        if has_P:
            weight_P_true = torch.exp(M_P - global_lse)
            Y += weight_P_true.unsqueeze(1) * V_P
        else:
            V_P = torch.zeros(n, d, device=device, dtype=torch.float32)
            
        if has_S:
            weight_S_true = torch.exp(M_S - global_lse)
            Y += weight_S_true.unsqueeze(1) * V_S
        else:
            V_S = torch.zeros(n, d, device=device, dtype=torch.float32)

        # Normalized Attention Weights Matrix (Phi)
        U_attn_weights = torch.exp(U_scores32 - global_lse.unsqueeze(1))
        T_len = U_attn_weights.shape[1]
        t_actual = min(t, T_len)
        
        # Fast Initialization (Highest Attention Heuristic)
        if self.score_method == 'rms':
            key_scores = torch.sqrt((U_attn_weights ** 2).mean(dim=0))
        elif self.score_method == 'max':
            key_scores = U_attn_weights.max(dim=0)[0]
        else:
            key_scores = U_attn_weights.mean(dim=0)
            
        _, init_indices = torch.topk(key_scores, t_actual, largest=True)

        # =====================================================================
        # ATTENTION-SPACE K-MEDOIDS (Discrete Optimal Transport)
        # Replaces greedy subset selection with globally optimal profile clustering.
        # It perfectly preserves original keys to retain uncorrupted RoPE phases.
        # =====================================================================
        X = U_attn_weights.T  # Shape [T_len, n_queries]
        current_indices = init_indices.clone()
        t_actual = current_indices.size(0)
        
        arange_t = torch.arange(t_actual, device=device).unsqueeze(1)
        
        for _ in range(self.k_medoids_iters):
            # 1. Distances to medoids: [T_len, t_actual]
            dists = torch.cdist(X, X[current_indices])
            
            # 2. Hard Assignment
            assign = torch.argmin(dists, dim=1)
            
            # 3. Vectorized cluster means
            mask_matrix = (assign.unsqueeze(0) == arange_t)
            counts = mask_matrix.sum(dim=1, keepdim=True).float()
            cluster_means = (mask_matrix.float() @ X) / counts.clamp(min=1)
            
            # 4. Distances to the continuous mean
            d_to_mean = torch.norm(X - cluster_means[assign], dim=1)
            
            # 5. Find closest points using top-k
            masked_dists = torch.where(mask_matrix, d_to_mean.unsqueeze(0), float('inf'))
            k_candidates = min(10, X.size(0))
            _, topk_indices = torch.topk(masked_dists, k=k_candidates, dim=1, largest=False)
            
            # 6. Fallback and Duplicate Handling (Fast Python Loop on CPU)
            topk_indices_cpu = topk_indices.cpu().tolist()
            counts_cpu = counts.squeeze(1).cpu().long().tolist()
            current_indices_cpu = current_indices.cpu().tolist()
            
            new_indices = []
            used_indices = set()
            
            for c in range(t_actual):
                num_valid = counts_cpu[c]
                if num_valid == 0:
                    # Empty cluster safety fallback
                    fallback = current_indices_cpu[c]
                    new_indices.append(fallback)
                    used_indices.add(fallback)
                    continue
                
                assigned_idx = None
                check_len = min(k_candidates, num_valid)
                for i in range(check_len):
                    cand = topk_indices_cpu[c][i]
                    if cand not in used_indices:
                        assigned_idx = cand
                        break
                
                # Fallback if all tokens in cluster were somehow already used
                if assigned_idx is None:
                    assigned_idx = topk_indices_cpu[c][0]
                    
                new_indices.append(assigned_idx)
                used_indices.add(assigned_idx)
                
            current_indices = torch.tensor(new_indices, device=device, dtype=torch.long)
            
        top_indices = current_indices
        # =====================================================================

        # Select actual keys mapped to the resolved Medoids
        C1 = U_K[top_indices]

        # NNLS for beta
        C1_scores32 = (queries @ C1.T.to(queries.dtype)).to(torch.float32) * inv_sqrt_d
        
        # A_tilde_ij = exp(score) / D_true = exp(score - global_lse)
        A_tilde = torch.exp(C1_scores32 - global_lse.unsqueeze(1))
        
        # m_tilde_i = M_U / D_true = weight_U_true
        m_tilde = weight_U_true
        
        w = self._nnls_pg(A_tilde, m_tilde, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
        beta = torch.log(w.clamp(min=1e-12)).to(dtype)

        # OLS for C_v
        C1_beta_scores = C1_scores32 + beta.to(torch.float32)
        C1_beta_lse = torch.logsumexp(C1_beta_scores, dim=-1)
        
        compact_lse_list = [C1_beta_lse]
        if has_P: compact_lse_list.append(M_P)
        if has_S: compact_lse_list.append(M_S)
        
        compact_lse = torch.logsumexp(torch.stack(compact_lse_list, dim=0), dim=0)
        
        weight_P_compact = torch.exp(M_P - compact_lse).unsqueeze(1) if has_P else 0.0
        weight_S_compact = torch.exp(M_S - compact_lse).unsqueeze(1) if has_S else 0.0
            
        # X = exp(C1_beta_scores) / D_compact = exp(C1_beta_scores - compact_lse)
        X_lsq = torch.exp(C1_beta_scores - compact_lse.unsqueeze(1))
        
        # Z = Y - (M_P / D_compact) * V_P - (M_S / D_compact) * V_S
        Z = Y - weight_P_compact * V_P - weight_S_compact * V_S
        
        C2 = torch.linalg.lstsq(X_lsq, Z).solution.to(dtype)
        
        return C1, beta, C2, top_indices.tolist()


def run_compaction(model, tokenizer, queries, past_kv, target_size, compacted_past=None, algorithm_kwargs=None, uncompacted_tail=None):
    """Unified compaction function using the sequential objective."""
    num_layers = len(past_kv)
    num_heads = past_kv[0][0].shape[1]
    seq_algo = SequentialCompactionAlgorithm(**(algorithm_kwargs or {}))
    compacted_list = []
    
    for l in tqdm(range(num_layers), desc="Compacting layers", leave=False):
        C1_heads, beta_heads, C2_heads = [], [], []
        U_K_layer = past_kv[l][0][0]
        U_V_layer = past_kv[l][1][0]
        
        if compacted_past is not None:
            P_K_layer = compacted_past[l][0][0]
            P_beta_layer = compacted_past[l][1][0]
            P_V_layer = compacted_past[l][2][0]
        else:
            P_K_layer = P_beta_layer = P_V_layer = None
            
        if uncompacted_tail is not None and uncompacted_tail[l][0].shape[2] > 0:
            S_K_layer = uncompacted_tail[l][0][0]
            S_V_layer = uncompacted_tail[l][1][0]
        else:
            S_K_layer = S_V_layer = None
            
        for h in range(num_heads):
            q = queries[l, h]
            U_K = U_K_layer[h]
            U_V = U_V_layer[h]
            P_K = P_K_layer[h] if P_K_layer is not None else None
            P_beta = P_beta_layer[h] if P_beta_layer is not None else None
            P_V = P_V_layer[h] if P_V_layer is not None else None
            S_K = S_K_layer[h] if S_K_layer is not None else None
            S_V = S_V_layer[h] if S_V_layer is not None else None
            
            C1, beta, C2, _ = seq_algo.compute_compacted_cache_sequential(
                U_K, U_V, q, target_size, P_K, P_beta, P_V, S_K=S_K, S_V=S_V
            )
            C1_heads.append(C1.unsqueeze(0).unsqueeze(0))
            beta_heads.append(beta.unsqueeze(0).unsqueeze(0))
            C2_heads.append(C2.unsqueeze(0).unsqueeze(0))
            
        compacted_list.append((torch.cat(C1_heads, dim=1), torch.cat(beta_heads, dim=1), torch.cat(C2_heads, dim=1)))
        
    return tuple(compacted_list)


def run_progressive_eval(cache, label_prefix, seq_len, model, tokenizer, args, num_seen_authors):
    summary = {}
    print(f"\nEvaluating {label_prefix}...")
    
    # Per-author eval
    for i in range(num_seen_authors):
        author_label = f"Author {chr(65+i)}" # A, B, C...
        summary[f"{label_prefix} ({author_label})"] = evaluate_compacted_cache_on_dataset(
            cache, model, tokenizer, seq_len, dataset_type="tofu", num_authors=1, author_offset=i, 
            label=f"{label_prefix}-{author_label}", batch_size=args.batch_size, temperature=0.0, 
            seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
        )
        
    # Full eval
    summary[f"{label_prefix} (Full)"] = evaluate_compacted_cache_on_dataset(
        cache, model, tokenizer, seq_len, dataset_type="tofu", num_authors=num_seen_authors, author_offset=0, 
        label=f"{label_prefix}-Full", batch_size=args.batch_size, temperature=0.0, 
        seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
        
    return summary


def print_summary(summary: Dict[str, pd.DataFrame], num_authors, num_tokens_per_author, model_name):
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
    print("  PROGRESSIVE SEQUENTIAL COMPACTION SUMMARY")
    print(f"  {num_authors} authors | R={num_tokens_per_author} tokens per author | model={model_name}")
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
    parser = argparse.ArgumentParser(description="Progressive Sequential KV Cache Compaction on TOFU Dataset")
    parser.add_argument("--num-authors", type=int, default=3, help="Total number of authors in the experiment")
    parser.add_argument("--num-tokens-per-author", type=int, default=256, help="Compaction budget per author")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--query-source", type=str, default="next_bio_plus_ss", choices=["next_bio", "next_bio_plus_ss"], 
                        help="Source of queries to compact current author. 'next_bio' uses the full context of the next author. 'next_bio_plus_ss' adds self-study queries about the current author appended after the next author.")
    parser.add_argument("--on-policy-ss", action="store_true", help="Enable on-policy self-study (uses compacted past context during query generation)")
    
    parser.add_argument("--corpus-format", type=str, default="biography", choices=["answers", "qa", "biography"])
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE correction when composing caches (naive concatenation, matching tofu_compaction_modular behavior).")
    parser.add_argument("--vllm", action="store_true", help="Initialize vLLM model for self-study generation")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.4, help="GPU memory utilization for vLLM")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096, help="Max model length for vLLM")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generating answers")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results_compaction_progressive/")
    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_name = args.model
    if model_name == "llama":
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif model_name == "llama3b":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "olmo":
        model_name = "allenai/Olmo-3-7B-Instruct"
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
            enforce_eager=True,
        )
    
    print(f"\nLoading {args.num_authors} TOFU authors...")
    all_authors = load_tofu_authors(num_authors=args.num_authors, seed=args.seed)
    
    biography_index = _load_biography_index(_BIOGRAPHY_JSON) if args.corpus_format == "biography" else {}
    
    # We use our own logic to load the required configs depending on query_source
    q_next_config = load_query_config("context-prefill")
    q_ss_config = load_query_config("self-study")
    
    # Validation: Self-study requires vLLM
    if args.query_source == "next_bio_plus_ss" and vllm_model is None:
        print("\n[WARNING] 'next_bio_plus_ss' requires --vllm but it was not provided.")
        print("          Defaulting to 'next_bio' only.")
        args.query_source = "next_bio"

    algorithm_kwargs = {
        "score_method": "rms",
        "nnls_iters": 2,
        "c2_method": "lsq",
        "nnls_lower_bound": 0.05,
        "nnls_upper_bound": 20.0,
        "k_medoids_iters": 0, 
    }
    
    # Extract sliding layer info for CompactedPrefixCache
    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)
    
    # Generators will be instantiated inside the loop since we need different configs
    # for extracting Author C's tokens vs generating SS for Author B.
    
    compacted_past = None
    past_seq_len = 0
    full_history_fmt_ctx = ""
    
    # Authors list for labeling (A, B, C...)
    authors = [chr(65 + j) for j in range(args.num_authors)]
    
    # Load A (author 0)
    current_corpus = build_corpus([all_authors[0]], args.corpus_format, biography_index)
    seq_len_current, current_cache_unshifted, idx_current, fmt_ctx_current, _ = extract_full_kv_cache(model, tokenizer, current_corpus, args.device, model_name=model_name)
    current_cache = current_cache_unshifted
    full_history_fmt_ctx = fmt_ctx_current
    
    summary = {}
    
    for i in range(1, args.num_authors):
        char_current = chr(65 + i - 1)
        char_next = chr(65 + i)
        print(f"\n{'='*50}")
        print(f"=== [Phase {i}] Context: {char_current}, Next: {char_next} ===")
        print(f"{'='*50}")
        
        next_corpus = build_corpus([all_authors[i]], args.corpus_format, biography_index)
        seq_len_next, next_cache_unshifted, idx_next, fmt_ctx_next, _ = extract_full_kv_cache(model, tokenizer, next_corpus, args.device, model_name=model_name)
        
        if args.no_rope:
            next_cache_shifted = next_cache_unshifted
        else:
            next_cache_shifted = shift_uncompacted_cache(next_cache_unshifted, past_seq_len + seq_len_current, model)
            
        # 1. Eval [compacted_past] + Full(Current) + Full(Next)
        combined_uncompacted = concat_uncompacted_caches(current_cache, next_cache_shifted)
        if compacted_past is not None:
            eval_cache = construct_combined_cache_tuple(compacted_past, combined_uncompacted)
            # Create a label for the past: e.g., C_A+C_B
            # At iteration i, we have i-1 authors compacted in the past.
            past_label = "+".join([f"C_{authors[j]}" for j in range(i-1)])
            detail_label = f"({past_label} + Full {char_current} + Full {char_next})"
        else:
            eval_cache = ensure_compacted_format(combined_uncompacted)
            detail_label = f"(Full {char_current} + Full {char_next})"
            
        label_full = f"Phase {i}.0 {detail_label}"
        # We evaluate i+1 authors (since i starts at 1, authors are A...author_i+1)
        summary.update(run_progressive_eval(eval_cache, label_full, past_seq_len + seq_len_current + seq_len_next, model, tokenizer, args, num_seen_authors=i+1))
        
        # =====================================================================
        # 2. GENERATE QUERIES TO COMPRESS CURRENT AUTHOR (B)
        # As requested: We match attn(C_A, B, C + ss_B) with attn(C_A, C_B, C + ss_B).
        # Therefore, the queries used to compress B must be:
        #   1. The full tokens of C (the next author).
        #   2. (Optional) Self-study questions about B, appended AFTER C.
        # =====================================================================
        print(f"\nGenerating queries for compacting {char_current}...")
        
        # 2A. Extract Full Tokens of C (Next Author)
        # We use 'context-prefill' config which simply extracts the query vectors (Q) 
        # of the actual tokens provided in the formatted context.
        print(f"  Extracting query vectors for full context of {char_next}...")
        next_gen_wrapper = QueryGenerator(model=model, tokenizer=tokenizer, config=q_next_config, device=args.device, dtype=torch.float16)
        
        if args.on_policy_ss and compacted_past is not None:
            # Condition C on C_A + B
            # To extract queries of C while attending to C_A, we pass C_A + B as past_key_values
            # and C as the new formatted_context.
            q_next_pkv = CompactedPrefixCache(
                compacted_past, 
                original_seq_len=past_seq_len,
                sliding_layer_indices=sliding_layer_indices,
                sliding_window=sliding_window
            )
            q_next_fmt_ctx = fmt_ctx_current + fmt_ctx_next
            # We want queries for C, which is at the end of (B + C)
            q_next_idx = range(seq_len_current, seq_len_current + seq_len_next)
            
            queries_next, _, _ = next_gen_wrapper.generate_queries(
                formatted_context=q_next_fmt_ctx, 
                past_key_values=q_next_pkv, 
                indices=q_next_idx
            )
            # Since B was in the formatted_context, the queries for C start at pos (past_seq_len + seq_len_current)
            # No additional shift needed as the cache already handled the rope_base.
            queries_next_shifted = queries_next
        else:
            # Condition C only on B (Off-policy)
            q_next_fmt_ctx = fmt_ctx_current + fmt_ctx_next
            q_next_idx = range(seq_len_current, seq_len_current + seq_len_next)
            queries_next, _, _ = next_gen_wrapper.generate_queries(formatted_context=q_next_fmt_ctx, indices=q_next_idx)
            # Shift them by past_seq_len so they sit exactly at C's global position
            queries_next_shifted = shift_queries(queries_next, past_seq_len, model) if not args.no_rope else queries_next
            
        queries_final = queries_next_shifted
        
        # 2B. (Optional) Self-Study Queries about B
        if args.query_source == "next_bio_plus_ss":
            print(f"  Generating self-study queries about {char_current}...")
            ss_gen_wrapper = QueryGenerator(model=model, tokenizer=tokenizer, config=q_ss_config, device=args.device, dtype=torch.float16, vllm_model=vllm_model)
            
            if args.on_policy_ss and compacted_past is not None:
                # Generate SS using C_A + B as context
                # Wrap compacted_past (3-tuple) in CompactedPrefixCache for the generator
                q_ss_pkv_wrapped = CompactedPrefixCache(
                    compacted_past, 
                    original_seq_len=past_seq_len,
                    sliding_layer_indices=sliding_layer_indices,
                    sliding_window=sliding_window
                )
                
                # In on-policy mode, we pass the wrapped cache to generate_queries
                # q_ss_fmt_ctx should be the text that corresponds to the tokens AFTER the cache.
                # Since cache is C_A, fmt_ctx should be B.
                queries_ss, _, _ = ss_gen_wrapper.generate_queries(
                    formatted_context=fmt_ctx_current, 
                    past_key_values=q_ss_pkv_wrapped, 
                    indices=idx_current
                )
                
                # These queries naturally sit after B. We want them after C. Shift by len(C).
                ss_shift = seq_len_next
            else:
                # Generate SS using only B as context
                q_ss_past_kv = current_cache_unshifted
                q_ss_fmt_ctx = fmt_ctx_current
                q_ss_idx = idx_current
                
                queries_ss, _, _ = ss_gen_wrapper.generate_queries(formatted_context=q_ss_fmt_ctx, past_key_values=q_ss_past_kv, indices=q_ss_idx)
                
                # These queries naturally sit after B (pos seq_len_current). 
                # We want them after C (pos past_seq_len + seq_len_current + seq_len_next).
                # Shift by past_seq_len + seq_len_next.
                ss_shift = past_seq_len + seq_len_next
                
            queries_ss_shifted = shift_queries(queries_ss, ss_shift, model) if not args.no_rope else queries_ss
            queries_final = torch.cat([queries_final, queries_ss_shifted], dim=2)
            
        # 3. Compact Current Author
        print(f"\nCompacting {char_current} using queries from {char_next}...")
        # S = next_cache_shifted (as uncompacted tail)
        C_current = ensure_compacted_format(run_compaction(
            model, tokenizer, queries_final, current_cache, args.num_tokens_per_author, 
            algorithm_kwargs=algorithm_kwargs, uncompacted_tail=next_cache_shifted
        ))
        
        compacted_past = compose_compacted_caches(compacted_past, C_current)
        past_seq_len += seq_len_current 
        
        # 5. Eval [compacted_past] + Full(Next)
        eval_cache_after = construct_combined_cache_tuple(compacted_past, next_cache_shifted)
        # We just compacted char_current, so we have i authors in the past now.
        past_label = "+".join([f"C_{authors[j]}" for j in range(i)])
        label_after = f"Phase {i}.1 ({past_label} + Full {char_next})"
        summary.update(run_progressive_eval(eval_cache_after, label_after, past_seq_len + seq_len_next, model, tokenizer, args, num_seen_authors=i+1))
        
        # Advance loop
        current_cache = next_cache_shifted
        current_cache_unshifted = next_cache_unshifted
        seq_len_current = seq_len_next
        fmt_ctx_current = fmt_ctx_next
        idx_current = idx_next
        full_history_fmt_ctx += fmt_ctx_next

    # Print Summary
    print_summary(summary, args.num_authors, args.num_tokens_per_author, model_name)

if __name__ == "__main__":
    main()
