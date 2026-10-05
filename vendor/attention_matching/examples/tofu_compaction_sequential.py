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
from models.generate import generate_with_compacted_cache_batch

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


def append_uncompacted_to_compacted(compacted_cache, uncompacted_tail):
    if uncompacted_tail[0][0].shape[2] == 0:
        return compacted_cache
    composed = []
    for (C1, beta, C2), (K, V) in zip(compacted_cache, uncompacted_tail):
        C1_new = torch.cat([C1, K], dim=2)
        beta_tail = torch.zeros(beta.shape[0], beta.shape[1], K.shape[2], device=beta.device, dtype=beta.dtype)
        beta_new = torch.cat([beta, beta_tail], dim=2)
        C2_new = torch.cat([C2, V], dim=2)
        composed.append((C1_new, beta_new, C2_new))
    return tuple(composed)


def concat_uncompacted_caches(cache1, cache2):
    if cache1[0][0].shape[2] == 0: return cache2
    if cache2[0][0].shape[2] == 0: return cache1
    composed = []
    for (K1, V1), (K2, V2) in zip(cache1, cache2):
        composed.append((torch.cat([K1, K2], dim=2), torch.cat([V1, V2], dim=2)))
    return tuple(composed)


def construct_combined_cache_tuple(compacted_prefix, uncompacted_suffix):
    if uncompacted_suffix[0][0].shape[2] == 0:
        return compacted_prefix
        
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
    composed = []
    for layer_a, layer_b in zip(cache_a, cache_b):
        C1_A, beta_A, C2_A = layer_a
        C1_B, beta_B, C2_B = layer_b
        C1 = torch.cat([C1_A, C1_B], dim=2)
        beta = torch.cat([beta_A, beta_B], dim=2)
        C2 = torch.cat([C2_A, C2_B], dim=2)
        composed.append((C1, beta, C2))
    return tuple(composed)


def construct_combined_cache(compacted_a, past_kv_b):
    """Combine compacted A and full B into a single CompactedPrefixCache."""
    num_layers = len(compacted_a)
    combined_cache_tuples = []
    for l in range(num_layers):
        C1_A, beta_A, C2_A = compacted_a[l]
        K_B, V_B = past_kv_b[l]
        
        # Merge tensors
        # K_B is (batch, heads, seq, dim) or (heads, seq, dim)
        seq_len_b = K_B.shape[-2]
        C1 = torch.cat([C1_A, K_B], dim=-2)
        
        # beta_B should be zeros of shape (batch, heads, seq_len_b)
        beta_shape = list(beta_A.shape)
        beta_shape[-1] = seq_len_b
        beta_B = torch.zeros(beta_shape, device=beta_A.device, dtype=beta_A.dtype)
        
        beta = torch.cat([beta_A, beta_B], dim=-1)
        C2 = torch.cat([C2_A, V_B], dim=-2)
        combined_cache_tuples.append((C1, beta, C2))
        
    return CompactedPrefixCache(compacted_cache=tuple(combined_cache_tuples))


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


def run_full_eval(cache, label_prefix, seq_len, model, tokenizer, args):
    summary = {}
    half = args.num_authors // 2
    print(f"\nEvaluating {label_prefix}...")
    # Author A
    summary[f"{label_prefix} (Author A)"] = evaluate_compacted_cache_on_dataset(
        cache, model, tokenizer, seq_len, dataset_type="tofu", num_authors=half, author_offset=0, label=f"{label_prefix}-A", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    # Author B
    summary[f"{label_prefix} (Author B)"] = evaluate_compacted_cache_on_dataset(
        cache, model, tokenizer, seq_len, dataset_type="tofu", num_authors=args.num_authors - half, author_offset=half, label=f"{label_prefix}-B", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    # Full (Author A + B)
    summary[f"{label_prefix} (Full)"] = evaluate_compacted_cache_on_dataset(
        cache, model, tokenizer, seq_len, dataset_type="tofu", num_authors=args.num_authors, author_offset=0, label=f"{label_prefix}-Full", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    # Joint Composition
    summary[f"{label_prefix} (Joint)"] = evaluate_compacted_cache_on_dataset(
        cache, model, tokenizer, seq_len, dataset_type="composition", label=f"{label_prefix}-joint", batch_size=args.batch_size, temperature=0.0, seed=args.seed, save_json=args.save_json, output_dir=args.output_dir
    )
    return summary


def print_summary(summary: Dict[str, pd.DataFrame], num_authors, num_tokens, model_name):
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
    print("  SEQUENTIAL COMPACTION COMPARISON SUMMARY")
    print(f"  {num_authors} authors | R={num_tokens} tokens | model={model_name}")
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
    parser = argparse.ArgumentParser(description="Sequential KV Cache Compaction on TOFU Dataset")
    parser.add_argument("--num-authors", type=int, default=2, help="Total number of authors in the experiment")
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query-config", type=str, default="ss-plus-repeat", help="Query config file name")
    parser.add_argument("--corpus-format", type=str, default="biography", choices=["answers", "qa", "biography"])
    parser.add_argument("--guided-ss", action="store_true", help="Compare Guided vs Independent self-study query generation")
    parser.add_argument("--uncompacted-tail-len", type=int, default=20, help="Number of tokens to leave uncompacted at the tail of each block for continuous context bridging.")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE correction when composing caches (naive concatenation, matching tofu_compaction_modular behavior).")
    parser.add_argument("--vllm", action="store_true", help="Initialize vLLM model for self-study generation")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.4, help="GPU memory utilization for vLLM")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096, help="Max model length for vLLM")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generating answers")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results_compaction_sequential/")
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
    half = args.num_authors // 2
    
    biography_index = _load_biography_index(_BIOGRAPHY_JSON) if args.corpus_format == "biography" else {}
    a_corpus = build_corpus(all_authors[:half], args.corpus_format, biography_index)
    b_corpus = build_corpus(all_authors[half:], args.corpus_format, biography_index)
    
    query_config = load_query_config(args.query_config)
    
    # Validation: Self-study requires vLLM
    if vllm_model is None and any(spec.get("method") == "self_study" for spec in query_config.get("specs", [])):
        print("\n[WARNING] Your query config contains 'self_study' specs but --vllm was not provided.")
        print("          Defaulting to 'repeat' queries only. To use self-study, please add the --vllm flag.")
        # Fallback to a simple repeat config if vllm is missing
        query_config = {"specs": [{"method": "repeat"}]}

    # Include k_medoids_iters here to feed to our new __init__
    algorithm_kwargs = {
        "score_method": "rms",
        "nnls_iters": 2,
        "c2_method": "lsq",
        "nnls_lower_bound": 0.05,
        "nnls_upper_bound": 20.0,
        "k_medoids_iters": 0, 
    }
    
    # Compact Author A
    print("\n=== [Phase 1] Compacting Author A ===")
    seq_len_a, past_kv_a, idx_a, fmt_ctx_a, _ = extract_full_kv_cache(model, tokenizer, a_corpus, args.device, model_name=model_name)
    
    N = args.uncompacted_tail_len
    num_tokens_a = args.num_tokens // 2
    num_tokens_b = args.num_tokens - num_tokens_a
    
    if N > 0:
        A_to_compact = tuple((k[:, :, :-N, :], v[:, :, :-N, :]) for k, v in past_kv_a)
        A_tail = tuple((k[:, :, -N:, :], v[:, :, -N:, :]) for k, v in past_kv_a)
    else:
        A_to_compact = past_kv_a
        A_tail = tuple((torch.empty(k.shape[0], k.shape[1], 0, k.shape[3], device=k.device, dtype=k.dtype),
                        torch.empty(v.shape[0], v.shape[1], 0, v.shape[3], device=v.device, dtype=v.dtype)) 
                       for k, v in past_kv_a)

    generator = QueryGenerator(model=model, tokenizer=tokenizer, config=query_config, device=args.device, dtype=torch.float16, vllm_model=vllm_model)
    queries_a, _, _ = generator.generate_queries(formatted_context=fmt_ctx_a, past_key_values=past_kv_a, indices=idx_a)
    
    compacted_a = run_compaction(model, tokenizer, queries_a, A_to_compact, num_tokens_a, algorithm_kwargs=algorithm_kwargs, uncompacted_tail=A_tail)
    state_a = append_uncompacted_to_compacted(compacted_a, A_tail)
    
    # Extract Full KV for Author B
    seq_len_b, past_kv_b, idx_b, fmt_ctx_b, _ = extract_full_kv_cache(model, tokenizer, b_corpus, args.device, model_name=model_name)
    
    # Globally shift B upfront so it seamlessly attaches to A
    if args.no_rope:
        print("  [--no-rope] Skipping RoPE shift on B — using naive composition.")
        past_kv_b_shifted = past_kv_b
    else:
        past_kv_b_shifted = shift_uncompacted_cache(past_kv_b, seq_len_a, model)
    
    combined_cache_tuple = construct_combined_cache_tuple(state_a, past_kv_b_shifted)

    # Intermediate evaluation
    summary = {}

    print("\nEvaluating Hybrid Cache (Compacted A + Full B)...")
    summary.update(run_full_eval(combined_cache_tuple, "Compact A Full B", seq_len_a + seq_len_b, model, tokenizer, args))

    # Query Generation for B
    print("Generating Independent Queries...")
    queries_ind, _, _ = generator.generate_queries(formatted_context=fmt_ctx_b, past_key_values=past_kv_b, indices=idx_b)
    
    if args.guided_ss:
        print("Generating Guided Queries...")

        fmt_ctx_prefix = tokenizer.apply_chat_template([{"role": "user", "content": a_corpus}], tokenize=False, add_generation_prompt=False)
        combined_fmt_ctx = fmt_ctx_prefix + fmt_ctx_b

        idx_b_combined = range(num_tokens_a, num_tokens_a + seq_len_b)
        combined_cache = CompactedPrefixCache(compacted_cache=combined_cache_tuple, original_seq_len=seq_len_a + seq_len_b)
        queries_guided, _, _ = generator.generate_queries(formatted_context=combined_fmt_ctx, past_key_values=combined_cache, indices=idx_b_combined)
    else:
        queries_guided = None

    # Slice B
    if N > 0:
        B_body = tuple((k[:, :, :-N, :], v[:, :, :-N, :]) for k, v in past_kv_b_shifted)
        B_tail = tuple((k[:, :, -N:, :], v[:, :, -N:, :]) for k, v in past_kv_b_shifted)
    else:
        B_body = past_kv_b_shifted
        B_tail = tuple((torch.empty(k.shape[0], k.shape[1], 0, k.shape[3], device=k.device, dtype=k.dtype),
                        torch.empty(v.shape[0], v.shape[1], 0, v.shape[3], device=v.device, dtype=v.dtype)) 
                       for k, v in past_kv_b_shifted)

    U_for_b = concat_uncompacted_caches(A_tail, B_body)

    print("\nRunning Sequential Compaction with Independent Queries...")
    comp_b_ind = run_compaction(model, tokenizer, queries_ind, U_for_b, num_tokens_b, compacted_past=compacted_a, algorithm_kwargs=algorithm_kwargs, uncompacted_tail=B_tail)
    
    cache_ind_base = compose_compacted_caches(compacted_a, comp_b_ind)
    cache_ind = append_uncompacted_to_compacted(cache_ind_base, B_tail)
    summary.update(run_full_eval(cache_ind, "Ind. Queries", seq_len_a + seq_len_b, model, tokenizer, args))

    if args.guided_ss:
        print("\nRunning Sequential Compaction with Guided Queries...")
        comp_b_guided = run_compaction(model, tokenizer, queries_guided, U_for_b, num_tokens_b, compacted_past=compacted_a, algorithm_kwargs=algorithm_kwargs, uncompacted_tail=B_tail)
        
        cache_guided_base = compose_compacted_caches(compacted_a, comp_b_guided)
        cache_guided = append_uncompacted_to_compacted(cache_guided_base, B_tail)
        summary.update(run_full_eval(cache_guided, "Guided Queries", seq_len_a + seq_len_b, model, tokenizer, args))

    # Print Summary
    print_summary(summary, 2, args.num_tokens, model_name)

if __name__ == "__main__":
    main()