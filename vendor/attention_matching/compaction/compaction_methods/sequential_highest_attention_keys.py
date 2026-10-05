# compaction/compaction_methods/sequential_highest_attention_keys.py
"""
Sequential compaction using the objective from tofu_compaction_sequential.

This method treats the cache as:
- P: preserved prefix (already compacted)
- U: current tokens to compact
- S: protected suffix (uncompacted tail)

If no prefix is excluded (compact range starts at 0), P is empty and
U includes all compactable tokens (recompact_all).
"""
from typing import Tuple, Dict, Optional, Any
import time
import torch

from .base import FullCacheCompactionAlgorithm
from ..algorithms.highest_attention_keys import HighestAttentionKeysCompaction
from ..query_generation import QueryConfig
from ..query_generation import QueryGenerator
from models.cache import CompactedPrefixCache, CompactedPrefixLayer, DynamicSlidingWindowLayer


class SequentialCompactionAlgorithm(HighestAttentionKeysCompaction):
    """Sequential objective for compacting U given P and S."""

    def compute_compacted_cache_sequential(
        self,
        U_K: torch.Tensor,
        U_V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        P_K: Optional[torch.Tensor] = None,
        P_beta: Optional[torch.Tensor] = None,
        P_V: Optional[torch.Tensor] = None,
        attention_bias_u: Optional[torch.Tensor] = None,
        S_K: Optional[torch.Tensor] = None,
        S_V: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        n, d = queries.shape
        device = U_K.device
        dtype = U_K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5

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

        lse_list = [M_U]
        if has_P:
            lse_list.append(M_P)
        if has_S:
            lse_list.append(M_S)
        global_lse = torch.logsumexp(torch.stack(lse_list, dim=0), dim=0)

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

        U_attn_weights = torch.exp(U_scores32 - global_lse.unsqueeze(1))
        if self.score_method == 'rms':
            key_scores = torch.sqrt((U_attn_weights ** 2).mean(dim=0))
        elif self.score_method == 'max':
            key_scores = U_attn_weights.max(dim=0)[0]
        else:
            key_scores = U_attn_weights.mean(dim=0)

        _, top_indices = torch.topk(key_scores, t, largest=True)
        C1 = U_K[top_indices]

        C1_scores32 = (queries @ C1.T.to(queries.dtype)).to(torch.float32) * inv_sqrt_d
        A_tilde = torch.exp(C1_scores32 - global_lse.unsqueeze(1))
        m_tilde = weight_U_true
        w = self._nnls_pg(A_tilde, m_tilde, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
        beta = torch.log(w.clamp(min=1e-12)).to(dtype)

        C1_beta_scores = C1_scores32 + beta.to(torch.float32)
        C1_beta_lse = torch.logsumexp(C1_beta_scores, dim=-1)

        compact_lse_list = [C1_beta_lse]
        if has_P:
            compact_lse_list.append(M_P)
        if has_S:
            compact_lse_list.append(M_S)
        compact_lse = torch.logsumexp(torch.stack(compact_lse_list, dim=0), dim=0)

        weight_P_compact = torch.exp(M_P - compact_lse).unsqueeze(1) if has_P else 0.0
        weight_S_compact = torch.exp(M_S - compact_lse).unsqueeze(1) if has_S else 0.0
        X = torch.exp(C1_beta_scores - compact_lse.unsqueeze(1))
        Z = Y - weight_P_compact * V_P - weight_S_compact * V_S
        C2 = torch.linalg.lstsq(X, Z).solution.to(dtype)

        return C1, beta, C2, top_indices.tolist()


class SequentialHighestAttentionKeysCompaction(FullCacheCompactionAlgorithm):
    """Full-cache sequential compaction using HighestAttentionKeys selection."""

    def __init__(
        self,
        score_method: str = 'rms',
        nnls_iters: int = 2,
        nnls_lower_bound: float = 0.05,
        nnls_upper_bound: float = 20.0,
        c2_method: str = 'lsq',
        config_name: Optional[str] = None,
        precomputed_budget_path: Optional[str] = None,
        max_ratio_per_head: float = 1.0,
        use_batched: bool = False,
        on_policy: bool = False,
        **_unused: Any,
    ):
        _ = (precomputed_budget_path, max_ratio_per_head, use_batched, on_policy, _unused)
        self.algorithm_kwargs = {
            'score_method': score_method,
            'nnls_iters': nnls_iters,
            'nnls_lower_bound': nnls_lower_bound,
            'nnls_upper_bound': nnls_upper_bound,
            'c2_method': c2_method,
        }
        self._name_instance = SequentialCompactionAlgorithm(**self.algorithm_kwargs)
        self.config_name = config_name

    def name(self) -> str:
        if self.config_name:
            return self.config_name
        return f"sequential_{self._name_instance.name()}"

    @staticmethod
    def _split_layer_cache(layer_tuple):
        if not isinstance(layer_tuple, (list, tuple)):
            raise TypeError(f"Expected layer cache to be tuple/list, got {type(layer_tuple)}")
        if len(layer_tuple) == 2:
            keys, values = layer_tuple
            bias = None
        elif len(layer_tuple) == 3:
            keys, bias, values = layer_tuple
        else:
            raise ValueError(f"Layer cache must have 2 or 3 elements (K,[bias],V); got {len(layer_tuple)}")
        return keys, bias, values

    @staticmethod
    def _compacted_prefix_to_tuple(cache: CompactedPrefixCache):
        layers = []
        for layer in cache.layers:
            if isinstance(layer, CompactedPrefixLayer):
                layers.append((layer.keys, layer.beta, layer.values))
            elif isinstance(layer, DynamicSlidingWindowLayer):
                keys = layer.keys
                values = layer.values
                beta = torch.zeros(
                    keys.shape[0],
                    keys.shape[1],
                    keys.shape[2],
                    device=keys.device,
                    dtype=keys.dtype,
                )
                layers.append((keys, beta, values))
            else:
                raise TypeError(f"Unsupported layer type in CompactedPrefixCache: {type(layer)}")
        return tuple(layers)

    def compact_kv_cache(
        self,
        past_key_values: Tuple,
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        compute_stats: bool = False,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        sliding_layer_indices: Optional[set] = None,
        past_key_values_for_queries: Optional[Any] = None,
        full_query_extraction: bool = False,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        kv_for_queries = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        if isinstance(past_key_values, CompactedPrefixCache):
            if sliding_layer_indices is None:
                sliding_layer_indices = getattr(past_key_values, "_sliding_layer_indices", None)
            past_key_values = self._compacted_prefix_to_tuple(past_key_values)

        sliding_layer_indices = sliding_layer_indices or set()
        num_layers = len(past_key_values)

        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        if batch_size != 1:
            raise NotImplementedError("Sequential compaction currently only supports batch_size=1")

        generator = QueryGenerator(
            model=model,
            tokenizer=tokenizer,
            config=query_config,
            device=past_key_values[ref_layer_idx][0].device,
            dtype=past_key_values[ref_layer_idx][0].dtype,
            vllm_model=vllm_model,
        )

        query_indices = indices
        if full_query_extraction:
            full_len = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
            query_indices = range(0, full_len)
        base_queries, query_stats, _ = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=kv_for_queries,
            indices=query_indices,
        )
        queries = base_queries.unsqueeze(0)

        print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")
        for method, method_stats in query_stats.get('methods_used', {}).items():
            print(f"  {method}: {method_stats['n_queries_actual_per_kv_head']} queries ({method_stats['fraction']:.1%})")
        print("Train queries shape:", queries.shape)

        if indices is not None:
            indices_list = list(indices)
            num_to_compact = len(indices_list)
            num_to_keep = seq_len - num_to_compact
            sub_target_size = target_size - num_to_keep
            if sub_target_size <= 0:
                raise ValueError(
                    f"target_size ({target_size}) must be greater than the number of "
                    f"positions to keep ({num_to_keep}). Got sub_target_size = {sub_target_size}"
                )

            all_indices = torch.arange(seq_len)
            compact_mask = torch.zeros(seq_len, dtype=torch.bool)
            compact_mask[indices_list] = True
            keep_indices = all_indices[~compact_mask].tolist()

            actual_target_size = sub_target_size
            is_partial_compaction = True
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size
            is_partial_compaction = False

        compacted_layers = []
        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'train_stats_time': 0.0,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_layers - len(sliding_layer_indices),
        }

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }
            all_stats['keep_indices'] = {
                'num_positions': len(keep_indices),
            }

        seq_algo = SequentialCompactionAlgorithm(**self.algorithm_kwargs)
        total_effective_article_tokens = 0

        for layer_idx in range(num_layers):
            keys_layer, bias_layer, values_layer = self._split_layer_cache(past_key_values[layer_idx])
            if bias_layer is not None and bias_layer.dim() == 2:
                bias_layer = bias_layer.unsqueeze(0)

            if layer_idx in sliding_layer_indices:
                print(f"Layer {layer_idx+1}/{num_layers}: sliding window (keeping original KV)")
                placeholder_C1 = keys_layer.new_zeros(1, num_heads, 0, head_dim)
                placeholder_beta = keys_layer.new_zeros(1, num_heads, 0)
                placeholder_C2 = values_layer.new_zeros(1, num_heads, 0, head_dim)
                compacted_layers.append((placeholder_C1, placeholder_beta, placeholder_C2))
                continue

            print(f"Compacting layer {layer_idx+1}/{num_layers}")

            C1_heads = []
            beta_heads = []
            C2_heads = []

            for head_idx in range(num_heads):
                K_full = keys_layer[0, head_idx, :, :]
                V_full = values_layer[0, head_idx, :, :]
                bias_full = bias_layer[0, head_idx, :] if bias_layer is not None else None

                if is_partial_compaction:
                    K_u = K_full[indices_list, :]
                    V_u = V_full[indices_list, :]
                    attn_bias_u = bias_full[indices_list] if bias_full is not None else None

                    K_keep = K_full[keep_indices, :]
                    V_keep = V_full[keep_indices, :]
                    bias_keep = bias_full[keep_indices] if bias_full is not None else None

                    compact_start = indices_list[0]
                    compact_end = indices_list[-1] + 1

                    keep_before_mask = [idx < compact_start for idx in keep_indices]
                    keep_after_mask = [idx >= compact_end for idx in keep_indices]

                    K_keep_before = K_keep[keep_before_mask, :] if any(keep_before_mask) else K_full.new_zeros(0, head_dim)
                    V_keep_before = V_keep[keep_before_mask, :] if any(keep_before_mask) else V_full.new_zeros(0, head_dim)
                    K_keep_after = K_keep[keep_after_mask, :] if any(keep_after_mask) else K_full.new_zeros(0, head_dim)
                    V_keep_after = V_keep[keep_after_mask, :] if any(keep_after_mask) else V_full.new_zeros(0, head_dim)

                    if bias_keep is not None:
                        bias_keep_before = bias_keep[keep_before_mask]
                        bias_keep_after = bias_keep[keep_after_mask]
                    else:
                        bias_keep_before = K_full.new_zeros(K_keep_before.shape[0])
                        bias_keep_after = K_full.new_zeros(K_keep_after.shape[0])

                    P_K = K_keep_before
                    P_beta = bias_keep_before
                    P_V = V_keep_before
                    S_K = K_keep_after
                    S_V = V_keep_after
                else:
                    K_u = K_full
                    V_u = V_full
                    attn_bias_u = bias_full
                    P_K = P_beta = P_V = None
                    S_K = S_V = None
                    bias_keep_before = None
                    bias_keep_after = None

                queries_head = queries[0, layer_idx, head_idx, :, :]
                head_target_size = actual_target_size

                if head_target_size == 0:
                    C1_compact = K_u.new_zeros(0, head_dim)
                    beta_compact = K_u.new_zeros(0)
                    C2_compact = V_u.new_zeros(0, head_dim)
                    selected_indices = []
                else:
                    C1_compact, beta_compact, C2_compact, selected_indices = seq_algo.compute_compacted_cache_sequential(
                        K_u,
                        V_u,
                        queries_head,
                        head_target_size,
                        P_K=P_K,
                        P_beta=P_beta,
                        P_V=P_V,
                        attention_bias_u=attn_bias_u,
                        S_K=S_K,
                        S_V=S_V,
                    )

                actual_returned_size = C1_compact.shape[0]
                total_effective_article_tokens += actual_returned_size
                beta_for_stats = beta_compact
                if actual_returned_size < head_target_size:
                    num_padding = head_target_size - actual_returned_size
                    C1_padding = K_u.new_zeros(num_padding, head_dim)
                    C2_padding = V_u.new_zeros(num_padding, head_dim)
                    beta_padding = K_u.new_full((num_padding,), float('-inf'))

                    C1_compact = torch.cat([C1_compact, C1_padding], dim=0)
                    beta_compact = torch.cat([beta_compact, beta_padding], dim=0)
                    C2_compact = torch.cat([C2_compact, C2_padding], dim=0)

                if is_partial_compaction:
                    beta_keep_before = bias_keep_before if bias_keep_before is not None else K_full.new_zeros(K_keep_before.shape[0])
                    beta_keep_after = bias_keep_after if bias_keep_after is not None else K_full.new_zeros(K_keep_after.shape[0])

                    C1 = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=0)
                    beta = torch.cat([beta_keep_before, beta_compact, beta_keep_after], dim=0)
                    C2 = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=0)
                else:
                    C1 = C1_compact
                    beta = beta_compact
                    C2 = C2_compact

                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))

                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_for_stats.min().item()) if len(beta_for_stats) > 0 else None,
                        'max': float(beta_for_stats.max().item()) if len(beta_for_stats) > 0 else None,
                        'mean': float(beta_for_stats.mean().item()) if len(beta_for_stats) > 0 else None,
                        'std': float(beta_for_stats.std().item()) if len(beta_for_stats) > 1 else None,
                        'num_less_than_minus_7': int((beta_for_stats < -7).sum().item()) if len(beta_for_stats) > 0 else 0,
                    }} if verbose_logging else {})
                }

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

            target_seq_len = max(h.shape[2] for h in C1_heads)
            for i in range(len(C1_heads)):
                curr_len = C1_heads[i].shape[2]
                if curr_len < target_seq_len:
                    pad_len = target_seq_len - curr_len
                    C1_heads[i] = torch.cat([
                        C1_heads[i],
                        C1_heads[i].new_zeros(1, 1, pad_len, head_dim)
                    ], dim=2)
                    C2_heads[i] = torch.cat([
                        C2_heads[i],
                        C2_heads[i].new_zeros(1, 1, pad_len, head_dim)
                    ], dim=2)
                    beta_heads[i] = torch.cat([
                        beta_heads[i],
                        beta_heads[i].new_full((1, 1, pad_len), float('-inf'))
                    ], dim=2)

            C1_layer = torch.cat(C1_heads, dim=1)
            beta_layer = torch.cat(beta_heads, dim=1)
            C2_layer = torch.cat(C2_heads, dim=1)
            compacted_layers.append((C1_layer, beta_layer, C2_layer))

        total_tensor_len = 0
        num_global_layers = num_layers - len(sliding_layer_indices)
        for layer_idx, layer_data in enumerate(compacted_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += layer_data[0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            num_kept = len(keep_indices) if keep_indices is not None else 0
            avg_tensor_compacted_len = actual_target_size + num_kept
        all_stats['tensor_compacted_seq_len'] = avg_tensor_compacted_len

        total_global_heads = num_global_layers * num_heads
        if total_global_heads > 0:
            effective_article_tokens = total_effective_article_tokens / total_global_heads
        else:
            effective_article_tokens = 0
        num_kept = len(keep_indices) if keep_indices is not None else 0
        effective_compacted_seq_len = effective_article_tokens + num_kept
        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats['effective_article_tokens'] = effective_article_tokens
        all_stats['tensor_article_tokens'] = tensor_article_tokens
        all_stats['effective_compacted_seq_len'] = effective_compacted_seq_len
        all_stats['query_generation'] = query_stats

        return tuple(compacted_layers), all_stats
