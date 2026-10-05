# compaction/query_generation/context_prefill.py
"""Context prefill query generation for KV cache compaction."""

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer

from evaluation.utils import detect_user_tags
from .config import ContextPrefillConfig


class ContextPrefillQueryGenerator:
    """
    Generate queries by extracting from the article portion of formatted_context.

    This class implements the simplest query generation method: extracting query
    vectors directly from the article tokens themselves by running a single prefill pass.
    The article "studies itself".
    """

    def __init__(
        self,
        model,  # The model instance (e.g., Qwen3ForCausalLM)
        tokenizer: AutoTokenizer,
        config: ContextPrefillConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the context prefill query generator.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The language model
        tokenizer : AutoTokenizer
            Tokenizer for the model
        config : ContextPrefillConfig
            Configuration for context prefill query generation
        device : str, optional
            Device to use. If None, uses model's device.
        dtype : torch.dtype, optional
            Data type for queries. If None, uses model's dtype.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

    def _extract_article_token_indices(self, formatted_context: str) -> range:
        """
        Extract the token indices corresponding to the article portion of formatted context.

        Parameters
        ----------
        formatted_context : str
            The formatted context string with chat template applied

        Returns
        -------
        range
            Token indices corresponding to the article portion
        """
        user_start_tag, user_end_tag = detect_user_tags(formatted_context)

        user_start_pos = formatted_context.find(user_start_tag)
        if user_start_pos == -1:
            raise ValueError(f"Could not find '{user_start_tag}' tag in formatted context")

        # The article starts after the user tag and a newline
        article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
        if article_text_start == -1:
            raise ValueError("Could not find newline after user start tag")
        article_text_start += 1  # Skip the newline itself

        # Find the end tag after the article
        article_text_end = formatted_context.find(user_end_tag, article_text_start)
        if article_text_end == -1:
            raise ValueError(f"Could not find '{user_end_tag}' tag after article content")

        # Tokenize the prefix (everything before article)
        prefix = formatted_context[:article_text_start]
        prefix_tokens = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        article_start_idx = prefix_tokens.input_ids.shape[1]

        # Tokenize up to and including the article
        context_up_to_article_end = formatted_context[:article_text_end]
        tokens_up_to_end = self.tokenizer(context_up_to_article_end, return_tensors="pt", add_special_tokens=False)
        article_end_idx = tokens_up_to_end.input_ids.shape[1]

        return range(article_start_idx, article_end_idx)

    def generate_queries(
        self,
        n_queries_per_attention_head: int,
        formatted_context: str,
        past_key_values: Optional[Tuple] = None,
        indices: Optional[range] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate context prefill queries (public interface for QueryGenerator).

        Parameters
        ----------
        n_queries_per_attention_head : int
            Number of queries to generate per attention head
        formatted_context : str
            The formatted context string
        past_key_values : tuple, optional
            Pre-computed KV cache for the context (not used in this mode)

        Returns
        -------
        queries : torch.Tensor
            Context prefill queries of shape (num_layers, num_attention_heads, n_tokens, head_dim)
        stats : dict
            Statistics about generation
        """
        # Get head_dim from model config
        head_dim = getattr(self.model.config, 'head_dim',
                           self.model.config.hidden_size // self.model.config.num_attention_heads)

        # Determine which token indices to extract queries from
        # If indices is provided (e.g., reasoning tokens), use that directly
        # Otherwise, fall back to detecting article boundaries from formatted_context
        if indices is not None:
            article_indices = indices
        else:
            article_indices = self._extract_article_token_indices(formatted_context)

        # Tokenize the full context
        # Use add_special_tokens=False since formatted_context already has <bos> from chat template
        context_inputs = self.tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False).to(self.device)

        # Select the token positions before the forward pass. The previous
        # implementation captured every query vector and subsampled afterward,
        # which retained several GiB for a 16K context even when only a handful
        # of training queries were requested.
        article_start = max(0, article_indices.start)
        article_stop = min(context_inputs.input_ids.shape[1], article_indices.stop)
        n_available = max(0, article_stop - article_start)
        if n_available > n_queries_per_attention_head:
            sampled_offsets = torch.randperm(
                n_available, device=context_inputs.input_ids.device
            )[:n_queries_per_attention_head]
            sampled_offsets = sampled_offsets.sort()[0]
        else:
            sampled_offsets = torch.arange(
                n_available, device=context_inputs.input_ids.device
            )
        selected_token_indices = sampled_offsets + article_start

        # Extract only the selected query vectors using a single prefill.
        query_vectors = self._extract_query_vectors_from_prefill(
            input_ids=context_inputs.input_ids,
            head_dim=head_dim,
            selected_token_indices=selected_token_indices,
        )

        stats = {
            'mode': 'context_prefill',
            'article_token_range': f'{article_indices.start}-{article_indices.stop}',
        }

        if query_vectors is None:
            # Return empty tensor
            num_layers = self.model.config.num_hidden_layers
            num_attention_heads = self.model.config.num_attention_heads
            empty_queries = torch.zeros((num_layers, num_attention_heads, 0, head_dim),
                                        device=self.device, dtype=self.dtype)
            stats['n_context_prefill_tokens_extracted'] = 0
            return empty_queries, stats

        stats['n_context_prefill_tokens_extracted'] = n_available
        if n_available > n_queries_per_attention_head:
            stats['n_context_prefill_tokens_subsampled'] = n_queries_per_attention_head
            stats['subsample_indices'] = sampled_offsets.detach().cpu().tolist()

        return query_vectors, stats

    def _extract_query_vectors_from_prefill(
        self,
        input_ids: torch.Tensor,
        head_dim: int,
        selected_token_indices: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Extract query vectors from pre-generated tokens using a single prefill pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs for the entire sequence
            Shape: (batch_size, seq_len)
        head_dim : int
            Dimension of each attention head
        selected_token_indices : torch.Tensor, optional
            Exact sequence positions whose query vectors should be retained.

        Returns
        -------
        query_vectors : torch.Tensor or None
            Query vectors of shape (num_layers, num_attention_heads, n_tokens, head_dim)
            where n_tokens is the number of selected positions.
        """
        try:
            num_layers = self.model.config.num_hidden_layers
            num_attention_heads = self.model.config.num_attention_heads

            # Storage for queries from all layers
            all_layer_queries = []

            # Hook function to capture only the preselected queries at each layer.
            def make_query_hook(layer_idx):
                def hook_fn(module, args, kwargs):
                    # Extract hidden states
                    hidden_states = None
                    if len(args) > 0:
                        hidden_states = args[0]
                    elif 'hidden_states' in kwargs:
                        hidden_states = kwargs['hidden_states']

                    if hidden_states is not None:
                        selected_hidden_states = hidden_states
                        if selected_token_indices is not None:
                            selected_hidden_states = hidden_states.index_select(
                                1, selected_token_indices
                            )
                        # Get position_embeddings from kwargs (passed by the model)
                        position_embeddings = kwargs.get('position_embeddings')

                        # Apply query projection: Q = hidden_states @ W_q
                        q = module.q_proj(selected_hidden_states)  # (batch, selected_len, num_heads * head_dim)

                        batch_size, seq_len, _ = q.shape
                        # Reshape to (batch, seq_len, num_attention_heads, head_dim)
                        q = q.view(batch_size, seq_len, num_attention_heads, head_dim)

                        # Apply q_norm
                        from models.qwen3.modeling_qwen3 import Qwen3Attention
                        from models.gemma3.modeling_gemma3 import Gemma3Attention
                        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
                            q = module.q_norm(q)

                        # Transpose to (batch, num_heads, seq_len, head_dim)
                        q = q.transpose(1, 2)

                        # Apply RoPE to queries if we have position embeddings
                        if position_embeddings is not None:
                            from models.qwen3.modeling_qwen3 import rotate_half
                            cos, sin = position_embeddings
                            if selected_token_indices is not None:
                                cos = cos.index_select(-2, selected_token_indices)
                                sin = sin.index_select(-2, selected_token_indices)
                            cos = cos.unsqueeze(1)  # Add head dimension
                            sin = sin.unsqueeze(1)
                            q = (q * cos) + (rotate_half(q) * sin)

                        # Transpose back to (batch, seq_len, num_heads, head_dim)
                        q = q.transpose(1, 2)

                        # q: (batch, selected_len, num_attention_heads, head_dim)
                        all_layer_queries.append(q[0])

                return hook_fn

            # Register hooks on all layers
            hooks = []
            try:
                for layer_idx in range(num_layers):
                    target_layer = self.model.model.layers[layer_idx].self_attn
                    handle = target_layer.register_forward_pre_hook(
                        make_query_hook(layer_idx),
                        with_kwargs=True
                    )
                    hooks.append(handle)

                # A full-context prefill is still required to produce the correct
                # hidden states, but only selected queries and one logit are retained.
                with torch.no_grad():
                    self.model(
                        input_ids=input_ids,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )

                # Process the captured queries
                # all_layer_queries: one (selected_len, heads, head_dim) tensor per layer
                if all_layer_queries:
                    # Stack to: (num_layers, seq_len, num_attention_heads, head_dim)
                    queries = torch.stack(all_layer_queries, dim=0)
                    # Transpose to: (num_layers, num_attention_heads, seq_len, head_dim)
                    queries = queries.permute(0, 2, 1, 3)

                    return queries
                else:
                    return None

            finally:
                # Always remove hooks, even if an error occurred
                for handle in hooks:
                    handle.remove()

        except Exception as e:
            raise RuntimeError(
                f"Failed to extract query vectors in context prefill mode: {e}"
            ) from e
