from __future__ import annotations

import importlib
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "attention_matching"
_BNB_INT8_CAST_LOGGER = "bitsandbytes.autograd._functions"
_BNB_INT8_CAST_MESSAGE = (
    "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during "
    "quantization"
)


class _ExpectedBnbInt8CastFilter(logging.Filter):
    """Drop the per-matmul BF16-to-FP16 notice without hiding other warnings."""

    installed_by_compaction_limit = True

    def filter(self, record: logging.LogRecord) -> bool:
        return _BNB_INT8_CAST_MESSAGE not in record.getMessage()


def _suppress_repeated_bnb_int8_cast_warning() -> None:
    logger = logging.getLogger(_BNB_INT8_CAST_LOGGER)
    if not any(
        getattr(existing, "installed_by_compaction_limit", False)
        for existing in logger.filters
    ):
        logger.addFilter(_ExpectedBnbInt8CastFilter())


def _activate_vendor() -> None:
    if not VENDOR_ROOT.exists():
        raise FileNotFoundError(
            f"Vendored Attention Matching implementation not found at {VENDOR_ROOT}"
        )
    vendor_path = str(VENDOR_ROOT)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    # The reference package has an import-order dependency: loading its
    # compaction registry before evaluation.utils creates a circular import.
    # Its own examples import evaluation first, so preserve that order here.
    importlib.import_module("evaluation.utils")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    weight_precision: str,
) -> tuple[Any, Any, int]:
    """Load the custom Qwen3 model required by the AM cache implementation."""

    if not device.startswith("cuda"):
        raise ValueError(
            "The Attention Matching runner currently requires a CUDA device."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use `compaction-limit plan` on CPU.")
    if "qwen3" not in model_name.lower():
        raise ValueError("This pilot supports Qwen3 checkpoints only.")

    if weight_precision == "int8":
        _suppress_repeated_bnb_int8_cast_warning()

    _activate_vendor()
    from models.qwen3 import Qwen3ForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: dict[str, Any] = {
        "attn_implementation": "sdpa",
        "device_map": {"": device},
    }
    if weight_precision == "bf16":
        model_kwargs["dtype"] = torch.bfloat16
    elif weight_precision in ("int8", "nf4"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "INT8/NF4 requires the optional quantized dependencies: "
                '`uv pip install -e ".[quantized]"`.'
            ) from error
        model_kwargs["dtype"] = torch.bfloat16
        if weight_precision == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
    else:
        raise ValueError(
            f"Unknown weight precision {weight_precision!r}; choose 'bf16', "
            "'int8', or 'nf4'."
        )

    model = Qwen3ForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    footprint = int(model.get_memory_footprint())
    return model, tokenizer, footprint


def audit_model_quantization(
    model: Any,
    requested_precision: str,
    footprint_bytes: int,
) -> dict[str, Any]:
    """Verify that BitsAndBytes weights are real quantized modules.

    KV precision is a QDQ quality simulation, but model INT8/NF4 must execute
    through BitsAndBytes quantized linear modules.  The returned audit is saved
    in every manifest and repeated in every summary row.
    """

    module_names = [
        type(module).__name__ for module in getattr(model, "modules", lambda: ())()
    ]
    int8_modules = sum(name == "Linear8bitLt" for name in module_names)
    int4_modules = sum(name == "Linear4bit" for name in module_names)
    quantization_config = getattr(getattr(model, "config", None), "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    elif quantization_config is not None and not isinstance(quantization_config, dict):
        quantization_config = str(quantization_config)

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in getattr(model, "parameters", lambda: ())()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in getattr(model, "buffers", lambda: ())()
    )
    if requested_precision == "bf16":
        verified = int8_modules == 0 and int4_modules == 0
        storage_bits = 16
    elif requested_precision == "int8":
        verified = int8_modules > 0 and int4_modules == 0
        storage_bits = 8
    elif requested_precision == "nf4":
        verified = int4_modules > 0 and int8_modules == 0
        storage_bits = 4
    else:
        raise ValueError(f"Unknown requested weight precision: {requested_precision}")

    return {
        "requested_precision": requested_precision,
        "nominal_storage_bits": storage_bits,
        "quantized_linear_8bit_modules": int8_modules,
        "quantized_linear_4bit_modules": int4_modules,
        "quantized_execution_verified": verified,
        "model_get_memory_footprint_bytes": int(footprint_bytes),
        "parameter_storage_bytes": int(parameter_bytes),
        "buffer_storage_bytes": int(buffer_bytes),
        "quantization_config": quantization_config,
    }


def extract_context_cache(
    model: Any,
    tokenizer: Any,
    corpus_text: str,
    device: str,
    model_name: str,
) -> tuple[int, Any, range, str, int]:
    _activate_vendor()
    from evaluation.utils import extract_full_kv_cache

    return extract_full_kv_cache(
        model,
        tokenizer,
        corpus_text,
        device,
        model_name=model_name,
    )


def build_query_config(max_queries_per_head: int) -> Any:
    _activate_vendor()
    from compaction.query_generation.config import (
        ContextPrefillConfig,
        QueryConfig,
        QueryMethodConfig,
    )

    return QueryConfig(
        method_configs=[
            QueryMethodConfig(
                method="context_prefill",
                fraction=1.0,
                config=ContextPrefillConfig(),
            )
        ],
        max_query_vectors_per_kv_head=max_queries_per_head,
        eval_queries_per_kv_head=min(max_queries_per_head, 256),
        verbose=False,
    )


def extract_compaction_queries(
    past_key_values_for_queries: Any,
    reference_cache: tuple,
    article_indices: range,
    formatted_context: str,
    model: Any,
    tokenizer: Any,
    max_queries_per_head: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate the corpus-prefill query bank once for reuse across AM fits."""

    _activate_vendor()
    from compaction.query_generation import QueryGenerator

    query_config = build_query_config(max_queries_per_head)
    reference_keys = reference_cache[0][0]
    generator = QueryGenerator(
        model=model,
        tokenizer=tokenizer,
        config=query_config,
        device=reference_keys.device,
        dtype=reference_keys.dtype,
    )
    queries, query_stats, _ = generator.generate_queries(
        formatted_context=formatted_context,
        past_key_values=past_key_values_for_queries,
        indices=article_indices,
    )
    if queries.shape[-2] <= 0:
        raise RuntimeError(
            "Context-prefill query extraction returned zero queries; refusing "
            "to run compaction with an invalid training objective."
        )
    return queries, query_stats


def build_compaction_method(
    method: str,
    composition_mode: str = "post",
    key_bits: int = 16,
    value_bits: int = 16,
    group_size: int = 64,
    quantizer: str = "kivi_layout_symmetric",
    aware_refinement_iterations: int = 2,
) -> Any:
    _activate_vendor()
    from compaction.compaction_methods.registry import get_compaction_method

    if composition_mode not in ("post", "pre", "aware"):
        raise ValueError(f"Unknown composition mode: {composition_mode}")

    if method == "am":
        method_kwargs = {
            "algorithm": "highest_attention_keys",
            "score_method": "rms",
            "nnls_iters": 2,
            "nnls_lower_bound": math.exp(-3),
            "nnls_upper_bound": math.exp(3),
            "c2_method": "lsq",
        }
        method_name = "AM-HighestAttnKeys"
    else:
        raise ValueError(f"Unknown compaction method: {method}")
    if composition_mode == "aware":
        if method != "am":
            raise ValueError(
                "Quantization-aware compaction is implemented for AM only."
            )
        from compaction.compaction_methods.per_layer_head import PerLayerHeadCompaction

        from .quantized_am import make_quantization_aware_algorithm

        algorithm_class = make_quantization_aware_algorithm()
        aware_kwargs = {
            key: value for key, value in method_kwargs.items() if key != "algorithm"
        }
        aware_kwargs.update(
            {
                "key_bits": key_bits,
                "value_bits": value_bits,
                "group_size": group_size,
                "quantizer": quantizer,
                "refinement_iterations": aware_refinement_iterations,
            }
        )
        return PerLayerHeadCompaction(
            algorithm_class=algorithm_class,
            algorithm_kwargs=aware_kwargs,
            config_name="AM-HighestAttnKeys-quantization-aware",
        )
    return get_compaction_method(method_name, method_kwargs=method_kwargs)


def compact_cache(
    method: str,
    entry_ratio: float,
    past_key_values: Any,
    seq_len: int,
    article_indices: range,
    formatted_context: str,
    model: Any,
    tokenizer: Any,
    max_queries_per_head: int,
    save_detailed_stats: bool = True,
    composition_mode: str = "post",
    key_bits: int = 16,
    value_bits: int = 16,
    group_size: int = 64,
    quantizer: str = "kivi_layout_symmetric",
    aware_refinement_iterations: int = 2,
    past_key_values_for_queries: Any | None = None,
    precomputed_queries: torch.Tensor | None = None,
    precomputed_query_stats: dict[str, Any] | None = None,
) -> tuple[tuple, dict[str, Any], int]:
    if not 0.0 < entry_ratio < 1.0:
        raise ValueError(
            "Compaction entry_ratio must lie strictly between zero and one."
        )

    article_tokens = len(article_indices)
    target_article_tokens = max(1, round(article_tokens * entry_ratio))
    non_article_tokens = seq_len - article_tokens
    target_total_tokens = non_article_tokens + target_article_tokens
    compaction_method = build_compaction_method(
        method,
        composition_mode=composition_mode,
        key_bits=key_bits,
        value_bits=value_bits,
        group_size=group_size,
        quantizer=quantizer,
        aware_refinement_iterations=aware_refinement_iterations,
    )
    query_config = build_query_config(max_queries_per_head)
    compacted_cache, stats = compaction_method.compact_kv_cache(
        past_key_values=past_key_values,
        target_size=target_total_tokens,
        indices=article_indices,
        query_config=query_config,
        model=model,
        tokenizer=tokenizer,
        formatted_context=formatted_context,
        compute_stats=False,
        verbose_logging=save_detailed_stats,
        past_key_values_for_queries=past_key_values_for_queries,
        precomputed_queries=precomputed_queries,
        precomputed_query_stats=precomputed_query_stats,
    )
    return tuple(compacted_cache), stats, target_article_tokens


def format_qa_prompts(
    tokenizer: Any,
    questions: Sequence[str],
    model_name: str,
    *,
    dataset_name: str = "tofu",
    answer_prefixes: Sequence[str] | None = None,
) -> list[str]:
    _activate_vendor()
    from evaluation.utils import format_question

    prefixes = list(answer_prefixes or ("" for _ in questions))
    if len(prefixes) != len(questions):
        raise ValueError("answer_prefixes must have the same length as questions.")
    prompts = []
    for question, answer_prefix in zip(questions, prefixes):
        if dataset_name == "ruler":
            content = question
        elif dataset_name == "tofu":
            content = (
                "Answer using only the provided context. Give a complete factual "
                f"answer in one or two sentences.\n\nQuestion: {question}"
            )
        elif dataset_name == "quality":
            content = (
                "Answer the multiple-choice question using only the provided "
                "context. Return only the letter A, B, C, or D.\n\n"
                f"{question}"
            )
        else:
            content = (
                "Answer using only the provided context. Return only the short "
                f"answer, without explanation.\n\nQuestion: {question}"
            )
        prompts.append(
            format_question(
                tokenizer,
                content,
                model_name=model_name,
                enable_thinking=False,
                answer_prefix=answer_prefix or None,
            )
        )
    return prompts


def answer_diagnostics(
    model: Any,
    tokenizer: Any,
    cache: tuple,
    prompt: str,
    answer: str,
    device: str,
    original_seq_len: int,
) -> dict[str, float | int]:
    _activate_vendor()
    from models.cache import CompactedPrefixCache
    from models.generate import get_sliding_layer_info

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Tokenize prompt+answer jointly so the teacher-forced continuation uses
    # exactly the same boundary tokens as generation. Encoding the answer in
    # isolation can select a different first token for whitespace-sensitive
    # tokenizers and materially corrupt NLL.
    joint_ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if joint_ids[: len(prompt_ids)] == prompt_ids:
        answer_ids = joint_ids[len(prompt_ids) :]
    else:
        # Chat templates normally end at a special-token boundary. Keep a safe
        # fallback for tokenizers whose final prompt token merges with text.
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        return {
            "answer_perplexity": float("nan"),
            "answer_nll": float("nan"),
            "answer_tokens": 0,
            "first_token_entropy": float("nan"),
            "first_token_margin": float("nan"),
            "reference_first_token_probability": float("nan"),
            "reference_first_token_rank": float("nan"),
            "post_answer_eos_probability": float("nan"),
            "post_answer_eos_rank": float("nan"),
            "post_answer_eos_logit_gap": float("nan"),
        }

    input_ids = torch.tensor(
        [prompt_ids + answer_ids],
        dtype=torch.long,
        device=device,
    )
    sliding_layer_indices, sliding_window = get_sliding_layer_info(model)
    moved_layers = tuple(
        (
            keys.to(device=device, dtype=model.dtype),
            beta.to(device=device, dtype=model.dtype),
            values.to(device=device, dtype=model.dtype),
        )
        for keys, beta, values in cache
    )
    prefix_cache = CompactedPrefixCache(
        moved_layers,
        original_seq_len=original_seq_len,
        sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
        sliding_window=sliding_window,
    )
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            past_key_values=prefix_cache,
            use_cache=True,
            return_dict=True,
        ).logits.float()

    start_idx = max(len(prompt_ids) - 1, 0)
    answer_logits = logits[:, start_idx : input_ids.size(1) - 1, :].contiguous()
    answer_labels = input_ids[:, start_idx + 1 : input_ids.size(1)].contiguous()
    loss = torch.nn.functional.cross_entropy(
        answer_logits.view(-1, answer_logits.size(-1)),
        answer_labels.view(-1),
        reduction="mean",
    )

    first_logits = answer_logits[0, 0]
    first_probabilities = torch.softmax(first_logits, dim=-1)
    first_log_probabilities = torch.log_softmax(first_logits, dim=-1)
    first_entropy = -(first_probabilities * first_log_probabilities).sum()
    top_two = torch.topk(first_logits, k=2).values
    first_reference_id = answer_labels[0, 0]
    first_reference_logit = first_logits[first_reference_id]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_probability = eos_rank = eos_logit_gap = float("nan")
    else:
        post_answer_logits = logits[0, -1]
        post_answer_probabilities = torch.softmax(post_answer_logits, dim=-1)
        eos_logit = post_answer_logits[eos_id]
        eos_probability = float(post_answer_probabilities[eos_id].item())
        eos_rank = int((post_answer_logits > eos_logit).sum().item() + 1)
        eos_logit_gap = float((eos_logit - post_answer_logits.max()).item())

    nll = float(loss.item())
    return {
        "answer_perplexity": math.exp(nll),
        "answer_nll": nll,
        "answer_tokens": int(answer_labels.numel()),
        "first_token_entropy": float(first_entropy.item()),
        "first_token_margin": float((top_two[0] - top_two[1]).item()),
        "reference_first_token_probability": float(
            first_probabilities[first_reference_id].item()
        ),
        "reference_first_token_rank": int(
            (first_logits > first_reference_logit).sum().item() + 1
        ),
        "post_answer_eos_probability": eos_probability,
        "post_answer_eos_rank": eos_rank,
        "post_answer_eos_logit_gap": eos_logit_gap,
    }


def generate_answers(
    model: Any,
    tokenizer: Any,
    cache: tuple,
    prompts: Sequence[str],
    max_new_tokens: int,
    original_seq_len: int,
) -> list[str]:
    _activate_vendor()
    from models.generate import generate_with_compacted_cache_batch

    return generate_with_compacted_cache_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=list(prompts),
        compacted_cache=cache,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        original_seq_len=original_seq_len,
    )


def count_generated_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
