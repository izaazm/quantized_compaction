# evaluation/run_reasoning_evaluation.py
"""
Warning: This eval runner uses online compaction, which is experimental and not fully tested. The compaction API will likely need changes before this can be set up well.

This script evaluates reasoning models on AIME problems with:
1. Baseline mode: Generate until </think> or max_seq_len, then force answer
2. Compaction mode: When hitting max_seq_len, compact the entire cache except the
    last protected_tokens, continue decoding, repeat until max_generated_tokens budget

Key differences from run_reasoning_evaluation.py:
- Simpler compaction: compact everything except protected_tokens at the end
- Fresh answer budget: always give 128 tokens for answer after </think>

Example usage:
    # Baseline: max 4096 tokens, force answer if limit hit
    python -m evaluation.run_reasoning_evaluation --mode baseline --max-seq-len 4096

    # Compaction: compact every phase, stop when reaching generation budget
    python -m evaluation.run_reasoning_evaluation --mode compaction --max-seq-len 1024 --target-size 0.1 --max-generated-tokens 4096

python -m evaluation.run_reasoning_evaluation --mode compaction --max-seq-len 1024 --target-size 0.1 --query-config repeat --max-generated-tokens 4096 --n-problems 1
"""
import argparse
import json
import math
import os
import random
import re
import time
import torch
from typing import Dict, List, Optional, Tuple, Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy optional for deterministic seeding
    np = None

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from compaction.compaction_methods import get_compaction_method, FullCacheCompactionAlgorithm
from compaction.query_generation import QueryConfig
from models.generate import (
    chunked_prefill,
    get_generation_params,
    get_sliding_layer_info,
    generate_with_compacted_cache,
)
from models.cache import CompactedPrefixCache

from .datasets import load_dataset
from .utils import (
    load_model_and_tokenizer,
    initialize_vllm,
)
from .configs.utils import load_algorithm_config, load_query_config


class ReasoningEvaluator:
    """
    Evaluate reasoning models with simplified mid-generation compaction.

    Compaction strategy:
    1. Generate up to max_seq_len tokens per phase
    2. Compact *everything* in current cache except last protected_tokens
    3. Continue generating with compacted cache
    4. Repeat until </think> found or max_generated_tokens reached
    5. Force answer if needed, always with fresh 128 token budget
    """

    FORCE_ANSWER_SEQUENCE = "\nI need to respond with the answer.\n</think>Final Answer: "
    MAX_ANSWER_TOKENS = 16

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = None,
        dtype: Optional[torch.dtype] = None,
        max_model_len: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.model = None
        self.tokenizer = None
        self.vllm_model = None

    def _ensure_model_loaded(self):
        """Load model and tokenizer if not already loaded."""
        if self.model is None:
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.model_name,
                self.device,
                self.dtype,
                self.max_model_len,
            )

    def _format_problem_prompt(self, problem_text: str, dataset_name: str) -> str:
        """Format the problem with instruction for step-by-step reasoning."""
        if dataset_name in {"gsm8k", "math500"}:
            instruction = (
                "Please think step by step. When you are ready to give your final answer, "
                "end your thinking and respond with only the numeric answer."
            )
        else:
            instruction = (
                "Please think step by step. When you are ready to give your final answer, "
                "end your thinking and respond with only the numeric answer (an integer from 0 to 999)."
            )
        content = f"{problem_text}\n\n{instruction}"

        messages = [{"role": "user", "content": content}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return prompt

    def _parse_aime_answer(self, response: str) -> Optional[int]:
        """Parse a 3-digit AIME answer (0-999) from model response."""
        # Prefer content right after the forced-answer sequence when present.
        if self.FORCE_ANSWER_SEQUENCE in response:
            after_force = response.split(self.FORCE_ANSWER_SEQUENCE)[-1].strip()
            first_num = re.search(r'\b(\d{1,3})\b', after_force)
            if first_num:
                answer = int(first_num.group(1))
                if 0 <= answer <= 999:
                    return answer
            after_think = after_force
        elif '</think>' in response:
            before_think = response.split('</think>')[0]
            boxed_matches = list(re.finditer(r'\\boxed\{([^}]*)\}', before_think))
            if boxed_matches:
                boxed_content = boxed_matches[-1].group(1)
                boxed_num = re.search(r'\b(\d{1,3})\b', boxed_content)
                if boxed_num:
                    answer = int(boxed_num.group(1))
                    if 0 <= answer <= 999:
                        return answer
            after_think = response.split('</think>')[-1].strip()
        else:
            after_think = response.strip()

        # Look for a standalone number (0-999)
        numbers = re.findall(r'\b(\d{1,3})\b', after_think)
        if numbers:
            answer = int(numbers[-1])
            if 0 <= answer <= 999:
                return answer

        # Fallback: any number in response after </think>
        if after_think:
            all_numbers = re.findall(r'(\d+)', after_think)
            for num_str in reversed(all_numbers):
                num = int(num_str)
                if 0 <= num <= 999:
                    return num

        return None

    def _parse_gsm8k_answer(self, response: str) -> Optional[str]:
        """Parse GSM8k numeric answer from model response."""
        # Prefer the part after </think> if present
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        # Try to find "The answer is X" pattern first
        match = re.search(r"[Tt]he answer is\s*(-?\d+(?:\.\d+)?)", response)
        if match:
            return match.group(1)

        # Try to find answer after #### marker (common in GSM8k solutions)
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
        if match:
            return match.group(1)

        # Fall back to last number in the text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        if numbers:
            return numbers[-1]

        return None

    def _parse_math500_answer(self, response: str) -> Optional[str]:
        """Parse MATH-500 answer from model response."""
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", response)
        if boxed_matches:
            return boxed_matches[-1].strip()

        final_match = re.search(r"Final Answer:\s*(.*)", response)
        if final_match:
            return final_match.group(1).strip()

        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        if numbers:
            return numbers[-1]

        return None

    def _normalize_math500_answer(self, answer: Optional[str]) -> Optional[str]:
        if answer is None:
            return None
        text = answer.strip()
        text = text.replace("$", "")
        text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
        text = text.replace("\\dfrac", "\\frac")
        text = text.replace("\\,", "")
        text = text.replace(" ", "")
        text = text.replace("\n", "")
        return text.rstrip(".")

    def _normalize_gsm8k_answer(self, answer: Optional[str]) -> Optional[str]:
        if answer is None:
            return None
        return answer.replace(",", "").strip()

    def _kv_to_compacted_format(
        self,
        kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Convert standard (K, V) cache to compacted (C1, beta, C2) format with zero beta."""
        result = []
        for layer_idx in range(len(kv_cache)):
            k = kv_cache[layer_idx][0]
            v = kv_cache[layer_idx][1]
            beta = torch.zeros(k.shape[0], k.shape[1], k.shape[2], device=k.device, dtype=k.dtype)
            result.append((k, beta, v))
        return tuple(result)


    def _get_cache_seq_len(
        self,
        cache: Tuple,
        sliding_layer_indices: Optional[set] = None,
    ) -> int:
        """Get the sequence length from a cache (max across non-sliding layers)."""
        sliding_layer_indices = sliding_layer_indices or set()
        max_len = 0
        for layer_idx in range(len(cache)):
            if layer_idx not in sliding_layer_indices:
                # Handle both (K, V) and (C1, beta, C2) formats
                layer_len = cache[layer_idx][0].shape[2]
                max_len = max(max_len, layer_len)
        return max_len

    def _get_effective_seq_len_from_stats(
        self,
        compaction_stats: Optional[Dict],
    ) -> Optional[int]:
        """Extract an integer effective sequence length from compaction stats."""
        if not compaction_stats:
            return None

        effective_len = compaction_stats.get('effective_compacted_seq_len')
        if effective_len is None:
            return None

        try:
            effective_value = float(effective_len)
        except (TypeError, ValueError):
            return None

        if effective_value <= 0:
            return None

        return int(math.ceil(effective_value))

    def _tensor_nbytes(self, tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _estimate_standard_kv_bytes(
        self,
        kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> int:
        total = 0
        for k, v in kv_cache:
            total += self._tensor_nbytes(k)
            total += self._tensor_nbytes(v)
        return total

    def _estimate_compacted_kv_bytes(
        self,
        compacted_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    ) -> int:
        total = 0
        for k, beta, v in compacted_cache:
            total += self._tensor_nbytes(k)
            total += self._tensor_nbytes(beta)
            total += self._tensor_nbytes(v)
        return total

    def _estimate_cache_bytes_from_layers(self, cache) -> int:
        total = 0
        for layer in cache.layers:
            total += self._tensor_nbytes(layer.keys)
            if hasattr(layer, "beta"):
                total += self._tensor_nbytes(layer.beta)
            total += self._tensor_nbytes(layer.values)
        return total

    def _estimate_standard_cache_bytes_for_seq_len(self, seq_len: int) -> int:
        """Estimate KV memory for a standard full-attention cache at a given sequence length."""
        if self.model is None:
            raise ValueError("Model must be loaded before estimating KV memory")

        config = self.model.config
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
        num_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
        num_kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
        head_dim = getattr(config, "head_dim", None)
        hidden_size = getattr(config, "hidden_size", None)
        if head_dim is None:
            if hidden_size is None or num_heads is None:
                raise ValueError("Unable to infer head_dim for KV memory estimation")
            head_dim = hidden_size // num_heads

        if num_layers is None or num_kv_heads is None:
            raise ValueError("Unable to infer layer/head counts for KV memory estimation")

        dtype_bytes = next(self.model.parameters()).element_size()
        return int(2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes)

    @torch.inference_mode()
    def evaluate_problem_baseline(
        self,
        problem_data: Dict,
        max_generated_tokens: int = 4096,
        seed: Optional[int] = None,
        dataset_name: str = "aime2025",
        deterministic: bool = True,
    ) -> Dict:
        """
        Evaluate a problem in baseline mode (no compaction).

        Generate until </think> or max_generated_tokens, then force answer if needed.
        """
        from vllm import SamplingParams

        problem_text = problem_data['article']
        if dataset_name in {"gsm8k", "math500"}:
            ground_truth = problem_data['questions'][0]['final_answer']
        else:
            ground_truth = problem_data['questions'][0]['ground_truth']

        prompt = self._format_problem_prompt(problem_text, dataset_name)
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        print(f"\nProblem: {problem_data['title']}")
        print(f"Prompt length: {prompt_len} tokens, max_generated_tokens: {max_generated_tokens}")

        start_time = time.time()

        gen_params = get_generation_params(self.model) if self.model is not None else {}
        temperature = gen_params.get('temperature') or 1.0
        top_k = gen_params.get('top_k') if gen_params.get('top_k') is not None else -1
        top_p = gen_params.get('top_p') or 1.0
        if deterministic:
            temperature = 0.0
            top_k = -1
            top_p = 1.0

        # Generate reasoning (stop on </think> or generation-token budget)
        max_reasoning_tokens = max_generated_tokens
        sampling_params = SamplingParams(
            max_tokens=max_reasoning_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop=["</think>"],
            include_stop_str_in_output=True,
            seed=seed,
        )

        self.vllm_model.wake_up()
        try:
            outputs = self.vllm_model.generate([prompt], sampling_params)
            reasoning_output = outputs[0].outputs[0]
            reasoning_text = reasoning_output.text
            reasoning_tokens = len(reasoning_output.token_ids)
            stopped_naturally = reasoning_output.finish_reason == "stop"

            if stopped_naturally:
                print(f"Stopped naturally after {reasoning_tokens} tokens")
                bare_think = reasoning_text.strip().endswith("</think>")
                if bare_think:
                    # Force answer sequence if model stopped right after </think>
                    final_only = "Final Answer: "
                    force_prompt = prompt + reasoning_text + final_only
                    answer_params = SamplingParams(
                        max_tokens=self.MAX_ANSWER_TOKENS,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                    )
                    answer_outputs = self.vllm_model.generate([force_prompt], answer_params)
                    answer_text = answer_outputs[0].outputs[0].text.strip()
                    answer_tokens = len(answer_outputs[0].outputs[0].token_ids)
                    full_response = reasoning_text + final_only + answer_text
                else:
                    # Continue to get the answer
                    continuation_prompt = prompt + reasoning_text
                    answer_params = SamplingParams(
                        max_tokens=self.MAX_ANSWER_TOKENS,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                    )
                    answer_outputs = self.vllm_model.generate([continuation_prompt], answer_params)
                    answer_text = answer_outputs[0].outputs[0].text.strip()
                    answer_tokens = len(answer_outputs[0].outputs[0].token_ids)
                    full_response = reasoning_text + answer_text
            else:
                # Force answer
                print(f"Hit token limit ({reasoning_tokens} tokens), forcing answer")
                force_prompt = prompt + reasoning_text + self.FORCE_ANSWER_SEQUENCE
                answer_params = SamplingParams(
                    max_tokens=self.MAX_ANSWER_TOKENS,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=seed,
                )
                answer_outputs = self.vllm_model.generate([force_prompt], answer_params)
                answer_text = answer_outputs[0].outputs[0].text.strip()
                answer_tokens = len(answer_outputs[0].outputs[0].token_ids)
                full_response = reasoning_text + self.FORCE_ANSWER_SEQUENCE + answer_text

        finally:
            self.vllm_model.sleep()

        generation_time = time.time() - start_time

        if dataset_name == "gsm8k":
            model_answer = self._parse_gsm8k_answer(full_response)
            normalized_model = self._normalize_gsm8k_answer(model_answer)
            normalized_gt = self._normalize_gsm8k_answer(str(ground_truth))
            is_correct = normalized_model == normalized_gt
        elif dataset_name == "math500":
            model_answer = self._parse_math500_answer(full_response)
            normalized_model = self._normalize_math500_answer(model_answer)
            normalized_gt = self._normalize_math500_answer(str(ground_truth))
            is_correct = normalized_model == normalized_gt
        else:
            model_answer = self._parse_aime_answer(full_response)
            try:
                ground_truth_int = int(ground_truth)
                is_correct = (model_answer == ground_truth_int) if model_answer is not None else False
            except ValueError:
                is_correct = False

        print(f"Ground truth: {ground_truth}, Model answer: {model_answer}, Correct: {is_correct}")
        print(f"Time: {generation_time:.2f}s")

        return {
            'problem_id': problem_data['article_id'],
            'problem_title': problem_data['title'],
            'ground_truth': ground_truth,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'reasoning_tokens': reasoning_tokens,
            'answer_tokens': answer_tokens,
            'total_generated_tokens': reasoning_tokens + answer_tokens,
            'stopped_naturally': stopped_naturally,
            'generation_time': generation_time,
            'prompt_tokens': prompt_len,
            'full_response': full_response,
            'peak_kv_memory_bytes': self._estimate_standard_cache_bytes_for_seq_len(prompt_len + reasoning_tokens + answer_tokens),
            'peak_kv_memory_mb': self._estimate_standard_cache_bytes_for_seq_len(prompt_len + reasoning_tokens + answer_tokens) / (1024 ** 2),
        }

    @torch.inference_mode()
    def evaluate_problem_compaction(
        self,
        problem_data: Dict,
        compaction_method: FullCacheCompactionAlgorithm,
        query_config: QueryConfig,
        compaction_interval: int = 4096,
        target_size: float = 0.1,
        max_generated_tokens: int = 4096,
        hybrid_compressed_budget: int = 2048,
        protected_tokens: int = 20,
        seed: Optional[int] = None,
        dataset_name: str = "aime2025",
        compaction_policy: str = "recompact_all",
        deterministic: bool = False,
    ) -> Dict:
        """
        Evaluate with mid-generation compaction.

        Strategy:
        1. Generate up to compaction_interval tokens in each phase
        2. Compact cache according to policy
        3. Repeat until </think> or total generated reaches max_generated_tokens
        4. Force answer with fresh 128 token budget
        """
        from vllm import SamplingParams

        self._ensure_model_loaded()
        device = next(self.model.parameters()).device

        problem_text = problem_data['article']
        if dataset_name in {"gsm8k", "math500"}:
            ground_truth = problem_data['questions'][0]['final_answer']
        else:
            ground_truth = problem_data['questions'][0]['ground_truth']

        prompt = self._format_problem_prompt(problem_text, dataset_name)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        print(f"\nProblem: {problem_data['title']}")
        print(
            f"Prompt: {prompt_len} tokens, compaction_interval: {compaction_interval}, "
            f"max_generated_tokens: {max_generated_tokens}, target: {target_size}"
        )

        start_time = time.time()

        gen_params = get_generation_params(self.model)
        temperature = gen_params['temperature'] if gen_params['temperature'] is not None else 1.0
        top_k = gen_params.get('top_k') if gen_params.get('top_k') is not None else -1
        top_p = gen_params.get('top_p') or 1.0
        if deterministic:
            temperature = 0.0
            top_k = -1
            top_p = 1.0
        hf_top_k = None if top_k is None or top_k < 0 else top_k

        sliding_layer_indices, sliding_window = get_sliding_layer_info(self.model)

        # Tracking
        phase_tokens = []
        phase_times = []
        compaction_times = []
        all_compaction_stats = []
        accumulated_text = ""
        compaction_count = 0
        stopped_naturally = False
        total_generated_tokens = 0
        peak_kv_memory_bytes = 0
        last_compacted_seq_len = 0

        # Cache state
        compacted_cache = None  # (C1, beta, C2) format
        original_seq_len = None
        last_token_str = None

        while total_generated_tokens < max_generated_tokens:
            phase_start = time.time()
            remaining_budget = max_generated_tokens - total_generated_tokens
            phase_generation_budget = min(compaction_interval, remaining_budget)
            if phase_generation_budget <= 0:
                break

            print(
                f"\n--- Phase {len(phase_tokens) + 1} (compactions so far: {compaction_count}) ---"
            )
            print(
                f"Phase generation budget: {phase_generation_budget}, "
                f"remaining total budget: {remaining_budget}"
            )

            if compacted_cache is None:
                # First phase: use vLLM
                sampling_params = SamplingParams(
                    max_tokens=phase_generation_budget,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stop=["</think>"],
                    include_stop_str_in_output=True,
                    seed=seed,
                )

                self.vllm_model.wake_up()
                try:
                    outputs = self.vllm_model.generate([prompt], sampling_params)
                    phase_output = outputs[0].outputs[0]
                    phase_text = phase_output.text
                    tokens_generated = len(phase_output.token_ids)
                    stopped_naturally = phase_output.finish_reason == "stop"
                finally:
                    self.vllm_model.sleep()

                accumulated_text = phase_text

                # Track peak KV for first vLLM phase
                peak_kv_memory_bytes = max(
                    peak_kv_memory_bytes,
                    self._estimate_standard_cache_bytes_for_seq_len(prompt_len + tokens_generated),
                )
            else:
                # Subsequent phases: use HF with compacted cache
                # -1 because the seed token (last_token_str) will also be added to the cache
                phase_text, tokens_generated, stopped_naturally, compacted_cache = generate_with_compacted_cache(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=last_token_str,
                    compacted_cache=compacted_cache,
                    max_new_tokens=phase_generation_budget,
                    original_seq_len=original_seq_len,
                    temperature=temperature,
                    top_k=hf_top_k,
                    top_p=top_p,
                    stop_strings=["</think>"],
                    return_cache=True,
                )
                accumulated_text += phase_text
                # Update original_seq_len to track logical sequence length (as if no compaction)
                # +1 for the prompt token (last_token_str), +tokens_generated for new tokens
                original_seq_len += 1 + tokens_generated

                peak_kv_memory_bytes = max(
                    peak_kv_memory_bytes,
                    self._estimate_compacted_kv_bytes(compacted_cache),
                )

            phase_time = time.time() - phase_start
            phase_tokens.append(tokens_generated)
            phase_times.append(phase_time)
            total_generated_tokens += tokens_generated

            print(
                f"Generated {tokens_generated} tokens in {phase_time:.2f}s "
                f"(total generated: {total_generated_tokens}/{max_generated_tokens})"
            )

            if stopped_naturally:
                print(f"Stopped naturally (</think> or EOS)")
                break

            if total_generated_tokens >= max_generated_tokens:
                print(f"Reached max_generated_tokens ({max_generated_tokens}), forcing answer")
                break

            # === COMPACTION ===
            print(f"\n--- Compaction {compaction_count + 1} ---")
            compaction_start = time.time()

            # Get the last token for generation seed
            full_text = prompt + accumulated_text
            all_input_ids = self.tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(device)
            last_token_id = all_input_ids[0, -1].item()
            last_token_str = self.tokenizer.decode([last_token_id], skip_special_tokens=False)

            if compacted_cache is None:
                # First compaction (after vLLM phase 1): need HF prefill to get KV cache
                prefix_input_ids = all_input_ids[:, :-1]
                prefix_len = prefix_input_ids.shape[1]

                print(f"Running HF prefill on {prefix_len} tokens...")
                prefill_start = time.time()
                outputs = self.model(input_ids=prefix_input_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                print(f"Prefill done in {time.time() - prefill_start:.2f}s")

                peak_kv_memory_bytes = max(
                    peak_kv_memory_bytes,
                    self._estimate_standard_kv_bytes(past_key_values),
                )

                # Convert to (C1, beta, C2) format with zero beta
                compacted_cache = self._kv_to_compacted_format(past_key_values)
                peak_kv_memory_bytes = max(
                    peak_kv_memory_bytes,
                    self._estimate_compacted_kv_bytes(compacted_cache),
                )
            else:
                # Subsequent compactions: we already have compacted_cache from generation
                # The cache already includes: compacted prefix + last_token_str + generated tokens
                pass

            # Logical sequence length of the un-compacted cache (full formatted context)
            logical_seq_len = all_input_ids.shape[1]

            # Wrap full cache for query generation (full context)
            cache_for_queries = CompactedPrefixCache(
                compacted_cache,
                original_seq_len=logical_seq_len,
                sliding_layer_indices=sliding_layer_indices if sliding_layer_indices else None,
                sliding_window=sliding_window,
            )

            peak_kv_memory_bytes = max(
                peak_kv_memory_bytes,
                self._estimate_cache_bytes_from_layers(cache_for_queries),
            )

            # Compact the current cache.
            total_seq_len = self._get_cache_seq_len(compacted_cache, sliding_layer_indices)
            # Hybrid mode concatenates until the compressed cache reaches a fixed budget,
            # then recompacts the entire live context into a new compacted cache.
            effective_policy = compaction_policy
            if compaction_policy == "hybrid":
                if total_seq_len < hybrid_compressed_budget:
                    effective_policy = "concatenate"
                else:
                    effective_policy = "recompact_all"

            if effective_policy == "concatenate":
                compactable_start = min(max(last_compacted_seq_len, 0), total_seq_len)
            else:
                compactable_start = 0
            compactable_end = max(0, total_seq_len - protected_tokens)
            compactable_len = max(0, compactable_end - compactable_start)

            if compactable_len <= 0:
                print(
                    f"Warning: Nothing to compact (seq_len={total_seq_len}, protected={protected_tokens})"
                )
                if compaction_policy == "hybrid":
                    print(
                        f"Hybrid cache is under budget ({total_seq_len} < {hybrid_compressed_budget}); skipping compaction"
                    )
                    continue
                if effective_policy == "concatenate":
                    print("Concatenate has no new tokens beyond protected tail; skipping compaction")
                    continue
                break

            if 0 < target_size < 1:
                target_compacted_size = max(1, int(compaction_interval * target_size))
            else:
                target_compacted_size = int(target_size)

            # Re-align target_total_size to match the actual minimum target size
            target_total_size = compactable_start + target_compacted_size + protected_tokens
            print(
                f"Compacting {compactable_len} / {total_seq_len} tokens -> {target_compacted_size} "
                f"(policy={effective_policy}, protecting last {protected_tokens}, excluding first {compactable_start})"
            )

            indices_to_compact = range(compactable_start, compactable_end)
            context_text = self.tokenizer.decode(all_input_ids[0], skip_special_tokens=False)
            compacted_cache, compaction_stats = compaction_method.compact_kv_cache(
                past_key_values=cache_for_queries,
                target_size=target_total_size,
                indices=indices_to_compact,
                query_config=query_config,
                model=self.model,
                tokenizer=self.tokenizer,
                formatted_context=context_text,
                compute_stats=False,
                vllm_model=self.vllm_model,
                sliding_layer_indices=sliding_layer_indices,
                past_key_values_for_queries=cache_for_queries,
                full_query_extraction=True,
            )

            peak_kv_memory_bytes = max(
                peak_kv_memory_bytes,
                self._estimate_cache_bytes_from_layers(cache_for_queries) + self._estimate_compacted_kv_bytes(compacted_cache),
            )

            # Keep logical sequence length aligned with true (pre-compaction) context length
            original_seq_len = logical_seq_len

            # Report compacted cache length (for visibility only)
            tensor_cache_len = self._get_cache_seq_len(compacted_cache, sliding_layer_indices)
            effective_cache_len = self._get_effective_seq_len_from_stats(compaction_stats)
            if effective_cache_len is not None:
                cache_len_desc = f"{effective_cache_len} (effective, tensor {tensor_cache_len})"
            else:
                cache_len_desc = f"{tensor_cache_len} (tensor)"

            last_compacted_seq_len = compactable_start + target_compacted_size

            compaction_time = time.time() - compaction_start
            compaction_times.append(compaction_time)
            all_compaction_stats.append({
                k: v for k, v in compaction_stats.items()
                if not isinstance(v, torch.Tensor)
            })
            compaction_count += 1

            print(f"Compaction done in {compaction_time:.2f}s, new cache len: {cache_len_desc}")

        # === ANSWER GENERATION ===
        if stopped_naturally and '</think>' in accumulated_text:
            print(f"Generating answer after natural </think>...")
            bare_think = accumulated_text.strip().endswith("</think>")

            if compacted_cache is None:
                # Use vLLM
                if bare_think:
                    final_only = "Final Answer: "
                    force_prompt = prompt + accumulated_text + final_only
                    answer_params = SamplingParams(
                        max_tokens=self.MAX_ANSWER_TOKENS,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                    )
                    self.vllm_model.wake_up()
                    try:
                        answer_outputs = self.vllm_model.generate([force_prompt], answer_params)
                        answer_text = answer_outputs[0].outputs[0].text.strip()
                        answer_tokens = len(answer_outputs[0].outputs[0].token_ids)
                    finally:
                        self.vllm_model.sleep()
                    full_response = accumulated_text + final_only + answer_text
                else:
                    continuation_prompt = prompt + accumulated_text
                    answer_params = SamplingParams(
                        max_tokens=self.MAX_ANSWER_TOKENS,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                    )
                    self.vllm_model.wake_up()
                    try:
                        answer_outputs = self.vllm_model.generate([continuation_prompt], answer_params)
                        answer_text = answer_outputs[0].outputs[0].text.strip()
                        answer_tokens = len(answer_outputs[0].outputs[0].token_ids)
                    finally:
                        self.vllm_model.sleep()
                    full_response = accumulated_text + answer_text
            else:
                if bare_think:
                    # Use HF with compacted cache and force answer sequence
                    final_only = "Final Answer: "
                    answer_text, answer_tokens, _ = generate_with_compacted_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=final_only,
                        compacted_cache=compacted_cache,
                        max_new_tokens=self.MAX_ANSWER_TOKENS,
                        original_seq_len=original_seq_len,
                        temperature=temperature,
                        top_k=hf_top_k,
                        top_p=top_p,
                    )
                    full_response = accumulated_text + final_only + answer_text
                else:
                    # Use HF with compacted cache
                    # The cache already has all tokens. We need to pop the last token from the
                    # cache and use it as the generation seed to avoid HF's empty input_ids issue.
                    full_text = prompt + accumulated_text
                    all_input_ids = self.tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(device)
                    last_token_id = all_input_ids[0, -1].item()
                    last_token_str = self.tokenizer.decode([last_token_id], skip_special_tokens=False)

                    # Pop the last token from cache (it will be re-added as the prompt)
                    trimmed_cache = tuple(
                        (keys[:, :, :-1, :], beta[:, :, :-1], values[:, :, :-1, :])
                        for keys, beta, values in compacted_cache
                    )

                    answer_text, answer_tokens, _ = generate_with_compacted_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=last_token_str,
                        compacted_cache=trimmed_cache,
                        max_new_tokens=self.MAX_ANSWER_TOKENS - 1,
                        original_seq_len=original_seq_len - 1,  # -1 because we popped a token
                        temperature=temperature,
                        top_k=hf_top_k,
                        top_p=top_p,
                    )
                    full_response = accumulated_text + answer_text

        elif not stopped_naturally:
            # Force answer
            print(f"Forcing answer...")

            if compacted_cache is None:
                # Need to build cache first
                full_text = prompt + accumulated_text + self.FORCE_ANSWER_SEQUENCE
                all_input_ids = self.tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(device)
                prefix_input_ids = all_input_ids[:, :-1]
                last_token_id = all_input_ids[0, -1].item()
                last_token_str = self.tokenizer.decode([last_token_id], skip_special_tokens=False)

                outputs = self.model(input_ids=prefix_input_ids, use_cache=True)
                compacted_cache = self._kv_to_compacted_format(outputs.past_key_values)
                original_seq_len = prefix_input_ids.shape[1]

                # Generate answer using the cache we just built
                answer_text, answer_tokens, _ = generate_with_compacted_cache(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=last_token_str,  # Just the seed token
                    compacted_cache=compacted_cache,
                    max_new_tokens=self.MAX_ANSWER_TOKENS,
                    original_seq_len=original_seq_len,
                    temperature=temperature,
                    top_k=hf_top_k,
                    top_p=top_p,
                )
            else:
                # Cache already has generated tokens, just add force answer sequence and generate
                answer_text, answer_tokens, _ = generate_with_compacted_cache(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=self.FORCE_ANSWER_SEQUENCE,  # Add force answer to existing cache
                    compacted_cache=compacted_cache,
                    max_new_tokens=self.MAX_ANSWER_TOKENS,
                    original_seq_len=original_seq_len,
                    temperature=temperature,
                    top_k=hf_top_k,
                    top_p=top_p,
                )

            full_response = accumulated_text + self.FORCE_ANSWER_SEQUENCE + answer_text
        else:
            # Stopped naturally but no </think>
            answer_text = ""
            answer_tokens = 0
            full_response = accumulated_text

        generation_time = time.time() - start_time

        if dataset_name == "gsm8k":
            model_answer = self._parse_gsm8k_answer(full_response)
            normalized_model = self._normalize_gsm8k_answer(model_answer)
            normalized_gt = self._normalize_gsm8k_answer(str(ground_truth))
            is_correct = normalized_model == normalized_gt
        elif dataset_name == "math500":
            model_answer = self._parse_math500_answer(full_response)
            normalized_model = self._normalize_math500_answer(model_answer)
            normalized_gt = self._normalize_math500_answer(str(ground_truth))
            is_correct = normalized_model == normalized_gt
        else:
            model_answer = self._parse_aime_answer(full_response)
            try:
                ground_truth_int = int(ground_truth)
                is_correct = (model_answer == ground_truth_int) if model_answer is not None else False
            except ValueError:
                is_correct = False

        print(f"Ground truth: {ground_truth}, Model answer: {model_answer}, Correct: {is_correct}")
        print(f"Total time: {generation_time:.2f}s")

        return {
            'problem_id': problem_data['article_id'],
            'problem_title': problem_data['title'],
            'ground_truth': ground_truth,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'phase_tokens': phase_tokens,
            'total_reasoning_tokens': sum(phase_tokens),
            'total_generated_tokens': total_generated_tokens + answer_tokens,
            'answer_tokens': answer_tokens,
            'stopped_naturally': stopped_naturally,
            'generation_time': generation_time,
            'phase_times': phase_times,
            'compaction_times': compaction_times,
            'total_compaction_time': sum(compaction_times),
            'prompt_tokens': prompt_len,
            'num_compactions': compaction_count,
            'compaction_interval': compaction_interval,
            'max_generated_tokens': max_generated_tokens,
            'target_size': target_size,
            'full_response': full_response,
            'compaction_stats': all_compaction_stats,
            'compaction_policy': compaction_policy,
            'peak_kv_memory_bytes': peak_kv_memory_bytes,
            'peak_kv_memory_mb': peak_kv_memory_bytes / (1024 ** 2),
        }

    def run_evaluation(
        self,
        dataset_name: str = "aime2025",
        mode: str = "baseline",
        compaction_method: Optional[FullCacheCompactionAlgorithm] = None,
        query_config: Optional[QueryConfig] = None,
        compaction_interval: int = 4096,
        target_size: float = 0.1,
        max_generated_tokens: int = 4096,
        protected_tokens: int = 20,
        seed: Optional[int] = None,
        deterministic: bool = True,
        n_problems: int = -1,
        start_problem: int = 0,
        log_dir: str = "logs/reasoning_evaluation",
        experiment_name: Optional[str] = None,
        algorithm_config_file: Optional[str] = None,
        query_config_file: Optional[str] = None,
        compaction_policy: str = "recompact_all",
        hybrid_compressed_budget: int = 2048,
    ) -> Dict:
        """Run evaluation across multiple problems."""
        if deterministic:
            seed = _apply_determinism(seed)

        dataset = load_dataset(dataset_name)

        if n_problems == -1:
            problem_indices = list(range(start_problem, len(dataset)))
        else:
            end_problem = min(start_problem + n_problems, len(dataset))
            problem_indices = list(range(start_problem, end_problem))

        print(f"\n{'='*60}")
        print(f"REASONING EVALUATION")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}, Problems: {len(problem_indices)}")
        print(f"Mode: {mode}, compaction_interval: {compaction_interval}")
        if mode == "baseline":
            print(f"Baseline max_generated_tokens: {max_generated_tokens}")
        if mode == "compaction":
            print(
                f"Target: {target_size}, max_generated_tokens: {max_generated_tokens}, protected: {protected_tokens}, "
                f"policy: {compaction_policy}"
            )
            if compaction_policy == "hybrid":
                print(f"Hybrid compressed budget: {hybrid_compressed_budget}")
            print(f"Method: {compaction_method.name() if compaction_method else 'N/A'}")
        print(f"{'='*60}\n")

        if self.vllm_model is None:
            print("Initializing vLLM...")
            self.vllm_model = initialize_vllm(self.model_name, max_model_len=self.max_model_len)

        self._ensure_model_loaded()

        all_results = []

        for problem_idx in problem_indices:
            problem_data = dataset[problem_idx]

            if mode == "baseline":
                result = self.evaluate_problem_baseline(
                    problem_data=problem_data,
                    max_generated_tokens=max_generated_tokens,
                    seed=seed,
                    dataset_name=dataset_name,
                    deterministic=deterministic,
                )
            else:
                if compaction_method is None:
                    raise ValueError("compaction_method required for compaction mode")
                result = self.evaluate_problem_compaction(
                    problem_data=problem_data,
                    compaction_method=compaction_method,
                    query_config=query_config,
                    compaction_interval=compaction_interval,
                    target_size=target_size,
                    max_generated_tokens=max_generated_tokens,
                    hybrid_compressed_budget=hybrid_compressed_budget,
                    protected_tokens=protected_tokens,
                    seed=seed,
                    dataset_name=dataset_name,
                    compaction_policy=compaction_policy,
                    deterministic=deterministic,
                )

            result['problem_idx'] = problem_idx
            all_results.append(result)

        # Compute stats
        total_problems = len(all_results)
        correct = sum(1 for r in all_results if r.get('is_correct', False))
        accuracy = correct / total_problems if total_problems > 0 else 0.0

        avg_reasoning_tokens = sum(
            r.get('total_reasoning_tokens', r.get('reasoning_tokens', 0))
            for r in all_results
        ) / total_problems if total_problems > 0 else 0

        avg_total_generated_tokens = sum(
            r.get('total_generated_tokens', 0)
            for r in all_results
        ) / total_problems if total_problems > 0 else 0

        avg_answer_tokens = sum(
            r.get('answer_tokens', 0)
            for r in all_results
        ) / total_problems if total_problems > 0 else 0

        avg_generation_time = sum(
            r.get('generation_time', 0) for r in all_results
        ) / total_problems if total_problems > 0 else 0

        max_peak_kv_memory_mb = max(
            (r.get('peak_kv_memory_mb', 0.0) for r in all_results),
            default=0.0,
        )
        avg_peak_kv_memory_mb = sum(
            r.get('peak_kv_memory_mb', 0.0) for r in all_results
        ) / total_problems if total_problems > 0 else 0.0

        overall_stats = {
            'total_problems': total_problems,
            'correct': correct,
            'accuracy': accuracy,
            'avg_reasoning_tokens': avg_reasoning_tokens,
            'avg_total_generated_tokens': avg_total_generated_tokens,
            'avg_answer_tokens': avg_answer_tokens,
            'avg_generation_time': avg_generation_time,
            'max_peak_kv_memory_mb': max_peak_kv_memory_mb,
            'avg_peak_kv_memory_mb': avg_peak_kv_memory_mb,
        }

        if mode == "compaction":
            avg_compaction_time = sum(
                r.get('total_compaction_time', 0) for r in all_results
            ) / total_problems if total_problems > 0 else 0
            avg_num_compactions = sum(
                r.get('num_compactions', 0) for r in all_results
            ) / total_problems if total_problems > 0 else 0
            overall_stats['avg_compaction_time'] = avg_compaction_time
            overall_stats['avg_num_compactions'] = avg_num_compactions

        # Save results
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{experiment_name}_{timestamp}" if experiment_name else f"reasoning2_{mode}_{timestamp}"
        filepath = log_path / f"{base_filename}.json"

        query_config_dict = None
        if query_config:
            query_config_dict = asdict(query_config)
            for mc in query_config_dict.get('method_configs', []):
                if mc.get('method') == 'self_study' and 'config' in mc and 'conversation_specs' in mc['config']:
                    for spec in mc['config']['conversation_specs']:
                        spec.pop('extraction_fn', None)

        output = {
            'timestamp': timestamp,
            'experiment_name': experiment_name,
            'algorithm_config_file': algorithm_config_file,
            'query_config_file': query_config_file,
            'config': {
                'model_name': self.model_name,
                'dataset_name': dataset_name,
                'mode': mode,
                'n_problems': len(problem_indices),
                'start_problem': start_problem,
                'compaction_interval': compaction_interval if mode == "compaction" else None,
                'target_size': target_size if mode == "compaction" else None,
                'max_generated_tokens': max_generated_tokens,
                'protected_tokens': protected_tokens if mode == "compaction" else None,
                'compaction_policy': compaction_policy if mode == "compaction" else None,
                'deterministic': deterministic,
                'seed': seed,
                'method': compaction_method.name() if compaction_method else None,
                'query_config': query_config_dict,
            },
            'overall_stats': overall_stats,
            'results': all_results,
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Accuracy: {correct}/{total_problems} = {accuracy:.2%}")
        print(f"Avg total generated tokens: {avg_total_generated_tokens:.0f}")
        print(f"  Avg reasoning tokens: {avg_reasoning_tokens:.0f}")
        print(f"  Avg answer tokens: {avg_answer_tokens:.0f}")
        print(f"Avg generation time: {avg_generation_time:.2f}s")
        print(f"Peak KV memory: {max_peak_kv_memory_mb:.2f} MB (max), {avg_peak_kv_memory_mb:.2f} MB (avg)")
        if mode == "compaction":
            print(f"Avg compactions: {overall_stats['avg_num_compactions']:.1f}")
            print(f"Avg compaction time: {overall_stats['avg_compaction_time']:.2f}s")
        print(f"Results saved to: {filepath}")
        print(f"{'='*60}\n")

        return output


def _apply_determinism(seed: Optional[int]) -> int:
    """Seed RNGs and enable deterministic kernels when possible."""
    if seed is None:
        seed = 0

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_available() and not cublas_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print(
            "Warning: CUBLAS_WORKSPACE_CONFIG was not set. "
            "Set it before launching for full determinism. "
            "Using ':4096:8' for next run."
        )
        return seed

    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        print(f"Warning: deterministic algorithms unavailable ({exc}). Proceeding without.")

    return seed


def main():
    parser = argparse.ArgumentParser(
        description='Reasoning evaluation with simplified compaction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'compaction'])
    parser.add_argument('--dataset-name', type=str, default='aime2025')
    parser.add_argument('--n-problems', type=int, default=-1)
    parser.add_argument('--start-problem', type=int, default=0)

    parser.add_argument(
        '--compaction-interval', '--max-seq-len', '--max-len', '--max_len',
        dest='compaction_interval',
        type=int,
        default=4096,
        help='Number of tokens to generate per phase before compacting the KV cache.',
    )
    parser.add_argument('--target-size', type=float, default=0.1)
    parser.add_argument('--max-generated-tokens', type=int, default=4096)
    parser.add_argument('--protected-tokens', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--method', type=str, default='AM-HighestAttnKeys-basic')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic generation (greedy decoding) (default: True)'
    )
    parser.add_argument(
        '--no-deterministic',
        action='store_false',
        dest='deterministic',
        help='Disable deterministic generation (use sampling)'
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        '--compaction-policy',
        type=str,
        default='recompact_all',
        choices=['recompact_all', 'concatenate', 'hybrid'],
        help=(
            "Compaction policy when compaction_interval is reached. "
            "'recompact_all' compacts both previous compacted cache and uncompacted context. "
            "'concatenate' compacts only uncompacted context, appending to previous compacted cache. "
            "'hybrid' concatenates until hybrid_budget, then recompacts all."
        ),
    )

    parser.add_argument(
        '--hybrid-compressed-budget',
        type=int,
        default=2048,
        help='Fixed compressed-cache budget for hybrid mode; concatenate until the cache reaches this budget, then recompact all.',
    )

    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--long-context', action='store_true')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Override the model max length used by vLLM/HF cache allocation.',
    )

    parser.add_argument('--algorithm-config', type=str, default='default')
    parser.add_argument('--query-config', type=str, default='repeat')
    parser.add_argument(
        '--precomputed-budget-path',
        type=str,
        default=None,
        help='Path to precomputed head budget proportions JSON file (for nonuniform head budgets)'
    )
    parser.add_argument(
        '--max-ratio-per-head',
        type=float,
        default=1.0,
        help='Maximum ratio per head when using precomputed budgets (default: 1.0). '
             'If budgets would assign a higher ratio, proportions are blended towards uniform.'
    )

    parser.add_argument('--log-dir', type=str, default='logs/reasoning_evaluation')
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device != 'cuda':
        raise RuntimeError("CUDA is required")

    if args.name is None:
        if args.mode == 'baseline':
            args.name = f"baseline_{args.max_generated_tokens}"
        else:
            args.name = f"compact_{args.compaction_interval}_{args.target_size}_{args.max_generated_tokens}"

    method_config = load_algorithm_config(args.algorithm_config, target_size=args.target_size)
    query_config = load_query_config(args.query_config)

    print(f"Algorithm config: {args.algorithm_config}")
    print(f"Query config: {args.query_config}")

    if args.max_model_len is not None:
        if args.max_model_len == -1:
            max_model_len = None
        else:
            max_model_len = args.max_model_len
    else:
        if args.mode == 'baseline':
            max_model_len = args.max_generated_tokens + 2048
        else:
            max_model_len = args.compaction_interval + 2048
    evaluator = ReasoningEvaluator(
        model_name=args.model_name,
        device=args.device,
        max_model_len=max_model_len,
    )

    compaction_method = None
    if args.mode == 'compaction':
        if args.method not in method_config:
            raise ValueError(f"Method '{args.method}' not found. Available: {list(method_config.keys())}")
        method_kwargs = dict(method_config[args.method])
        if args.precomputed_budget_path is not None:
            method_kwargs['precomputed_budget_path'] = args.precomputed_budget_path
            method_kwargs['max_ratio_per_head'] = args.max_ratio_per_head
        compaction_method = get_compaction_method(args.method, method_kwargs)

    evaluator.run_evaluation(
        dataset_name=args.dataset_name,
        mode=args.mode,
        compaction_method=compaction_method,
        query_config=query_config,
        compaction_interval=args.compaction_interval,
        target_size=args.target_size,
        max_generated_tokens=args.max_generated_tokens,
        hybrid_compressed_budget=args.hybrid_compressed_budget,
        protected_tokens=args.protected_tokens,
        seed=args.seed,
        n_problems=args.n_problems,
        start_problem=args.start_problem,
        log_dir=args.log_dir,
        experiment_name=args.name,
        algorithm_config_file=args.algorithm_config,
        query_config_file=args.query_config,
        compaction_policy=args.compaction_policy,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
