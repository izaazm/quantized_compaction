#!/usr/bin/env python
"""
Compare compaction schemes for two-stage TOFU pipeline.

Pipeline:
- Stage 1: Compact author A into a compacted cache
- Stage 2: Process author B and either:
  - concatenate: compact author B and append to A's compacted cache
  - recompact_all: recompact both uncompacted B and already-compacted A
  - baseline: no compaction
"""
import argparse
import json
import sys
import os
import copy
import random
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed as hf_set_seed

# Setup paths for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from evaluation.utils import load_model_and_tokenizer
from evaluation.configs.utils import load_algorithm_config, load_query_config
from compaction.compaction_methods import get_compaction_method
from models.generate import generate_with_compacted_cache_batch, chunked_prefill, get_sliding_layer_info
from models.cache import CompactedPrefixCache, clone_compacted_prefix_cache

from cartridges.cartridges.data.tofu.evals import TOFUQAGenerateDataset
from cartridges.cartridges.data.tofu.utils import load_tofu_authors


_BIOGRAPHY_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "cartridges", "data_synthesis", "tofu_author_biographies.json"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_biography_index(json_path: str) -> dict:
    """Load biography index for TOFU authors."""
    json_path = os.path.abspath(json_path)
    if not os.path.exists(json_path):
        print(f"  [biography] JSON not found at {json_path}, falling back to 'qa' format.")
        return {}
    with open(json_path) as f:
        data = json.load(f)
    return {entry["author_idx"]: entry for entry in data}


def build_corpus(authors, corpus_format: str, biography_index: dict) -> str:
    """Build corpus text from authors."""
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


def _kv_to_compacted_format(kv_cache):
    """Convert standard (K, V) cache to compacted (C1, beta, C2) format with zero beta."""
    if hasattr(kv_cache, "key_cache"):
        kv_tuples = tuple(zip(kv_cache.key_cache, kv_cache.value_cache))
    else:
        kv_tuples = kv_cache
        
    result = []
    for k, v in kv_tuples:
        beta = torch.zeros(k.shape[0], k.shape[1], k.shape[2], device=k.device, dtype=k.dtype)
        result.append((k.contiguous(), beta.contiguous(), v.contiguous()))
    return tuple(result)


def extract_compacted_cache_tuples(cache: CompactedPrefixCache) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Extract C1, beta, C2 tensors from CompactedPrefixCache layers, padding beta with zeros."""
    tuples = []
    for layer in cache.layers:
        keys = layer.keys
        values = layer.values
        if hasattr(layer, 'beta'):
            original_beta = layer.beta
            base_len = layer.base_len
            current_len = keys.shape[2]
            if current_len > base_len:
                new_tokens = current_len - base_len
                zeros = torch.zeros(
                    original_beta.shape[0], original_beta.shape[1], new_tokens,
                    dtype=original_beta.dtype, device=original_beta.device
                )
                beta = torch.cat([original_beta, zeros], dim=2)
            else:
                beta = original_beta
        else:
            beta = torch.zeros(
                keys.shape[0], keys.shape[1], keys.shape[2],
                dtype=keys.dtype, device=keys.device
            )
        tuples.append((keys.contiguous(), beta.contiguous(), values.contiguous()))
    return tuple(tuples)


def _slice_compacted_cache(
    cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    start_idx: int,
    end_idx: Optional[int] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Slice key, beta, value tensors in the cache along the sequence dimension."""
    sliced = []
    for k, beta, v in cache:
        layer_len = k.shape[2]
        s = min(max(start_idx, 0), layer_len)
        e = layer_len if end_idx is None else min(max(end_idx, s), layer_len)
        sliced.append((
            k[:, :, s:e, :].contiguous(),
            beta[:, :, s:e].contiguous(),
            v[:, :, s:e, :].contiguous()
        ))
    return tuple(sliced)


def _concat_compacted_caches(
    base_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    new_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    sliding_layer_indices: Optional[set] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Concatenate two compacted caches along the sequence dimension."""
    sliding_layer_indices = sliding_layer_indices or set()
    combined = []
    for layer_idx in range(len(base_cache)):
        if layer_idx in sliding_layer_indices:
            # sliding window layers don't concatenate, they just keep the latest (B)
            # make sure it is contiguous
            new_k, new_beta, new_v = new_cache[layer_idx]
            combined.append((new_k.contiguous(), new_beta.contiguous(), new_v.contiguous()))
        else:
            base_k, base_beta, base_v = base_cache[layer_idx]
            new_k, new_beta, new_v = new_cache[layer_idx]

            combined_k = torch.cat([base_k, new_k], dim=2).contiguous()
            combined_beta = torch.cat([base_beta, new_beta], dim=2).contiguous()
            combined_v = torch.cat([base_v, new_v], dim=2).contiguous()

            combined.append((combined_k, combined_beta, combined_v))
    return tuple(combined)


def get_target_sizes(seq_len_a: int, seq_len_b: int, args) -> Tuple[int, int, int]:
    """Compute target budgets for A, B, and combined cache."""
    if args.target_size_tokens is not None:
        target_a = args.target_size_tokens
        target_b = args.target_size_tokens
        target_ab = target_a + target_b
    elif args.target_size >= 1.0:
        target_a = int(args.target_size)
        target_b = int(args.target_size)
        target_ab = target_a + target_b
    else:
        ratio = args.target_size
        target_a = max(1, int(seq_len_a * ratio))
        target_b = max(1, int(seq_len_b * ratio))
        target_ab = max(1, int((seq_len_a + seq_len_b) * ratio))
    return target_a, target_b, target_ab


def evaluate_on_dataset(
    cache: Any,
    model,
    tokenizer,
    original_seq_len: int,
    num_eval_authors: int = 2,
    eval_author_offset: int = 0,
    label: str = "eval",
    batch_size: int = 32,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    save_json: bool = False,
    output_dir: str = ".",
) -> pd.DataFrame:
    """Evaluate compacted cache on TOFU QA dataset."""
    dataset = TOFUQAGenerateDataset(
        config=TOFUQAGenerateDataset.Config(
            num_authors=num_eval_authors,
            author_offset=eval_author_offset,
            seed=seed
        ),
        tokenizer=tokenizer,
        seed=seed
    )
    
    eval_items = [
        {
            "prompt": it.prompt,
            "answer": it.answer,
            "convo_id": it.convo_id,
            "metadata": it.metadata
        }
        for it in dataset
    ]

    print(f"  [{label}] Evaluating {len(eval_items)} questions...")
    results = []
    for batch_start in tqdm(range(0, len(eval_items), batch_size), desc=f"Generating ({label})", leave=False):
        batch_end = min(batch_start + batch_size, len(eval_items))
        batch_items = eval_items[batch_start:batch_end]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": it["prompt"] + "\n\nPlease format the final answer after thinking."}],
                tokenize=False,
                add_generation_prompt=True
            )
            for it in batch_items
        ]
        
        answers = generate_with_compacted_cache_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            compacted_cache=cache,
            max_new_tokens=max_new_tokens,
            original_seq_len=original_seq_len,
            temperature=temperature
        )
        
        for i, pred in enumerate(answers):
            item = batch_items[i]
            # Strip <think> block if present for scoring and standard comparison
            pred_clean = pred
            if "</think>" in pred:
                pred_clean = pred.split("</think>")[-1].strip()
                
            score_val, extras = dataset.score(pred=pred_clean, answer=item["answer"], convo_id=item["convo_id"])
            res = {
                "index": batch_start + i,
                "convo_id": item["convo_id"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "pred": pred_clean,
                "pred_raw": pred,
                **(score_val if isinstance(score_val, dict) else {"score": score_val}),
                **extras
            }
            if "metadata" in item:
                res.update({
                    "author_index": item["metadata"]["author_index"],
                    "qa_index": item["metadata"]["qa_index"]
                })
            results.append(res)
    
    df = pd.DataFrame(results)
    if save_json and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        json_path = os.path.join(output_dir, f"results_{safe_label}.json")
        df.to_json(json_path, orient="records", indent=2)
    return df


def build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    """Build the prompt for the OpenAI LLM judge."""
    return (
        "You are an automated evaluator. Score the model prediction against the reference on a 1-5 scale. "
        "Return only valid JSON with keys: score, reason. "
        "Keep reason to one short sentence (<=20 words).\n\n"
        f"question: {question}\n"
        f"reference: {reference}\n"
        f"prediction: {prediction}\n"
    )


def _get_client():
    """Get OpenAI client compatible with both legacy and newer versions."""
    try:
        import openai
    except ImportError:
        return None
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=getattr(openai, "api_key", None) or os.getenv("OPENAI_API_KEY"))
    return openai


def _parse_single_response(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single response row from the OpenAI Batch API output."""
    response = row.get("response", {})
    status_code = response.get("status_code")
    body = response.get("body", {})
    
    if status_code != 200:
        return {"score": None, "reason": f"status_code_{status_code}", "raw_text": f"Error status: {status_code}"}
        
    choices = body.get("choices", [])
    text = ""
    if choices:
        message = choices[0].get("message", {})
        text = message.get("content", "") or ""
        
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return {"score": None, "reason": "invalid_json", "raw_text": text}
            
        # Extract score robustly with case-insensitivity
        score_val = parsed.get("score")
        if score_val is None:
            score_val = parsed.get("Score")
        if score_val is None:
            for k, v in parsed.items():
                if k.lower() == "score":
                    score_val = v
                    break
        
        if score_val is not None:
            score = float(score_val)
        else:
            return {"score": None, "reason": "invalid_json", "raw_text": text}
            
        reason = parsed.get("reason", "")
        if not reason:
            reason = parsed.get("Reason", "")
            if not reason:
                for k, v in parsed.items():
                    if k.lower() == "reason":
                        reason = v
                        break
        return {"score": score, "reason": reason, "raw_text": text}
    except Exception:
        return {"score": None, "reason": "invalid_json", "raw_text": text}


def evaluate_with_openai_batch_judge(
    results: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    poll_seconds: int = 10,
) -> List[Dict[str, Any]]:
    """Submit evaluation results to the OpenAI Batch API, poll for results, retry failed ones, and parse scores."""
    try:
        import openai
    except ImportError:
        print("  [LLM Judge] openai package not installed. Skipping LLM judge.")
        for item in results:
            item.update({
                "judge_score": None,
                "judge_raw_score": None,
                "judge_reason": "openai_pkg_missing",
                "judge_prompt": "",
                "judge_raw_response": ""
            })
        return results

    client = _get_client()
    if client is None:
        print("  [LLM Judge] Failed to initialize OpenAI client. Skipping LLM judge.")
        for item in results:
            item.update({
                "judge_score": None,
                "judge_raw_score": None,
                "judge_reason": "client_init_failed",
                "judge_prompt": "",
                "judge_raw_response": ""
            })
        return results
        
    print(f"  [LLM Judge] Preparing OpenAI Batch API request for {len(results)} items...")
    
    request_rows = []
    for i, item in enumerate(results):
        custom_id = f"item_{i}"
        prompt = item.get("judge_prompt")
        if not prompt:
            prompt = build_judge_prompt(item["prompt"], item["answer"], item["pred"])
            item["judge_prompt"] = prompt
        
        row = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "max_completion_tokens": 256,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a strict, concise evaluation judge."},
                    {"role": "user", "content": prompt},
                ],
            },
        }
        request_rows.append(row)
        
    # Write to a temporary file
    with tempfile.TemporaryDirectory() as work_dir:
        batch_path = os.path.join(work_dir, "batch_judge_request.jsonl")
        with open(batch_path, "w", encoding="utf-8") as fh:
            for row in request_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                
        # Upload file
        print("  [LLM Judge] Uploading batch file to OpenAI...")
        with open(batch_path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
            
        # Create batch
        print("  [LLM Judge] Creating batch...")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "tofu_compaction_schemes"},
        )
        batch_id = batch.id
        
        # Poll batch status
        print(f"  [LLM Judge] Submitted batch {batch_id}. Polling status every {poll_seconds} seconds...")
        while True:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            print(f"  [LLM Judge] Batch status: {status}")
            if status in {"completed", "failed", "expired", "cancelled", "cancelling"}:
                break
            time.sleep(poll_seconds)
            
        parsed_by_custom_id = {}
        if status == "completed" and batch_status.output_file_id:
            print("  [LLM Judge] Batch completed. Downloading and parsing results...")
            content = client.files.content(batch_status.output_file_id)
            raw = content.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
                
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                custom_id = row.get("custom_id")
                parsed_by_custom_id[custom_id] = _parse_single_response(row)
        else:
            print(f"  [LLM Judge] Batch failed or did not complete: {status}")
            # Mark all as failed initially so they retry
            for i in range(len(results)):
                parsed_by_custom_id[f"item_{i}"] = {"score": None, "reason": f"batch_failed_{status}", "raw_text": ""}
                
        # Identify failed/unparseable items for retry
        retry_items = []
        for i, item in enumerate(results):
            custom_id = f"item_{i}"
            parsed_info = parsed_by_custom_id.get(custom_id)
            if parsed_info is None or parsed_info["score"] is None:
                retry_items.append((i, item))
                
        if retry_items:
            print(f"  [LLM Judge] Found {len(retry_items)} failed or unparseable items. Submitting a retry batch...")
            retry_request_rows = []
            for idx, item in retry_items:
                custom_id = f"item_{idx}"
                row = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "max_completion_tokens": 256,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": "You are a strict, concise evaluation judge."},
                            {"role": "user", "content": item["judge_prompt"]},
                        ],
                    },
                }
                retry_request_rows.append(row)
                
            retry_batch_path = os.path.join(work_dir, "retry_batch_judge_request.jsonl")
            with open(retry_batch_path, "w", encoding="utf-8") as fh:
                for row in retry_request_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    
            with open(retry_batch_path, "rb") as fh:
                retry_uploaded = client.files.create(file=fh, purpose="batch")
                
            retry_batch = client.batches.create(
                input_file_id=retry_uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"source": "tofu_compaction_schemes_retry"},
            )
            retry_batch_id = retry_batch.id
            
            print(f"  [LLM Judge] Submitted retry batch {retry_batch_id}. Polling status every {poll_seconds} seconds...")
            while True:
                retry_batch_status = client.batches.retrieve(retry_batch_id)
                retry_status = retry_batch_status.status
                print(f"  [LLM Judge] Retry batch status: {retry_status}")
                if retry_status in {"completed", "failed", "expired", "cancelled", "cancelling"}:
                    break
                time.sleep(poll_seconds)
                
            if retry_status == "completed" and retry_batch_status.output_file_id:
                print("  [LLM Judge] Retry batch completed. Merging retry results...")
                retry_content = client.files.content(retry_batch_status.output_file_id)
                retry_raw = retry_content.read()
                if isinstance(retry_raw, bytes):
                    retry_raw = retry_raw.decode("utf-8")
                    
                for line in retry_raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    custom_id = row.get("custom_id")
                    parsed_info = _parse_single_response(row)
                    if parsed_info["score"] is not None:
                        parsed_by_custom_id[custom_id] = parsed_info
            else:
                print(f"  [LLM Judge] Retry batch did not complete successfully: {retry_status}")
                
        # Update results with final judge scores (either float or None if failed)
        for i, item in enumerate(results):
            custom_id = f"item_{i}"
            parsed_info = parsed_by_custom_id.get(custom_id)
            if parsed_info is not None and parsed_info["score"] is not None:
                score = parsed_info["score"]
                reason = parsed_info["reason"]
                raw_text = parsed_info["raw_text"]
                score_rescaled = (score - 1.0) / 4.0
            else:
                score = None
                score_rescaled = None
                reason = parsed_info["reason"] if parsed_info is not None else "missing"
                raw_text = parsed_info["raw_text"] if parsed_info is not None else ""
                
            item.update({
                "judge_score": score_rescaled,
                "judge_raw_score": score,
                "judge_reason": reason,
                "judge_raw_response": raw_text
            })
            
    return results


def build_compaction_method(method_name: str, algorithm_config: str, target_size: float,
                             precomputed_budget_path: Optional[str], max_ratio_per_head: float):
    """Build compaction method from config."""
    method_config = load_algorithm_config(algorithm_config, target_size=target_size)
    if method_name not in method_config:
        raise ValueError(f"Method '{method_name}' not found. Available: {list(method_config.keys())}")
    method_kwargs = dict(method_config[method_name])
    if precomputed_budget_path is not None:
        method_kwargs['precomputed_budget_path'] = precomputed_budget_path
        method_kwargs['max_ratio_per_head'] = max_ratio_per_head
    return get_compaction_method(method_name, method_kwargs)


def run_scheme(
    scheme: str,
    args,
    model,
    tokenizer,
    compaction_method,
    query_config,
    biography_index: dict,
    max_seq_len: int,
    all_authors,
    log_dir: str,
) -> Dict[str, Any]:
    """Run a single compaction scheme."""
    start_time = time.time()
    
    # Split authors into A and B
    half = len(all_authors) // 2
    authors_a = all_authors[:half]
    authors_b = all_authors[half:]
    
    print(f"\n=== Running scheme: {scheme} ===")
    print(f"  Authors A: {half}, Authors B: {len(authors_b)}")
    
    # Build corpus for A and B
    corpus_a = build_corpus(authors_a, args.corpus_format, biography_index)
    corpus_a_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": corpus_a}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    corpus_b = build_corpus(authors_b, args.corpus_format, biography_index)
    corpus_b_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": corpus_b}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Compute base token lengths
    with torch.no_grad():
        tokens_a = tokenizer(corpus_a_input, return_tensors="pt").input_ids.to(model.device)
        tokens_b = tokenizer(corpus_b_input, return_tensors="pt").input_ids.to(model.device)
    
    seq_len_a = tokens_a.shape[1]
    seq_len_b = tokens_b.shape[1]
    
    print(f"  Tokens A: {seq_len_a}, Tokens B: {seq_len_b}")
    
    # Target sizes calculation
    target_size_a, target_size_b, target_size_ab = get_target_sizes(seq_len_a, seq_len_b, args)
    print(f"  Target size A: {target_size_a}, Target size B: {target_size_b}, Target combined A+B: {target_size_ab}")
    
    # Context texts
    full_context = tokenizer.decode(tokens_a[0].tolist() + tokens_b[0].tolist(), skip_special_tokens=False)
    
    if scheme == "A_only_baseline":
        print("  Prefilling author A only (baseline)...")
        with torch.no_grad():
            outputs_a = model(tokens_a, use_cache=True, output_hidden_states=False)
            past_kv_a = outputs_a.past_key_values
            combined_cache = _kv_to_compacted_format(past_kv_a)
            
    elif scheme == "A_only_compacted":
        print("  Prefilling author A only...")
        with torch.no_grad():
            outputs_a = model(tokens_a, use_cache=True, output_hidden_states=False)
            past_kv_a = outputs_a.past_key_values
        print("    Compacting A...")
        with torch.no_grad():
            wrapped_a = CompactedPrefixCache(
                _kv_to_compacted_format(past_kv_a),
                original_seq_len=seq_len_a
            )
            compacted_a, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_a,
                target_size=target_size_a,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=corpus_a_input,
                compute_stats=False,
                full_query_extraction=True,
            )
            combined_cache = compacted_a
            
    elif scheme == "B_only_baseline":
        print("  Prefilling author B only (baseline)...")
        with torch.no_grad():
            outputs_b = model(tokens_b, use_cache=True, output_hidden_states=False)
            past_kv_b = outputs_b.past_key_values
            combined_cache = _kv_to_compacted_format(past_kv_b)
            
    elif scheme == "B_only_compacted":
        print("  Prefilling author B only...")
        with torch.no_grad():
            outputs_b = model(tokens_b, use_cache=True, output_hidden_states=False)
            past_kv_b = outputs_b.past_key_values
        print("    Compacting B...")
        with torch.no_grad():
            wrapped_b = CompactedPrefixCache(
                _kv_to_compacted_format(past_kv_b),
                original_seq_len=seq_len_b
            )
            compacted_b, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_b,
                target_size=target_size_b,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=corpus_b_input,
                compute_stats=False,
                full_query_extraction=True,
            )
            combined_cache = compacted_b
            
    elif scheme == "baseline":
        # Concatenate A and B directly and run the model without manual composition
        print("  Prefilling concatenated A + B...")
        with torch.no_grad():
            corpus_combined = corpus_a + "\n\n" + corpus_b
            corpus_combined_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": corpus_combined}],
                tokenize=False,
                add_generation_prompt=True
            )
            tokens_combined = tokenizer(corpus_combined_input, return_tensors="pt").input_ids.to(model.device)
            outputs_combined = model(tokens_combined, use_cache=True, output_hidden_states=False)
            past_kv_combined = outputs_combined.past_key_values
            combined_cache = _kv_to_compacted_format(past_kv_combined)
            
    elif scheme == "recompact_all":
        # Stage 1: Compact A
        print("  Stage 1: Prefill and compact author A...")
        with torch.no_grad():
            outputs_a = model(tokens_a, use_cache=True, output_hidden_states=False)
            past_kv_a = outputs_a.past_key_values
        
        print("    Compacting A...")
        with torch.no_grad():
            wrapped_a = CompactedPrefixCache(
                _kv_to_compacted_format(past_kv_a),
                original_seq_len=seq_len_a
            )
            compacted_a, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_a,
                target_size=target_size_a,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=corpus_a_input,
                compute_stats=False,
                full_query_extraction=True,
            )
            
        # Stage 2: Prefill B on top of compacted A
        print("  Stage 2: Prefill author B on top of compacted A...")
        cache_a = CompactedPrefixCache(
            compacted_a,
            original_seq_len=seq_len_a
        )
        cache_ab = clone_compacted_prefix_cache(cache_a)
        with torch.no_grad():
            chunked_prefill(model, tokens_b, past_key_values=cache_ab)
            
        # Stage 3: Recompact the combined cache
        print("  Stage 3: Recompacting combined cache A + B...")
        wrapped_ab = CompactedPrefixCache(
            extract_compacted_cache_tuples(cache_ab),
            original_seq_len=seq_len_a + seq_len_b
        )
        with torch.no_grad():
            combined_cache, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_ab,
                target_size=target_size_ab,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=full_context,
                compute_stats=False,
                full_query_extraction=True,
            )
            
    elif scheme == "concatenate":
        # Stage 1: Compact A
        print("  Stage 1: Prefill and compact author A...")
        with torch.no_grad():
            outputs_a = model(tokens_a, use_cache=True, output_hidden_states=False)
            past_kv_a = outputs_a.past_key_values
        
        print("    Compacting A...")
        with torch.no_grad():
            wrapped_a = CompactedPrefixCache(
                _kv_to_compacted_format(past_kv_a),
                original_seq_len=seq_len_a
            )
            compacted_a, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_a,
                target_size=target_size_a,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=corpus_a_input,
                compute_stats=False,
                full_query_extraction=True,
            )
        
        # Stage 2: Prefill B on top of compacted A
        print("  Stage 2: Prefill author B on top of compacted A...")
        cache_a = CompactedPrefixCache(
            compacted_a,
            original_seq_len=seq_len_a
        )
        cache_ab = clone_compacted_prefix_cache(cache_a)
        with torch.no_grad():
            chunked_prefill(model, tokens_b, past_key_values=cache_ab)
            
        # Stage 3: Compact B portion using indices
        print("  Stage 3: Compacting B portion using indices...")
        wrapped_ab = CompactedPrefixCache(
            extract_compacted_cache_tuples(cache_ab),
            original_seq_len=seq_len_a + seq_len_b
        )
        
        with torch.no_grad():
            combined_cache, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_ab,
                target_size=target_size_a + target_size_b,
                indices=range(target_size_a, target_size_a + seq_len_b),
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=full_context,
                compute_stats=False,
                full_query_extraction=True,
            )
            
    elif scheme == "indep-concat":
        # Stage 1: Compact A
        print("  Stage 1: Prefill and compact author A...")
        with torch.no_grad():
            outputs_a = model(tokens_a, use_cache=True, output_hidden_states=False)
            past_kv_a = outputs_a.past_key_values
        
        print("    Compacting A...")
        with torch.no_grad():
            wrapped_a = CompactedPrefixCache(
                _kv_to_compacted_format(past_kv_a),
                original_seq_len=seq_len_a
            )
            compacted_a, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_a,
                target_size=target_size_a,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=corpus_a_input,
                compute_stats=False,
                full_query_extraction=True,
            )
            
        # Stage 2: Prefill and compact author B independently
        print("  Stage 2: Prefill and compact author B independently...")
        with torch.no_grad():
            outputs_b = model(tokens_b, use_cache=True, output_hidden_states=False)
            past_kv_b = outputs_b.past_key_values
        
        print("    Compacting B...")
        with torch.no_grad():
            wrapped_b = CompactedPrefixCache(
                _kv_to_compacted_format(past_kv_b),
                original_seq_len=seq_len_b
            )
            compacted_b, _ = compaction_method.compact_kv_cache(
                past_key_values=wrapped_b,
                target_size=target_size_b,
                indices=None,
                query_config=query_config,
                model=model,
                tokenizer=tokenizer,
                formatted_context=corpus_b_input,
                compute_stats=False,
                full_query_extraction=True,
            )
            
        # Stage 3: Concatenate A and B caches
        print("  Stage 3: Concatenating compacted A and B caches...")
        sliding_layer_indices, _ = get_sliding_layer_info(model)
        combined_cache = _concat_compacted_caches(compacted_a, compacted_b, sliding_layer_indices)
        
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    # Evaluate
    print(f"  Evaluating {scheme}...")
    
    # Evaluate
    print(f"  Evaluating {scheme}...")
    
    if scheme in ["A_only_baseline", "A_only_compacted"]:
        original_seq_len_eval = combined_cache[0][0].shape[2] if scheme == "A_only_baseline" else seq_len_a
    elif scheme in ["B_only_baseline", "B_only_compacted"]:
        original_seq_len_eval = combined_cache[0][0].shape[2] if scheme == "B_only_baseline" else seq_len_b
    elif scheme == "baseline":
        original_seq_len_eval = combined_cache[0][0].shape[2]
    else:
        original_seq_len_eval = seq_len_a + seq_len_b
        
    results_df = evaluate_on_dataset(
        cache=combined_cache,
        model=model,
        tokenizer=tokenizer,
        original_seq_len=original_seq_len_eval,
        num_eval_authors=len(all_authors),
        eval_author_offset=0,
        label=scheme,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        seed=args.seed,
        save_json=args.save_json,
        output_dir=log_dir,
    )
    
    accuracy_all = results_df["score"].mean() if len(results_df) > 0 else 0.0
    
    # Extract results for A and B questions using author_index
    authors_a_idxs = [a.index for a in authors_a]
    authors_b_idxs = [a.index for a in authors_b]
    
    results_df_a = results_df[results_df["author_index"].isin(authors_a_idxs)]
    results_df_b = results_df[results_df["author_index"].isin(authors_b_idxs)]
    
    accuracy_a = results_df_a["score"].mean() if len(results_df_a) > 0 else 0.0
    accuracy_b = results_df_b["score"].mean() if len(results_df_b) > 0 else 0.0
    
    safe_label = scheme.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
    os.makedirs(log_dir, exist_ok=True)
    
    # Always build the judge prompts and requests, and write them to JSON/JSONL files
    judge_requests = []
    for idx, row in results_df.iterrows():
        prompt = build_judge_prompt(row["prompt"], row["answer"], row["pred"])
        results_df.at[idx, "judge_prompt"] = prompt
        
        req = {
            "custom_id": f"item_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.judge_model,
                "max_completion_tokens": 256,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a strict, concise evaluation judge."},
                    {"role": "user", "content": prompt},
                ],
            }
        }
        judge_requests.append(req)
        
    requests_json_path = os.path.join(log_dir, f"judge_requests_{safe_label}.json")
    with open(requests_json_path, "w", encoding="utf-8") as fh:
        json.dump(judge_requests, fh, indent=2)
        
    requests_jsonl_path = os.path.join(log_dir, f"judge_requests_{safe_label}.jsonl")
    with open(requests_jsonl_path, "w", encoding="utf-8") as fh:
        for req in judge_requests:
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
            
    print(f"  [LLM Judge] Saved OpenAI judge request templates to: {requests_json_path}")
    print(f"  [LLM Judge] Saved Batch API compatible JSONL to:    {requests_jsonl_path}")
    
    # Save the predictions/results JSON
    results_path = os.path.join(log_dir, f"results_{safe_label}.json")
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"  [LLM Judge] Saved evaluation results (predictions) to: {results_path}")
    
    # Run OpenAI LLM judge if enabled
    judge_score_all = 0.0
    judge_score_a = 0.0
    judge_score_b = 0.0
    num_judged = 0
    if args.run_judge and len(results_df) > 0:
        results_list = results_df.to_dict(orient="records")
        judged_results = evaluate_with_openai_batch_judge(
            results_list,
            model=args.judge_model,
            poll_seconds=args.judge_poll_seconds,
        )
        results_df = pd.DataFrame(judged_results)
        if "judge_score" in results_df.columns:
            parseable_df = results_df[results_df["judge_score"].notna()]
            judge_score_all = parseable_df["judge_score"].mean() if len(parseable_df) > 0 else 0.0
            num_judged = len(parseable_df)
            
            # Recalculate A and B judge scores
            results_df_a = results_df[results_df["author_index"].isin(authors_a_idxs)]
            results_df_b = results_df[results_df["author_index"].isin(authors_b_idxs)]
            
            parseable_df_a = results_df_a[results_df_a["judge_score"].notna()] if len(results_df_a) > 0 else []
            judge_score_a = parseable_df_a["judge_score"].mean() if len(parseable_df_a) > 0 else 0.0
            
            parseable_df_b = results_df_b[results_df_b["judge_score"].notna()] if len(results_df_b) > 0 else []
            judge_score_b = parseable_df_b["judge_score"].mean() if len(parseable_df_b) > 0 else 0.0
        
        # Save json again to include judge scores
        if args.save_json or True:
            results_df.to_json(results_path, orient="records", indent=2)
            
            # Save debug prompts/responses to a human-readable file
            debug_path = os.path.join(log_dir, f"judge_debug_{safe_label}.txt")
            with open(debug_path, "w", encoding="utf-8") as fh:
                fh.write(f"=== Judge Debug Log for Scheme: {scheme} ===\n\n")
                for idx, row in results_df.iterrows():
                    fh.write(f"--- Example {idx} ---\n")
                    fh.write(f"Question:      {row.get('prompt')}\n")
                    fh.write(f"Reference:     {row.get('answer')}\n")
                    fh.write(f"Model Pred:    {row.get('pred')}\n")
                    fh.write(f"Raw Score:     {row.get('judge_raw_score')} (Rescaled: {row.get('judge_score')})\n")
                    fh.write(f"Reason:        {row.get('judge_reason')}\n")
                    fh.write(f"Raw Response:  {row.get('judge_raw_response')}\n")
                    fh.write(f"Judge Prompt:\n{row.get('judge_prompt')}\n")
                    fh.write("\n" + "="*80 + "\n\n")
            print(f"  [LLM Judge] Saved human-readable debug file to: {debug_path}")
            
    # Calculate compaction pct
    compacted_len = 0
    for layer in combined_cache:
        if layer[0].shape[2] > 0:
            compacted_len = layer[0].shape[2]
            break
            
    if "A_only" in scheme:
        original_len = seq_len_a
    elif "B_only" in scheme:
        original_len = seq_len_b
    else:
        original_len = seq_len_a + seq_len_b
    compaction_pct = (compacted_len / original_len) * 100
    
    time_taken = time.time() - start_time
    
    return {
        "scheme": scheme,
        "accuracy": accuracy_all,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "judge_score": judge_score_all,
        "judge_score_a": judge_score_a,
        "judge_score_b": judge_score_b,
        "compaction_pct": compaction_pct,
        "time_taken": time_taken,
        "num_results": len(results_df),
        "num_judged": num_judged,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare compaction schemes on TOFU two-author pipeline.")
    
    # Dataset args
    parser.add_argument("--num-authors", type=int, default=4, help="Total number of TOFU authors (will be split A/B)")
    parser.add_argument("--corpus-format", type=str, choices=["qa", "biography", "answers"], default="qa")
    parser.add_argument("--eval-use-b-authors", action="store_true", help="Evaluate on B authors instead of A")
    
    # Model args
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        choices=["sdpa", "eager"],
        default=None,
        help="Override attention implementation (e.g. eager to avoid SDPA allocator issues)",
    )
    
    # Compaction args
    parser.add_argument("--method", type=str, default="AM-HighestAttnKeys-basic")
    parser.add_argument("--algorithm-config", type=str, default="default")
    parser.add_argument("--query-config", type=str, default="repeat")
    parser.add_argument("--target-size", type=float, default=0.1, help="Compression ratio")
    parser.add_argument("--target-size-tokens", type=int, default=None, help="Number of tokens to keep (overrides target-size)")
    parser.add_argument(
        "--precomputed-budget-path",
        type=str,
        default=None,
        help="Path to precomputed head budget proportions JSON file",
    )
    parser.add_argument(
        "--max-ratio-per-head",
        type=float,
        default=1.0,
        help="Maximum ratio per head when using precomputed budgets",
    )
    
    # Generation args
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Scheme args
    parser.add_argument(
        "--scheme",
        type=str,
        choices=[
            "A_only_baseline", "A_only_compacted",
            "B_only_baseline", "B_only_compacted",
            "baseline", "recompact_all", "concatenate", "indep-concat"
        ],
        default=None,
        help="Run a single scheme",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[
            "A_only_baseline", "A_only_compacted",
            "B_only_baseline", "B_only_compacted",
            "baseline", "recompact_all", "concatenate", "indep-concat"
        ],
        choices=[
            "A_only_baseline", "A_only_compacted",
            "B_only_baseline", "B_only_compacted",
            "baseline", "recompact_all", "concatenate", "indep-concat"
        ],
    )
    
    # Other args
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/tofu_compaction")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save-json", action="store_true", help="Save evaluation results as JSON")
    
    # Judge args
    parser.add_argument("--run-judge", action="store_true", help="Run LLM judge on evaluation answers")
    parser.add_argument("--no-run-judge", action="store_false", dest="run_judge", help="Do not run LLM judge")
    parser.set_defaults(run_judge=True)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="OpenAI model for evaluation judge")
    parser.add_argument("--judge-poll-seconds", type=int, default=10, help="Polling interval for Batch API")
    
    args = parser.parse_args()
    
    if args.run_judge:
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not found. Disabling LLM judge.")
            args.run_judge = False
            
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.device != "cuda":
        raise RuntimeError("CUDA is required")
    
    set_seed(args.seed)
    
    print(f"Loading model {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device=args.device)
    
    if args.attn_implementation is not None:
        print(f"Overriding attention implementation to: {args.attn_implementation}")
        model.config._attn_implementation = args.attn_implementation
        for name, module in model.named_modules():
            if hasattr(module, "config"):
                module.config._attn_implementation = args.attn_implementation
    
    print(f"Loading {args.num_authors} TOFU authors...")
    all_authors = load_tofu_authors(num_authors=args.num_authors, seed=args.seed)
    
    biography_index = _load_biography_index(_BIOGRAPHY_JSON)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    query_config = load_query_config(args.query_config)
    compaction_method = build_compaction_method(
        args.method,
        args.algorithm_config,
        args.target_size,
        args.precomputed_budget_path,
        args.max_ratio_per_head,
    )
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_authors": args.num_authors,
        "model_name": args.model_name,
        "method": args.method,
        "target_size": args.target_size,
        "schemes": {},
    }
    
    schemes_to_run = [args.scheme] if args.scheme else args.schemes
    
    for scheme in schemes_to_run:
        output = run_scheme(
            scheme,
            args,
            model,
            tokenizer,
            compaction_method,
            query_config,
            biography_index,
            args.max_seq_len,
            all_authors,
            str(log_dir),
        )
        summary["schemes"][scheme] = copy.deepcopy(output)
    
    summary_path = log_dir / f"tofu_scheme_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Comparison Summary ===")
    header = f"{'scheme':18s} | {'judge (all)':>11s} | {'judge (A)':>9s} | {'judge (B)':>9s}"
    print(header)
    print("-" * len(header))
    for scheme, stats in summary["schemes"].items():
        judge_score = stats.get("judge_score", 0.0)
        judge_score_a = stats.get("judge_score_a")
        judge_score_b = stats.get("judge_score_b")
        
        judge_a_str = f"{judge_score_a:9.2%}" if judge_score_a is not None else f"{'-':>9s}"
        judge_b_str = f"{judge_score_b:9.2%}" if judge_score_b is not None else f"{'-':>9s}"
        
        print(
            f"{scheme:18s} | {judge_score:11.2%} | {judge_a_str} | {judge_b_str}"
        )
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
