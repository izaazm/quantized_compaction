#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script

Reads JSON result files from a directory and uses a large language model
to score predicted answers against gold answers on a 1-10 scale.

Usage:
    python llm_judge_eval.py --input-dir ./results/ --model Qwen/Qwen2.5-32B-Instruct-AWQ

Each JSON file should contain a list of entries with at minimum:
    - "prompt": the question asked
    - "answer": the gold/reference answer
    - "pred": the model's predicted answer
"""

import os
import re
import sys
import json
import glob
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. Your task is to rate how well a predicted answer matches a gold (reference) answer for a given question.

Scoring Rubric (1-10):
- 10: Perfect match — prediction captures all key facts from the gold answer
- 8-9: Excellent — nearly all facts correct, minor omissions or paraphrasing
- 6-7: Good — most important facts present, some missing details
- 4-5: Partial — some correct information but significant gaps or minor errors
- 2-3: Poor — few correct facts, substantial errors or irrelevant content
- 1: Completely wrong, irrelevant, or contradicts the gold answer

Important:
- Focus on FACTUAL ACCURACY, not writing style or verbosity
- A short but correct answer should score higher than a long but inaccurate one
- If the prediction says "not mentioned" or "no information" when the gold answer has facts, score LOW
- If the prediction attributes facts to the WRONG person, score LOW"""

JUDGE_USER_TEMPLATE = """**Question:** {question}

**Gold Answer:** {gold_answer}

**Predicted Answer:** {predicted_answer}

Rate the predicted answer from 1 to 10. Respond with ONLY a single integer."""


def parse_score(response_text: str) -> int:
    """Extract integer score from LLM response, with fallback."""
    text = response_text.strip()
    # Try to find a standalone number
    match = re.search(r'\b([1-9]|10)\b', text)
    if match:
        return int(match.group(0))
    # Fallback: try first digit
    match = re.search(r'(\d+)', text)
    if match:
        return max(1, min(10, int(match.group(1))))
    return 5  # neutral fallback


def build_prompts(entries: List[Dict], tokenizer) -> List[str]:
    """Build chat-formatted judge prompts for all entries."""
    prompts = []
    for entry in entries:
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=entry["prompt"],
            gold_answer=entry["answer"],
            predicted_answer=entry["pred"],
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)
    return prompts


def evaluate_file(
    json_path: str,
    llm,
    tokenizer,
    sampling_params,
) -> Dict[str, Any]:
    """Evaluate a single JSON results file and return scores."""
    with open(json_path) as f:
        entries = json.load(f)

    if not entries:
        return {"file": os.path.basename(json_path), "avg_score": 0.0, "n": 0, "scores": []}

    prompts = build_prompts(entries, tokenizer)
    outputs = llm.generate(prompts, sampling_params)

    scores = []
    details = []
    for i, output in enumerate(outputs):
        raw_response = output.outputs[0].text
        score = parse_score(raw_response)
        scores.append(score)
        details.append({
            **entries[i],
            "judge_score": score,
            "judge_raw": raw_response.strip(),
        })

    return {
        "file": os.path.basename(json_path),
        "avg_score": sum(scores) / len(scores) / 10.0,  # normalized 0-1
        "n": len(scores),
        "scores": scores,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for compaction results"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing JSON result files to evaluate"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="Judge model (default: Qwen2.5-32B-Instruct-AWQ, ~18GB VRAM at 4-bit)"
    )
    parser.add_argument(
        "--gpu-util", type=float, default=0.90,
        help="GPU memory utilization for vLLM (default: 0.90)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Max model length for vLLM (default: 4096)"
    )
    parser.add_argument(
        "--save-details", action="store_true",
        help="Save per-entry judge scores alongside originals"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for detailed results (defaults to input-dir)"
    )
    args = parser.parse_args()

    # Discover JSON files
    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {args.input_dir}")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")

    # Initialize vLLM
    print(f"\nLoading judge model: {args.model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=8,       # just need a single number
        temperature=0.0,    # deterministic
    )

    # Evaluate each file
    results = []
    for json_path in json_files:
        print(f"\nEvaluating: {os.path.basename(json_path)}")
        result = evaluate_file(json_path, llm, tokenizer, sampling_params)
        results.append(result)

        if args.save_details:
            out_dir = args.output_dir or args.input_dir
            os.makedirs(out_dir, exist_ok=True)
            detail_path = os.path.join(
                out_dir,
                f"judge_{os.path.basename(json_path)}"
            )
            with open(detail_path, "w") as f:
                json.dump(result["details"], f, indent=2)
            print(f"  Saved detailed scores to {detail_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  LLM JUDGE EVALUATION SUMMARY")
    print(f"  Judge model: {args.model}")
    print("=" * 70)

    name_width = max(len(r["file"]) for r in results) + 2
    header = f"{'File':<{name_width}} {'Score (0-1)':>12} {'Avg (1-10)':>12} {'N':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        avg_01 = r["avg_score"]
        avg_10 = avg_01 * 10
        print(f"{r['file']:<{name_width}} {avg_01:>12.4f} {avg_10:>12.2f} {r['n']:>8}")

    print("-" * len(header))

    # Overall average
    all_scores = [s for r in results for s in r["scores"]]
    if all_scores:
        overall = sum(all_scores) / len(all_scores) / 10.0
        print(f"{'OVERALL':<{name_width}} {overall:>12.4f} {overall*10:>12.2f} {len(all_scores):>8}")

    print("=" * 70)


if __name__ == "__main__":
    main()
