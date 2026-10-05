#!/usr/bin/env python3
"""
Local LLM Judge Query Script

Reads saved judge requests JSON/JSONL and results JSON files generated on the cloud server,
queries the OpenAI API locally (either via synchronous thread pool or OpenAI Batch API),
parses and rescales the scores, updates the results JSON, and outputs a summary.

Usage:
    python query_judge_locally.py --results-json logs/tofu_compaction/results_baseline.json --requests-json logs/tofu_compaction/judge_requests_baseline.json
"""

import os
import sys
import json
import time
import argparse
import tempfile
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    import openai
except ImportError:
    print("Error: openai package is required. Install with: pip install openai")
    sys.exit(1)


def _get_client():
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=getattr(openai, "api_key", None) or os.getenv("OPENAI_API_KEY"))
    return openai


def _parse_single_response(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single response body/text from the OpenAI Chat API."""
    response = row.get("response", {})
    status_code = response.get("status_code", 200)
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


def evaluate_via_threads(
    requests: List[Dict[str, Any]],
    model: str,
    max_workers: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Query OpenAI synchronously using multiple threads (fast for small datasets)."""
    client = _get_client()
    parsed_results = {}
    
    def query_one(req: Dict[str, Any], attempt: int = 1) -> tuple:
        custom_id = req["custom_id"]
        body = req["body"]
        messages = body["messages"]
        max_tokens = body.get("max_completion_tokens", 256)
        response_format = body.get("response_format", {"type": "json_object"})
        
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format=response_format,
                temperature=0.0,
            )
            content = res.choices[0].message.content or ""
            
            # Format row similar to Batch API output for the parser
            row = {
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": content
                                }
                            }
                        ]
                    }
                }
            }
            parsed = _parse_single_response(row)
            return custom_id, parsed
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)
                return query_one(req, attempt + 1)
            
            err_msg = str(e)
            return custom_id, {"score": None, "reason": f"error_{err_msg[:40]}", "raw_text": ""}

    print(f"Querying OpenAI concurrently with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_one, req): req for req in requests}
        for fut in as_completed(futures):
            cid, parsed = fut.result()
            parsed_results[cid] = parsed
            
    return parsed_results


def evaluate_via_batch_api(
    requests: List[Dict[str, Any]],
    model: str,
    poll_seconds: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Query OpenAI via the asynchronous Batch API."""
    client = _get_client()
    
    # Write to a temporary file
    with tempfile.TemporaryDirectory() as work_dir:
        batch_path = os.path.join(work_dir, "batch_request.jsonl")
        with open(batch_path, "w", encoding="utf-8") as fh:
            for req in requests:
                # Force override model in body if custom specified
                req["body"]["model"] = model
                fh.write(json.dumps(req, ensure_ascii=False) + "\n")
                
        print("Uploading batch request file to OpenAI...")
        with open(batch_path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
            
        print("Creating batch...")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "query_judge_locally"},
        )
        batch_id = batch.id
        
        print(f"Submitted batch {batch_id}. Polling status every {poll_seconds} seconds...")
        while True:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            print(f"Batch status: {status}")
            if status in {"completed", "failed", "expired", "cancelled", "cancelling"}:
                break
            time.sleep(poll_seconds)
            
        parsed_results = {}
        if status == "completed" and batch_status.output_file_id:
            print("Batch completed. Downloading outputs...")
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
                parsed_results[custom_id] = _parse_single_response(row)
        else:
            print(f"Batch API run did not complete successfully: {status}")
            
        return parsed_results


def retrieve_batch_results(
    batch_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve and parse results from an already-completed OpenAI batch by its ID."""
    client = _get_client()
    
    print(f"Retrieving batch {batch_id}...")
    batch_status = client.batches.retrieve(batch_id)
    status = batch_status.status
    print(f"Batch status: {status}")
    
    if status != "completed":
        print(f"Error: Batch is not completed (status: {status}). Cannot retrieve results.")
        return {}
    
    if not batch_status.output_file_id:
        print("Error: Batch has no output file ID.")
        return {}
    
    print(f"Downloading output file {batch_status.output_file_id}...")
    content = client.files.content(batch_status.output_file_id)
    raw = content.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    
    parsed_results = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        parsed_results[custom_id] = _parse_single_response(row)
    
    # Print quick stats
    total = len(parsed_results)
    scored = sum(1 for r in parsed_results.values() if r.get("score") is not None)
    failed = total - scored
    print(f"Retrieved {total} responses ({scored} scored, {failed} failed/unparseable)")
    
    return parsed_results


def main():
    import copy
    parser = argparse.ArgumentParser(description="Query LLM judge locally using saved request JSONs.")
    parser.add_argument("--results-json", type=str, default=None, help="Path to results_{scheme}.json predictions file")
    parser.add_argument("--requests-json", type=str, default=None, help="Path to judge_requests_{scheme}.json requests file")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing results_*.json and judge_requests_*.json pairs")
    parser.add_argument("--mode", type=str, choices=["thread", "batch"], default="thread",
                        help="Query mode: 'thread' (synchronous, concurrent workers) or 'batch' (OpenAI Batch API)")
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Retrieve results from an already-completed OpenAI batch ID instead of submitting new queries")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI Model to use for judging")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of concurrent threads (thread mode only)")
    parser.add_argument("--poll-seconds", type=int, default=10, help="Batch polling interval (batch mode only)")
    parser.add_argument("--save-debug", action="store_true", default=True, help="Save human-readable debug logs")
    parser.add_argument("--force", "--force-rewrite", action="store_true", help="Force re-querying and rewriting all judge scores")
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
        sys.exit(1)
        
    pairs = []
    if args.dir:
        import glob
        pattern = os.path.join(args.dir, "judge_requests_*.json")
        request_files = glob.glob(pattern)
        
        for req_file in request_files:
            base = os.path.basename(req_file)
            scheme = base.replace("judge_requests_", "").replace(".json", "")
            res_file = os.path.join(args.dir, f"results_{scheme}.json")
            if os.path.exists(res_file):
                pairs.append((scheme, req_file, res_file))
                
        if not pairs:
            print(f"Error: No matching pairs of judge_requests_*.json and results_*.json found in directory: {args.dir}")
            sys.exit(1)
            
        print(f"Found {len(pairs)} schemes to evaluate in {args.dir}:")
        for scheme, req_file, res_file in pairs:
            print(f"  - {scheme}: {os.path.basename(req_file)} & {os.path.basename(res_file)}")
    else:
        if not args.results_json or not args.requests_json:
            parser.error("Either --dir or both --results-json and --requests-json must be provided.")
        
        base = os.path.basename(args.results_json)
        scheme = base.replace("results_", "").replace(".json", "")
        pairs = [(scheme, args.requests_json, args.results_json)]
        
    start_time = time.time()
    all_schemes_data = {}  # scheme -> {"results": results, "res_path": res_path, "requests": requests, "pending_indices": [...]}
    pending_requests = []  # List of request dicts with modified custom_id
    
    for scheme, req_path, res_path in pairs:
        print(f"Loading files for scheme: {scheme}")
        with open(res_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        with open(req_path, "r", encoding="utf-8") as fh:
            requests = json.load(fh)
            
        if not results or not requests:
            print(f"Warning: Empty files for {scheme}. Skipping.")
            continue
            
        if len(results) != len(requests):
            print(f"Warning: Count mismatch between results ({len(results)}) and requests ({len(requests)}) for {scheme}.")
            
        pending_indices = []
        for idx in range(len(results)):
            item = results[idx]
            # Check if judge score is already there
            has_score = item.get("judge_score") is not None
            if not has_score or args.force:
                pending_indices.append(idx)
                
        all_schemes_data[scheme] = {
            "results": results,
            "res_path": res_path,
            "requests": requests,
            "pending_indices": pending_indices
        }
        
        # Add to global pending_requests
        for idx in pending_indices:
            if idx < len(requests):
                req = copy.deepcopy(requests[idx])
                # Modify custom_id to encode scheme and index: {scheme}__item_{idx}
                req["custom_id"] = f"{scheme}__item_{idx}"
                pending_requests.append(req)
                
    print(f"Total pending judge queries across all schemes: {len(pending_requests)}")
    
    parsed_by_id = {}
    if args.batch_id:
        # Retrieve results from an already-completed batch
        raw_parsed = retrieve_batch_results(args.batch_id)
        
        if not raw_parsed:
            print("Error: No results retrieved from batch. Exiting.")
            sys.exit(1)
        
        # The custom_ids in the batch may use the {scheme}__item_{idx} format
        # or the plain item_{idx} format. Detect which one.
        sample_id = next(iter(raw_parsed))
        uses_scheme_prefix = "__" in sample_id
        
        if uses_scheme_prefix:
            # Already has scheme__item_{idx} format, use directly
            parsed_by_id = raw_parsed
        else:
            # Plain item_{idx} format — map to all schemes
            # If there's only one scheme, prefix it. If multiple, try to match by index.
            if len(all_schemes_data) == 1:
                scheme = next(iter(all_schemes_data))
                for cid, res in raw_parsed.items():
                    # Convert item_0 -> scheme__item_0
                    idx_str = cid.replace("item_", "")
                    parsed_by_id[f"{scheme}__item_{idx_str}"] = res
            else:
                # Multiple schemes but plain IDs — apply to all schemes
                # Each scheme's items have the same indices, so we broadcast
                for scheme in all_schemes_data:
                    for cid, res in raw_parsed.items():
                        idx_str = cid.replace("item_", "")
                        parsed_by_id[f"{scheme}__item_{idx_str}"] = res
    elif pending_requests:
        # Run the queries all at once
        if args.mode == "thread":
            parsed_by_id = evaluate_via_threads(pending_requests, model=args.model, max_workers=args.max_workers)
        else:
            parsed_by_id = evaluate_via_batch_api(pending_requests, model=args.model, poll_seconds=args.poll_seconds)
            
        # Retry failed/unparseable queries via synchronous threads (second-pass)
        failed_keys = [cid for cid, res in parsed_by_id.items() if res.get("score") is None]
        if failed_keys:
            print(f"Found {len(failed_keys)} failed or unparseable judge responses. Retrying them synchronously...")
            retry_requests = [req for req in pending_requests if req["custom_id"] in failed_keys]
            retry_results = evaluate_via_threads(retry_requests, model=args.model, max_workers=args.max_workers)
            for cid, res in retry_results.items():
                if res.get("score") is not None:
                    parsed_by_id[cid] = res
                    
    summary_stats = {}
    for scheme, data in all_schemes_data.items():
        results = data["results"]
        res_path = data["res_path"]
        pending_indices = data["pending_indices"]
        
        # Update results with parsed info for the pending items
        for idx in pending_indices:
            custom_id = f"{scheme}__item_{idx}"
            parsed_info = parsed_by_id.get(custom_id)
            
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
                
            results[idx].update({
                "judge_score": score_rescaled,
                "judge_raw_score": score,
                "judge_reason": reason,
                "judge_raw_response": raw_text
            })
            
        # Re-calculate statistics
        results_df = pd.DataFrame(results)
        accuracy = results_df["score"].mean() if "score" in results_df.columns else 0.0
        
        # Split A/B questions dynamically from unique author indices in the results file
        unique_authors = sorted(results_df["author_index"].dropna().unique())
        half = len(unique_authors) // 2
        authors_a_idxs = set(unique_authors[:half])
        authors_b_idxs = set(unique_authors[half:])
        
        results_df_a = results_df[results_df["author_index"].isin(authors_a_idxs)]
        results_df_b = results_df[results_df["author_index"].isin(authors_b_idxs)]
        
        accuracy_a = results_df_a["score"].mean() if len(results_df_a) > 0 and "score" in results_df_a.columns else None
        accuracy_b = results_df_b["score"].mean() if len(results_df_b) > 0 and "score" in results_df_b.columns else None
        
        judge_score = 0.0
        num_judged = 0
        judge_score_a = None
        judge_score_b = None
        if "judge_score" in results_df.columns:
            parseable_df = results_df[results_df["judge_score"].notna()]
            judge_score = parseable_df["judge_score"].mean() if len(parseable_df) > 0 else 0.0
            num_judged = len(parseable_df)
            
            parseable_df_a = results_df_a[results_df_a["judge_score"].notna()] if len(results_df_a) > 0 else []
            judge_score_a = parseable_df_a["judge_score"].mean() if len(parseable_df_a) > 0 else None
            
            parseable_df_b = results_df_b[results_df_b["judge_score"].notna()] if len(results_df_b) > 0 else []
            judge_score_b = parseable_df_b["judge_score"].mean() if len(parseable_df_b) > 0 else None
            
        # Save back to results json
        results_df.to_json(res_path, orient="records", indent=2)
        print(f"Updated/verified judge scores in: {res_path}")
        
        # Save debug prompts/responses to human-readable log file
        if args.save_debug:
            out_dir = os.path.dirname(res_path) or "."
            base_name = os.path.basename(res_path).replace("results_", "").replace(".json", "")
            debug_path = os.path.join(out_dir, f"judge_debug_{base_name}.txt")
            with open(debug_path, "w", encoding="utf-8") as fh:
                fh.write(f"=== Local Judge Debug Log for {base_name} ===\n\n")
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
            print(f"Saved human-readable debug logs to: {debug_path}")
            
        elapsed = time.time() - start_time
        summary_stats[scheme] = {
            "accuracy": accuracy,
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,
            "judge_score": judge_score,
            "judge_score_a": judge_score_a,
            "judge_score_b": judge_score_b,
            "results": len(results_df),
            "judged": num_judged,
            "time_taken": elapsed
        }

    # Print final summary table comparison
    if len(summary_stats) > 0:
        print("\n=== Local Comparison Summary ===")
        header = f"{'scheme':18s} | {'judge (all)':>11s} | {'judge (A)':>9s} | {'judge (B)':>9s}"
        print(header)
        print("-" * len(header))
        for scheme, stats in summary_stats.items():
            js = stats["judge_score"]
            js_a = stats["judge_score_a"]
            js_b = stats["judge_score_b"]
            
            js_a_str = f"{js_a:9.2%}" if js_a is not None else f"{'-':>9s}"
            js_b_str = f"{js_b:9.2%}" if js_b is not None else f"{'-':>9s}"
            
            print(f"{scheme:18s} | {js:11.2%} | {js_a_str} | {js_b_str}")
        print("=" * len(header))


if __name__ == "__main__":
    main()
