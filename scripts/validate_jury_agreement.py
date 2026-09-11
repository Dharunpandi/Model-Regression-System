#!/usr/bin/env python3
"""
Validation Script: SLM Jury vs. GPT-4o Judge Agreement & Reproducibility Analysis.

Compares judgment quality and alignment between GPT-4o and the SLM Jury (Qwen3-4B via Ollama).
Computes:
1. Overall agreement % (within_one: |gpt4o - slm| <= 1).
2. Exact match % (gpt4o == slm).
3. Agreement % restricted to cases where gpt4o score <= 2 (Rubber-stamp detector).
4. Reproducibility test: 10 cases evaluated 3 times each to verify determinism.

Outputs a clean ASCII table and exports full raw results to JSON/CSV.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from judges.gpt4o import gpt4o_judge
from judges.slm_jury import slm_jury_judge
from src.schema import GoldenDataset


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Loads cases from a golden dataset JSON file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unrecognized dataset structure in {dataset_path}")


def run_agreement_validation(
    cases: List[Dict[str, Any]],
    ollama_model: str = "qwen3:4b",
    gpt_model: str = "gpt-4o",
    limit: Optional[int] = None,
    mock_gpt: bool = False,
    mock_jury: bool = False,
) -> Dict[str, Any]:
    """
    Runs both judges against dataset cases and calculates agreement metrics.
    """
    eval_cases = cases[:limit] if limit else cases
    records = []

    print(f"\n[1/2] Evaluating {len(eval_cases)} test cases across both judges...")
    print("=" * 70)

    for idx, case in enumerate(eval_cases, 1):
        case_id = case.get("id", f"case_{idx:03d}")
        email = case.get("input", "")
        golden_category = str(case.get("expected_category", ""))
        golden_summary = case.get("expected_summary", "")

        # Candidate output under test: if case contains mock candidate output use it,
        # otherwise use expected as candidate (or slight variations for testing)
        candidate_category = case.get("model_category", golden_category)
        candidate_summary = case.get("model_summary", golden_summary)
        model_output = {
            "category": candidate_category,
            "summary": candidate_summary,
        }

        print(f"  [{idx:02d}/{len(eval_cases):02d}] Evaluating {case_id}...", end="", flush=True)

        start_time = time.perf_counter()
        
        # Run GPT-4o judge
        gpt_res = gpt4o_judge(
            email=email,
            golden_category=golden_category,
            golden_summary=golden_summary,
            model_output=model_output,
            model_name=gpt_model,
        )
        gpt_score = int(gpt_res.get("score", 1))

        # Run SLM Jury judge
        jury_res = slm_jury_judge(
            email=email,
            golden_category=golden_category,
            golden_summary=golden_summary,
            model_output=model_output,
            model_name=ollama_model,
        )
        jury_score = int(jury_res.get("score", 1))

        elapsed = time.perf_counter() - start_time

        exact_match = (gpt_score == jury_score)
        within_one = (abs(gpt_score - jury_score) <= 1)
        case_was_wrong = (gpt_score <= 2)

        record = {
            "case_id": case_id,
            "difficulty": case.get("difficulty", "medium"),
            "email_preview": (email[:45] + "...") if len(email) > 45 else email,
            "golden_category": golden_category,
            "candidate_category": candidate_category,
            "gpt4o_score": gpt_score,
            "slm_jury_score": jury_score,
            "exact_match": exact_match,
            "within_one": within_one,
            "case_was_wrong": case_was_wrong,
            "elapsed_s": round(elapsed, 2),
            "jury_trail": jury_res.get("trail", {}),
            "gpt4o_trail": gpt_res.get("trail", {}),
        }
        records.append(record)
        match_symbol = "✓" if exact_match else ("~" if within_one else "✗")
        print(f" GPT-4o: {gpt_score} | Jury: {jury_score} [{match_symbol}] ({elapsed:.1f}s)")

    # Compute aggregate metrics
    total = len(records)
    exact_count = sum(1 for r in records if r["exact_match"])
    within_one_count = sum(1 for r in records if r["within_one"])
    
    wrong_cases = [r for r in records if r["case_was_wrong"]]
    wrong_total = len(wrong_cases)
    wrong_agreed_count = sum(1 for r in wrong_cases if r["within_one"])
    wrong_exact_count = sum(1 for r in wrong_cases if r["exact_match"])

    exact_pct = (exact_count / total * 100) if total else 0.0
    within_one_pct = (within_one_count / total * 100) if total else 0.0
    wrong_agreed_pct = (wrong_agreed_count / wrong_total * 100) if wrong_total else 0.0
    wrong_exact_pct = (wrong_exact_count / wrong_total * 100) if wrong_total else 0.0

    return {
        "records": records,
        "total_cases": total,
        "exact_match_pct": exact_pct,
        "within_one_pct": within_one_pct,
        "wrong_cases_count": wrong_total,
        "wrong_cases_within_one_pct": wrong_agreed_pct,
        "wrong_cases_exact_pct": wrong_exact_pct,
    }


def run_reproducibility_check(
    cases: List[Dict[str, Any]],
    sample_size: int = 10,
    repeats: int = 3,
    ollama_model: str = "qwen3:4b",
) -> Dict[str, Any]:
    """
    Runs reproducibility check by evaluating sample cases multiple times with temp=0/seed=42.
    """
    n_sample = min(sample_size, len(cases))
    sample_cases = random.sample(cases, n_sample) if len(cases) > n_sample else list(cases)

    print(f"\n[2/2] Running Reproducibility Check ({n_sample} cases x {repeats} runs each)...")
    print("=" * 70)

    repro_results = []
    identical_count = 0

    for idx, case in enumerate(sample_cases, 1):
        case_id = case.get("id", f"case_{idx:03d}")
        email = case.get("input", "")
        golden_category = str(case.get("expected_category", ""))
        golden_summary = case.get("expected_summary", "")
        model_output = {
            "category": case.get("model_category", golden_category),
            "summary": case.get("model_summary", golden_summary),
        }

        scores = []
        for r in range(repeats):
            res = slm_jury_judge(
                email=email,
                golden_category=golden_category,
                golden_summary=golden_summary,
                model_output=model_output,
                model_name=ollama_model,
            )
            scores.append(int(res.get("score", 1)))

        is_deterministic = len(set(scores)) == 1
        if is_deterministic:
            identical_count += 1

        repro_results.append({
            "case_id": case_id,
            "scores": scores,
            "deterministic": is_deterministic,
        })
        status_sym = "✓" if is_deterministic else "✗"
        print(f"  Case {case_id}: runs={scores} -> {'Deterministic' if is_deterministic else 'Variance detected'} [{status_sym}]")

    reproducibility_pct = (identical_count / n_sample * 100) if n_sample else 0.0

    return {
        "sample_size": n_sample,
        "repeats": repeats,
        "deterministic_cases": identical_count,
        "reproducibility_pct": reproducibility_pct,
        "details": repro_results,
    }


def print_summary_table(agreement_data: Dict[str, Any], repro_data: Dict[str, Any]):
    """Prints a structured ASCII report of validation results."""
    records = agreement_data["records"]
    
    print("\n" + "=" * 80)
    print("                      SLM JURY VALIDATION REPORT")
    print("=" * 80)

    # Detailed Per-Case Table
    header = f"{'Case ID':<10} | {'Diff':<7} | {'Email Snippet':<28} | {'GPT-4o':<6} | {'Jury':<5} | {'Match':<5}"
    print(header)
    print("-" * 80)
    for r in records:
        match_str = "EXACT" if r["exact_match"] else ("±1" if r["within_one"] else "DIFF")
        print(f"{r['case_id']:<10} | {r['difficulty']:<7} | {r['email_preview']:<28} | {r['gpt4o_score']:<6} | {r['slm_jury_score']:<5} | {match_str:<5}")
    print("-" * 80)

    # Aggregate Metrics
    print("\n" + "=" * 80)
    print("                           SUMMARY METRICS")
    print("=" * 80)
    print(f"  • Total Evaluated Cases          : {agreement_data['total_cases']}")
    print(f"  • Exact Score Agreement (0-diff) : {agreement_data['exact_match_pct']:.1f}%")
    print(f"  • Overall Agreement (<= 1-diff)  : {agreement_data['within_one_pct']:.1f}%")
    print(f"  • Hard Failure Cases (GPT-4o<=2) : {agreement_data['wrong_cases_count']}")
    print(f"  • Agreement on Failures (<=1)    : {agreement_data['wrong_cases_within_one_pct']:.1f}% (Rubber-stamp check)")
    print(f"  • Exact Agreement on Failures    : {agreement_data['wrong_cases_exact_pct']:.1f}%")
    print(f"  • Jury Determinism (3-run test)  : {repro_data['reproducibility_pct']:.1f}% ({repro_data['deterministic_cases']}/{repro_data['sample_size']} cases)")
    print("=" * 80)

    # Readiness Assessment
    print("\n[READINESS ASSESSMENT]")
    if agreement_data["within_one_pct"] >= 85.0 and agreement_data["wrong_cases_within_one_pct"] >= 80.0 and repro_data["reproducibility_pct"] >= 90.0:
        print("  🟢 PASSED: SLM Jury demonstrates strong alignment and determinism.")
        print("  NOTE: You may consider enabling JUDGE_MODE='slm_jury' in CI after inspecting raw trails.")
    else:
        print("  🟡 CAUTION: Agreement thresholds below recommended CI promotion bar (>=85% overall, >=80% on failures).")
        print("  IMPORTANT: Keep JUDGE_MODE='gpt4o' in production CI until calibration examples are expanded.")
    print("=" * 80 + "\n")


def save_results(
    output_path: Path,
    agreement_data: Dict[str, Any],
    repro_data: Dict[str, Any],
):
    """Saves raw validation results to JSON and CSV formats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = output_path.with_suffix(".json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_cases": agreement_data["total_cases"],
            "exact_match_pct": agreement_data["exact_match_pct"],
            "within_one_pct": agreement_data["within_one_pct"],
            "wrong_cases_count": agreement_data["wrong_cases_count"],
            "wrong_cases_within_one_pct": agreement_data["wrong_cases_within_one_pct"],
            "wrong_cases_exact_pct": agreement_data["wrong_cases_exact_pct"],
            "reproducibility_pct": repro_data["reproducibility_pct"],
        },
        "reproducibility": repro_data,
        "cases": agreement_data["records"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full JSON report to: {json_path}")

    # Save CSV report
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "difficulty", "golden_category", "candidate_category", "gpt4o_score", "slm_jury_score", "exact_match", "within_one", "case_was_wrong", "elapsed_s"])
        for r in agreement_data["records"]:
            writer.writerow([
                r["case_id"],
                r["difficulty"],
                r["golden_category"],
                r["candidate_category"],
                r["gpt4o_score"],
                r["slm_jury_score"],
                r["exact_match"],
                r["within_one"],
                r["case_was_wrong"],
                r["elapsed_s"],
            ])
    print(f"Saved tabular CSV report to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate SLM Jury agreement with GPT-4o judge.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "golden_dataset_v1.json",
        help="Path to golden dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "jury_agreement_results.json",
        help="Path for saving validation results (JSON & CSV)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen3:4b",
        help="Ollama model tag to use for jury agents (default: qwen3:4b)",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model tag for benchmark judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of dataset cases to evaluate",
    )
    parser.add_argument(
        "--repro-samples",
        type=int,
        default=10,
        help="Number of random samples for reproducibility test (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("       STARTING SLM JURY vs. GPT-4o AGREEMENT VALIDATION")
    print(f"       Dataset: {args.dataset}")
    print(f"       Ollama Model: {args.ollama_model} | Benchmark Model: {args.gpt_model}")
    print("=" * 80)

    cases = load_dataset(args.dataset)
    print(f"Loaded {len(cases)} cases from dataset.")

    # 1. Run agreement validation
    agreement_data = run_agreement_validation(
        cases=cases,
        ollama_model=args.ollama_model,
        gpt_model=args.gpt_model,
        limit=args.limit,
    )

    # 2. Run reproducibility test
    repro_data = run_reproducibility_check(
        cases=cases,
        sample_size=args.repro_samples,
        repeats=3,
        ollama_model=args.ollama_model,
    )

    # 3. Print report and save artifacts
    print_summary_table(agreement_data, repro_data)
    save_results(args.output, agreement_data, repro_data)


if __name__ == "__main__":
    main()
