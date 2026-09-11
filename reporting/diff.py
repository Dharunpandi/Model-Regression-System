"""
Diff computation and scorecard aggregation module for evaluation runs.

Computes case-level score diffs (1-4 integer scale), identifies regressed and improved
cases, handles added/new cases gracefully, and determines alert severity against thresholds.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .config import DEFAULT_CRITICAL_THRESHOLD, DEFAULT_WARN_THRESHOLD, RUNS_DIR


def _load_run_data(run_ref: Union[str, Path, Dict[str, Any], Any], runs_dir: Path = RUNS_DIR) -> Dict[str, Any]:
    """
    Helper to normalize and load run data whether passed as a run_id string,
    file Path, EvalRun model instance, or raw dict.
    """
    if isinstance(run_ref, dict):
        return run_ref
    elif hasattr(run_ref, "model_dump"):  # Pydantic model (e.g. EvalRun)
        return run_ref.model_dump()
    elif isinstance(run_ref, (str, Path)):
        path = Path(run_ref)
        if not path.is_file():
            # If passed a run_id instead of a full file path
            candidate_path = Path(runs_dir) / f"{run_ref}.json"
            if candidate_path.is_file():
                path = candidate_path
            else:
                # Search inside runs_dir for matching filename or run_id
                matches = list(Path(runs_dir).glob(f"*{run_ref}*.json"))
                if matches:
                    path = matches[0]
                else:
                    raise FileNotFoundError(f"Evaluation run '{run_ref}' not found in {runs_dir}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported run reference type: {type(run_ref)}")


def compute_diff(
    old_run_id: Union[str, Path, Dict[str, Any], Any],
    new_run_id: Union[str, Path, Dict[str, Any], Any],
    db: Any = None,
    runs_dir: Path = RUNS_DIR,
) -> List[Dict[str, Any]]:
    """
    Computes per-case score diffs between two evaluation runs on the 1-4 integer scale.

    Handles cases where a test case exists in the new run but not the old run (new test case added)
    by setting delta=0, regressed=False, improved=False, and new_case=True.

    Returns a list of dicts with:
      - case_id: str
      - old_score: int | None
      - new_score: int
      - delta: int (new_score - old_score, or 0 if new_case)
      - regressed: bool (delta < 0)
      - improved: bool (delta > 0)
      - new_case: bool
      - old_output: dict (category, summary)
      - new_output: dict (category, summary)
      - input: str (email text)
      - golden_category: str
      - golden_summary: str
    """
    old_data = _load_run_data(old_run_id, runs_dir=runs_dir)
    new_data = _load_run_data(new_run_id, runs_dir=runs_dir)

    old_cases_list = old_data.get("case_scores", old_data.get("cases", []))
    new_cases_list = new_data.get("case_scores", new_data.get("cases", []))

    old_by_id = {c.get("case_id", c.get("id")): c for c in old_cases_list}

    diffs: List[Dict[str, Any]] = []

    for new_c in new_cases_list:
        case_id = str(new_c.get("case_id", new_c.get("id", "")))
        new_score = int(new_c.get("summary_relevance_score", new_c.get("score", 1)))
        
        # Candidate model outputs
        new_cat = new_c.get("actual_category", new_c.get("model_category", ""))
        new_sum = new_c.get("actual_summary", new_c.get("model_summary", ""))
        new_output = {"category": new_cat, "summary": new_sum}

        # Reference details
        input_text = new_c.get("input", new_c.get("email", ""))
        golden_cat = str(new_c.get("expected_category", new_c.get("golden_category", "")))
        golden_sum = new_c.get("expected_summary", new_c.get("golden_summary", ""))

        old_c = old_by_id.get(case_id)

        if old_c is None:
            # Case added in new run: treat delta as 0 and flag as new_case
            diffs.append({
                "case_id": case_id,
                "old_score": None,
                "new_score": new_score,
                "delta": 0,
                "regressed": False,
                "improved": False,
                "new_case": True,
                "old_output": None,
                "new_output": new_output,
                "input": input_text,
                "golden_category": golden_cat,
                "golden_summary": golden_sum,
            })
        else:
            old_score = int(old_c.get("summary_relevance_score", old_c.get("score", 1)))
            old_cat = old_c.get("actual_category", old_c.get("model_category", ""))
            old_sum = old_c.get("actual_summary", old_c.get("model_summary", ""))
            old_output = {"category": old_cat, "summary": old_sum}

            delta = new_score - old_score
            regressed = delta < 0
            improved = delta > 0

            diffs.append({
                "case_id": case_id,
                "old_score": old_score,
                "new_score": new_score,
                "delta": delta,
                "regressed": regressed,
                "improved": improved,
                "new_case": False,
                "old_output": old_output,
                "new_output": new_output,
                "input": input_text,
                "golden_category": golden_cat,
                "golden_summary": golden_sum,
            })

    return diffs


def build_scorecard(diffs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates per-case diffs into summary scorecard metrics.

    Returns dict with:
      - total_cases: int
      - regressed_count: int
      - improved_count: int
      - new_cases_count: int
      - avg_score_old: float (rounded to 2 decimal places)
      - avg_score_new: float (rounded to 2 decimal places)
      - regression_rate: float (percentage of total cases, rounded to 1 decimal place)
    """
    total_cases = len(diffs)
    if total_cases == 0:
        return {
            "total_cases": 0,
            "regressed_count": 0,
            "improved_count": 0,
            "new_cases_count": 0,
            "avg_score_old": 0.0,
            "avg_score_new": 0.0,
            "regression_rate": 0.0,
        }

    regressed_count = sum(1 for d in diffs if d.get("regressed", False))
    improved_count = sum(1 for d in diffs if d.get("improved", False))
    new_cases_count = sum(1 for d in diffs if d.get("new_case", False))

    old_scores = [d["old_score"] for d in diffs if d.get("old_score") is not None]
    new_scores = [d["new_score"] for d in diffs if d.get("new_score") is not None]

    avg_score_old = round(sum(old_scores) / len(old_scores), 2) if old_scores else 0.0
    avg_score_new = round(sum(new_scores) / len(new_scores), 2) if new_scores else 0.0

    regression_rate = round((regressed_count / total_cases) * 100.0, 1)

    return {
        "total_cases": total_cases,
        "regressed_count": regressed_count,
        "improved_count": improved_count,
        "new_cases_count": new_cases_count,
        "avg_score_old": avg_score_old,
        "avg_score_new": avg_score_new,
        "regression_rate": regression_rate,
    }


def get_alert_status(
    scorecard: Dict[str, Any],
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> Tuple[str, str]:
    """
    Evaluates regression rate against warning and critical thresholds.
    Accepts thresholds as fractions (e.g. 0.03, 0.08) or percentages (e.g. 3.0, 8.0).

    Returns:
      (status: "pass" | "warning" | "critical", emoji: str)
    """
    regression_rate = float(scorecard.get("regression_rate", 0.0))

    # Normalize thresholds to percentages (e.g. 0.03 -> 3.0%)
    warn_pct = warn_threshold * 100.0 if warn_threshold <= 1.0 else warn_threshold
    crit_pct = critical_threshold * 100.0 if critical_threshold <= 1.0 else critical_threshold

    if regression_rate >= crit_pct:
        return ("critical", "🚨")
    elif regression_rate >= warn_pct:
        return ("warning", "⚠️")
    else:
        return ("pass", "✅")
