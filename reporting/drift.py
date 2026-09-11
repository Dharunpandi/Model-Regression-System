"""
Drift detection module for long-term model regression and degradation tracking.

Computes moving averages across multi-run windows and checks for slow score decline
against a persistent, explicitly calibrated baseline reference point.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_DRIFT_WINDOW,
    DRIFT_BASELINE_FILE,
    RUNS_DIR,
)


def get_recent_run_averages(
    db: Any = None,
    n: int = 10,
    runs_dir: Path = RUNS_DIR,
) -> List[Tuple[str, float, str]]:
    """
    Retrieves the last N runs ordered oldest first, returning a list of tuples:
      [(run_id, avg_score, timestamp), ...]
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []

    run_files = sorted(runs_path.glob("*.json"), key=os.path.getmtime)
    results: List[Tuple[str, float, str]] = []

    for p in run_files:
        if p.name.startswith("."):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            cases = data.get("case_scores", data.get("cases", []))
            if cases:
                scores = [
                    int(c.get("summary_relevance_score", c.get("score", 1)))
                    for c in cases
                ]
                avg = round(sum(scores) / len(scores), 3)
                results.append((
                    data.get("run_id", p.stem),
                    avg,
                    str(data.get("timestamp", "")),
                ))
        except Exception:
            continue

    return results[-n:]


def load_or_create_drift_baseline(
    recent_runs: List[Tuple[str, float, str]],
    baseline_file: Path = DRIFT_BASELINE_FILE,
    initial_window: int = DEFAULT_DRIFT_WINDOW,
) -> float:
    """
    Loads existing baseline from disk. If none exists, establishes one from the
    earliest available window of runs and persists it.

    DOES NOT recompute the baseline on subsequent calls.
    """
    base_path = Path(baseline_file)
    if base_path.is_file():
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("baseline_avg", 4.0))
        except Exception:
            pass

    # Establish initial baseline from earliest available window
    if not recent_runs:
        return 4.0

    eval_window = recent_runs[:initial_window]
    baseline_avg = round(sum(r[1] for r in eval_window) / len(eval_window), 3)

    # Persist baseline
    set_drift_baseline(baseline_avg, baseline_file=base_path)
    return baseline_avg


def set_drift_baseline(
    baseline_avg: float,
    baseline_file: Path = DRIFT_BASELINE_FILE,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Explicitly calibrates or resets the drift baseline reference point.
    """
    base_path = Path(baseline_file)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or {}
    payload = {
        "baseline_avg": round(float(baseline_avg), 3),
        "updated_at": meta.get("timestamp"),
        "notes": meta.get("notes", "Calibrated baseline score"),
    }
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return base_path



def check_drift(
    recent_runs: List[Tuple[str, float, str]],
    window: int = DEFAULT_DRIFT_WINDOW,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    baseline_file: Path = DRIFT_BASELINE_FILE,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates whether the moving average of the most recent `window` runs has
    dropped by >= drift_threshold below the established baseline.

    Returns None if fewer than `window` runs exist.
    """
    if len(recent_runs) < window:
        return None

    # Load persistent baseline (or compute from earliest if first run)
    baseline_avg = load_or_create_drift_baseline(
        recent_runs=recent_runs,
        baseline_file=baseline_file,
        initial_window=window,
    )

    # Compute moving average across the most recent `window` runs
    recent_window = recent_runs[-window:]
    moving_avg = round(sum(r[1] for r in recent_window) / len(recent_window), 3)

    drift_amount = round(baseline_avg - moving_avg, 3)
    drift_detected = drift_amount >= drift_threshold

    return {
        "drift_detected": drift_detected,
        "moving_avg": moving_avg,
        "baseline_avg": baseline_avg,
        "drift_amount": drift_amount,
        "drift_threshold": drift_threshold,
        "window": window,
        "runs_evaluated": len(recent_runs),
        "recent_runs": recent_runs,
    }
