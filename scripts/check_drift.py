#!/usr/bin/env python3
"""
Standalone Drift Detection Script.

Runs on a separate schedule (e.g. daily cron, nightly CI, or scheduled GitHub Action).
Tracks multi-run moving average against a persistent baseline reference point.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.config import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_DRIFT_WINDOW,
    DRIFT_BASELINE_FILE,
    RUNS_DIR,
    SLACK_WEBHOOK_URL,
)
from reporting.drift import (
    check_drift,
    get_recent_run_averages,
    load_or_create_drift_baseline,
    set_drift_baseline,
)
from reporting.slack_alert import send_drift_alert


def print_drift_summary(drift_result: dict | None, window: int, threshold: float):
    """Prints a formatted ASCII summary of drift detection results."""
    print("\n" + "=" * 75)
    print("                    MODEL DRIFT MONITORING REPORT")
    print("=" * 75)

    if drift_result is None:
        print(f"  ℹ️ Insufficient history: Fewer than {window} evaluation runs available.")
        print("  Baseline will be established automatically once sufficient runs are recorded.")
        print("=" * 75 + "\n")
        return

    baseline_avg = drift_result["baseline_avg"]
    moving_avg = drift_result["moving_avg"]
    drift_amount = drift_result["drift_amount"]
    drift_detected = drift_result["drift_detected"]
    runs_evaluated = drift_result["runs_evaluated"]

    status_str = "⚠️ SLOW DRIFT DETECTED" if drift_detected else "✅ NO DRIFT DETECTED"

    print(f"  • Status             : {status_str}")
    print(f"  • Evaluated Runs     : {runs_evaluated} runs in history (Moving window = {window})")
    print(f"  • Calibrated Baseline: {baseline_avg:.3f} / 4.000")
    print(f"  • Recent Moving Avg  : {moving_avg:.3f} / 4.000")
    print(f"  • Score Drop (Delta) : {drift_amount:+.3f} points")
    print(f"  • Drift Threshold    : {threshold:.3f} points")
    print("-" * 75)

    if drift_detected:
        print("  ⚠️ WARNING: The moving average score has degraded beyond the drift threshold.")
        print("     This indicates cumulative quality degradation across prompt/dataset revisions.")
    else:
        print("  ✅ Performance remains stable relative to the calibrated baseline.")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check evaluation run history for long-term score drift."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_DRIFT_WINDOW,
        help=f"Number of recent runs to include in moving average (default: {DEFAULT_DRIFT_WINDOW})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=f"Score drop threshold to trigger drift alert (default: {DEFAULT_DRIFT_THRESHOLD})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing run JSON files (default: runs/)",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=DRIFT_BASELINE_FILE,
        help="Path to persistent baseline JSON file (default: data/drift_baseline.json)",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recalibrate the persistent baseline from the earliest available run window",
    )
    parser.add_argument(
        "--set-baseline",
        type=float,
        default=None,
        help="Manually set a specific baseline score value (e.g. 3.85)",
    )
    parser.add_argument(
        "--report-url",
        type=str,
        default=os.environ.get("REPORT_URL", "TODO_HOSTING_URL_PENDING"),
        help="Public URL for trend dashboard (for Slack alert link)",
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Send Slack alert if drift is detected and webhook is configured",
    )

    args = parser.parse_args()

    # Retrieve history
    recent_runs = get_recent_run_averages(n=50, runs_dir=args.runs_dir)

    # Handle manual recalibration
    if args.set_baseline is not None:
        set_drift_baseline(
            baseline_avg=args.set_baseline,
            baseline_file=args.baseline_file,
            metadata={"notes": "Manually configured baseline via CLI"},
        )
        print(f"Drift baseline manually set to {args.set_baseline:.3f} and saved to {args.baseline_file}")

    elif args.recalibrate:
        if not recent_runs:
            print(f"Cannot recalibrate: No runs found in {args.runs_dir}")
            sys.exit(1)
        eval_window = recent_runs[:args.window]
        new_base = round(sum(r[1] for r in eval_window) / len(eval_window), 3)
        set_drift_baseline(
            baseline_avg=new_base,
            baseline_file=args.baseline_file,
            metadata={"notes": f"Recalibrated from first {len(eval_window)} runs"},
        )
        print(f"Drift baseline recalibrated to {new_base:.3f} and saved to {args.baseline_file}")

    # Check drift
    drift_result = check_drift(
        recent_runs=recent_runs,
        window=args.window,
        drift_threshold=args.threshold,
        baseline_file=args.baseline_file,
    )

    print_drift_summary(drift_result, window=args.window, threshold=args.threshold)

    # Slack notification if drift detected
    if drift_result and drift_result.get("drift_detected", False):
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL") or SLACK_WEBHOOK_URL
        if webhook_url:
            try:
                send_drift_alert(
                    drift_result=drift_result,
                    report_url=args.report_url,
                    webhook_url=webhook_url,
                )
                print("Slack drift alert sent successfully.")
            except Exception as e:
                print(f"Warning: Failed to send Slack drift alert: {e}", file=sys.stderr)
        elif args.slack:
            print("Warning: --slack passed but SLACK_WEBHOOK_URL is not configured.", file=sys.stderr)

        # Non-zero exit code when drift is detected
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
