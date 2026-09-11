#!/usr/bin/env python3
"""
Orchestration Script: Per-Run Evaluation Diff, Report Generation, and Alerting.

Workflow:
1. Loads new evaluation run and designated baseline run (configurable).
2. Computes case-by-case score diffs (1-4 integer scale).
3. Builds scorecard metrics (total cases, regressions, improvements, avg scores).
4. Gathers historical trend data.
5. Generates standalone HTML report with Chart.js trend visualization.
6. Evaluates alert status against warning (3%) and critical (8%) thresholds.
7. Dispatches Slack Block Kit alert (if SLACK_WEBHOOK_URL is set or --slack passed).
8. Exits with code 1 if status == 'critical' (blocking CI), else 0.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.config import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_TREND_N_RUNS,
    DEFAULT_WARN_THRESHOLD,
    REPORTS_DIR,
    RUNS_DIR,
    SLACK_WEBHOOK_URL,
)
from reporting.diff import build_scorecard, compute_diff, get_alert_status
from reporting.report import generate_report, get_trend_data
from reporting.slack_alert import send_slack_alert


def find_latest_two_runs(runs_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Finds the two most recent run JSON files in runs_dir."""
    if not runs_dir.exists():
        return None, None
    run_files = sorted(
        [p for p in runs_dir.glob("*.json") if not p.name.startswith(".")],
        key=os.path.getmtime,
        reverse=True,
    )
    if len(run_files) >= 2:
        return run_files[0], run_files[1]
    elif len(run_files) == 1:
        return run_files[0], None
    return None, None


def print_cli_summary(
    new_run_id: str,
    baseline_run_id: Optional[str],
    prompt_version: str,
    scorecard: Dict[str, Any],
    status: str,
    emoji: str,
    report_path: Path,
    report_url: Optional[str],
    slack_sent: bool,
):
    """Prints a clean human-readable terminal summary table."""
    print("\n" + "=" * 75)
    print("                EVALUATION RUN DIFF & REGRESSION REPORT")
    print("=" * 75)
    print(f"  • New Run ID      : {new_run_id}")
    print(f"  • Baseline Run ID : {baseline_run_id or 'None (Baseline Initial Run)'}")
    print(f"  • Prompt Version  : {prompt_version}")
    print(f"  • Overall Status  : {emoji} {status.upper()}")
    print("-" * 75)
    print("  SCORECARD SUMMARY:")
    print(f"    - Total Cases Evaluated : {scorecard['total_cases']}")
    print(f"    - Regressions Detected  : {scorecard['regressed_count']} cases (Score drops)")
    print(f"    - Improvements Detected : {scorecard['improved_count']} cases")
    print(f"    - New Cases Added       : {scorecard['new_cases_count']} cases")
    print(f"    - Average Score (Old)   : {scorecard['avg_score_old']} / 4.0")
    print(f"    - Average Score (New)   : {scorecard['avg_score_new']} / 4.0")
    print(f"    - Regression Rate       : {scorecard['regression_rate']}%")
    print("-" * 75)
    print(f"  • HTML Report Path : {report_path}")
    if report_url and not report_url.startswith("TODO"):
        print(f"  • HTML Report URL  : {report_url}")
    else:
        print("  • HTML Report URL  : [TODO] Local file only (Cloud report hosting not yet configured)")
    print(f"  • Slack Alert Sent : {'Yes' if slack_sent else 'No (Skipped / Webhook not configured)'}")
    print("=" * 75)

    if status == "critical":
        print("\n🚨 CRITICAL REGRESSION ALERT: Regression rate exceeded critical threshold (>= 8.0%).")
        print("   Exiting with status code 1 to block CI pipeline.\n")
    elif status == "warning":
        print("\n⚠️ REGRESSION WARNING: Regression rate exceeded warning threshold (>= 3.0%).\n")
    else:
        print("\n✅ PASSED: Evaluation met quality thresholds.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation diff, generate HTML report, and dispatch Slack alerts."
    )
    parser.add_argument(
        "--new-run",
        type=str,
        default=None,
        help="Path or run_id for current/new evaluation run (defaults to latest in /runs)",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default=None,
        help="Path or run_id for designated baseline run (defaults to previous run or main baseline)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing evaluation run files (default: runs/)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory where HTML reports will be saved (default: reports/)",
    )
    parser.add_argument(
        "--report-url",
        type=str,
        default=os.environ.get("REPORT_URL", "TODO_HOSTING_URL_PENDING"),
        help="Public URL where the HTML report is hosted (for Slack alert link)",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=DEFAULT_WARN_THRESHOLD,
        help="Warning regression rate threshold (default: 0.03)",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=DEFAULT_CRITICAL_THRESHOLD,
        help="Critical regression rate threshold (default: 0.08)",
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Force dispatching Slack alert if webhook URL is set",
    )

    args = parser.parse_args()

    # 1. Resolve new and baseline runs
    latest_path, prev_path = find_latest_two_runs(args.runs_dir)

    new_run_ref = args.new_run or latest_path
    baseline_run_ref = args.baseline_run or prev_path or new_run_ref

    if new_run_ref is None:
        print(f"Error: No evaluation runs found in '{args.runs_dir}'. Run an evaluation first.")
        sys.exit(1)

    # 2. Compute case-level diffs
    diffs = compute_diff(
        old_run_id=baseline_run_ref,
        new_run_id=new_run_ref,
        runs_dir=args.runs_dir,
    )

    # 3. Build scorecard metrics
    scorecard = build_scorecard(diffs)

    # 4. Gather trend data
    trend_data = get_trend_data(n=DEFAULT_TREND_N_RUNS, runs_dir=args.runs_dir)

    # Extract metadata
    from reporting.diff import _load_run_data
    new_data = _load_run_data(new_run_ref, runs_dir=args.runs_dir)
    base_data = _load_run_data(baseline_run_ref, runs_dir=args.runs_dir)

    new_run_id = str(new_data.get("run_id", "current_run"))
    baseline_run_id = str(base_data.get("run_id", "")) if baseline_run_ref != new_run_ref else None
    prompt_version = str(new_data.get("prompt_version", "v1"))
    model_name = str(new_data.get("model", "llama-3.3-70b-versatile"))
    timestamp = str(new_data.get("timestamp", ""))

    # 5. Evaluate alert status
    status, emoji = get_alert_status(
        scorecard,
        warn_threshold=args.warn_threshold,
        critical_threshold=args.critical_threshold,
    )

    # 6. Generate HTML report
    html_path = generate_report(
        run_id=new_run_id,
        diffs=diffs,
        scorecard=scorecard,
        trend_data=trend_data,
        output_dir=args.reports_dir,
        baseline_run_id=baseline_run_id,
        prompt_version=prompt_version,
        model=model_name,
        timestamp=timestamp,
        alert_status=status,
        alert_emoji=emoji,
    )

    # 7. Send Slack alert (if webhook configured or flag provided)
    slack_sent = False
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL") or SLACK_WEBHOOK_URL
    if webhook_url:
        try:
            send_slack_alert(
                scorecard=scorecard,
                status=status,
                emoji=emoji,
                report_url=args.report_url,
                run_id=new_run_id,
                prompt_version=prompt_version,
                webhook_url=webhook_url,
            )
            slack_sent = True
        except Exception as e:
            print(f"Warning: Failed to send Slack alert: {e}", file=sys.stderr)
    elif args.slack:
        print("Warning: --slack passed but SLACK_WEBHOOK_URL is not set.", file=sys.stderr)

    # 8. Print terminal summary
    print_cli_summary(
        new_run_id=new_run_id,
        baseline_run_id=baseline_run_id,
        prompt_version=prompt_version,
        scorecard=scorecard,
        status=status,
        emoji=emoji,
        report_path=html_path,
        report_url=args.report_url,
        slack_sent=slack_sent,
    )

    # 9. CI Block check
    if status == "critical":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
