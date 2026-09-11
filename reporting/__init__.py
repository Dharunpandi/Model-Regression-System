"""
Reporting & Alerting Layer (Phase 4).
"""

from .config import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_DRIFT_WINDOW,
    DEFAULT_TREND_N_RUNS,
    DEFAULT_WARN_THRESHOLD,
    DRIFT_BASELINE_FILE,
    REPORTS_DIR,
    RUNS_DIR,
    SLACK_WEBHOOK_URL,
)
from .diff import build_scorecard, compute_diff, get_alert_status
from .drift import check_drift, get_recent_run_averages, set_drift_baseline
from .report import generate_report, get_trend_data
from .slack_alert import send_drift_alert, send_slack_alert

__all__ = [
    "DEFAULT_WARN_THRESHOLD",
    "DEFAULT_CRITICAL_THRESHOLD",
    "DEFAULT_DRIFT_WINDOW",
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_TREND_N_RUNS",
    "RUNS_DIR",
    "REPORTS_DIR",
    "DRIFT_BASELINE_FILE",
    "SLACK_WEBHOOK_URL",
    "compute_diff",
    "build_scorecard",
    "get_alert_status",
    "generate_report",
    "get_trend_data",
    "send_slack_alert",
    "send_drift_alert",
    "get_recent_run_averages",
    "check_drift",
    "set_drift_baseline",
]
