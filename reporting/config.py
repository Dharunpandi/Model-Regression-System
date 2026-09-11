"""
Configuration constants and environment overrides for reporting, alerting, and drift detection.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Existing regression thresholds (reuse from project spec: 3% warning, 8% critical)
DEFAULT_WARN_THRESHOLD: float = float(
    os.environ.get("WARN_THRESHOLD", os.environ.get("REGRESSION_WARN_THRESHOLD", "0.03"))
)
DEFAULT_CRITICAL_THRESHOLD: float = float(
    os.environ.get("CRITICAL_THRESHOLD", os.environ.get("REGRESSION_CRITICAL_THRESHOLD", "0.08"))
)

# Drift detection settings
DEFAULT_DRIFT_WINDOW: int = int(os.environ.get("DRIFT_WINDOW", "7"))
DEFAULT_DRIFT_THRESHOLD: float = float(os.environ.get("DRIFT_THRESHOLD", "0.15"))

# Trend visualization settings
DEFAULT_TREND_N_RUNS: int = int(os.environ.get("TREND_N_RUNS", "10"))

# Storage paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RUNS_DIR: Path = Path(os.environ.get("RUNS_DIR", str(PROJECT_ROOT / "runs")))
REPORTS_DIR: Path = Path(os.environ.get("REPORTS_DIR", str(PROJECT_ROOT / "reports")))
DRIFT_BASELINE_FILE: Path = Path(
    os.environ.get("DRIFT_BASELINE_FILE", str(PROJECT_ROOT / "data" / "drift_baseline.json"))
)

# Slack Webhook configuration
SLACK_WEBHOOK_URL: str | None = os.environ.get("SLACK_WEBHOOK_URL")
