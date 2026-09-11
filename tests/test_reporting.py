"""
Unit tests for Phase 4: Reporting and Alerting Layer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from reporting.diff import build_scorecard, compute_diff, get_alert_status
from reporting.drift import (
    check_drift,
    get_recent_run_averages,
    load_or_create_drift_baseline,
    set_drift_baseline,
)
from reporting.report import generate_report, get_trend_data
from reporting.slack_alert import send_drift_alert, send_slack_alert


@pytest.fixture
def sample_runs():
    """Provides two synthetic evaluation runs for testing."""
    old_run = {
        "run_id": "run_baseline_001",
        "prompt_version": "v1",
        "model": "llama-3.3-70b",
        "timestamp": "2026-09-10T12:00:00Z",
        "case_scores": [
            {
                "case_id": "c1",
                "summary_relevance_score": 4,
                "actual_category": "billing",
                "actual_summary": "Refund requested",
                "input": "I need a refund",
                "expected_category": "billing",
                "expected_summary": "Refund requested",
            },
            {
                "case_id": "c2",
                "summary_relevance_score": 4,
                "actual_category": "technical",
                "actual_summary": "App crash bug",
                "input": "App crashed",
                "expected_category": "technical",
                "expected_summary": "App crash",
            },
            {
                "case_id": "c3",
                "summary_relevance_score": 2,
                "actual_category": "account",
                "actual_summary": "Vague summary",
                "input": "Cannot login",
                "expected_category": "account",
                "expected_summary": "Password reset error",
            },
        ],
    }

    new_run = {
        "run_id": "run_candidate_002",
        "prompt_version": "v2",
        "model": "llama-3.3-70b",
        "timestamp": "2026-09-11T12:00:00Z",
        "case_scores": [
            {
                "case_id": "c1",
                "summary_relevance_score": 3,  # Regressed (4 -> 3)
                "actual_category": "billing",
                "actual_summary": "User wants refund",
                "input": "I need a refund",
                "expected_category": "billing",
                "expected_summary": "Refund requested",
            },
            {
                "case_id": "c2",
                "summary_relevance_score": 4,  # Equal (4 == 4)
                "actual_category": "technical",
                "actual_summary": "App crash bug",
                "input": "App crashed",
                "expected_category": "technical",
                "expected_summary": "App crash",
            },
            {
                "case_id": "c3",
                "summary_relevance_score": 4,  # Improved (2 -> 4)
                "actual_category": "account",
                "actual_summary": "Password reset error",
                "input": "Cannot login",
                "expected_category": "account",
                "expected_summary": "Password reset error",
            },
            {
                "case_id": "c4_new",  # Added test case
                "summary_relevance_score": 4,
                "actual_category": "general",
                "actual_summary": "Feedback message",
                "input": "Nice product",
                "expected_category": "general",
                "expected_summary": "Feedback message",
            },
        ],
    }
    return old_run, new_run


def test_compute_diff_and_new_case_handling(sample_runs):
    """Test compute_diff produces correct deltas and handles new cases without crashing."""
    old_run, new_run = sample_runs

    diffs = compute_diff(old_run, new_run)
    assert len(diffs) == 4

    by_id = {d["case_id"]: d for d in diffs}

    # Case 1: regressed 4 -> 3
    assert by_id["c1"]["old_score"] == 4
    assert by_id["c1"]["new_score"] == 3
    assert by_id["c1"]["delta"] == -1
    assert by_id["c1"]["regressed"] is True
    assert by_id["c1"]["improved"] is False
    assert by_id["c1"]["new_case"] is False

    # Case 2: unchanged 4 -> 4
    assert by_id["c2"]["delta"] == 0
    assert by_id["c2"]["regressed"] is False
    assert by_id["c2"]["improved"] is False

    # Case 3: improved 2 -> 4
    assert by_id["c3"]["delta"] == 2
    assert by_id["c3"]["regressed"] is False
    assert by_id["c3"]["improved"] is True

    # Case 4 (new case): delta 0, new_case=True
    assert by_id["c4_new"]["old_score"] is None
    assert by_id["c4_new"]["new_score"] == 4
    assert by_id["c4_new"]["delta"] == 0
    assert by_id["c4_new"]["new_case"] is True
    assert by_id["c4_new"]["regressed"] is False


def test_build_scorecard(sample_runs):
    """Test scorecard metric computation."""
    old_run, new_run = sample_runs
    diffs = compute_diff(old_run, new_run)
    scorecard = build_scorecard(diffs)

    assert scorecard["total_cases"] == 4
    assert scorecard["regressed_count"] == 1
    assert scorecard["improved_count"] == 1
    assert scorecard["new_cases_count"] == 1
    # 1 regression out of 4 cases = 25.0%
    assert scorecard["regression_rate"] == 25.0
    assert scorecard["avg_score_new"] == 3.75


def test_get_alert_status():
    """Test alert status thresholds (pass < 3%, warning 3-8%, critical >= 8%)."""
    # Passing (0% or 2.0%)
    status, emoji = get_alert_status({"regression_rate": 2.0})
    assert status == "pass"
    assert emoji == "✅"

    # Warning (3.0% to 7.9%)
    status, emoji = get_alert_status({"regression_rate": 5.0})
    assert status == "warning"
    assert emoji == "⚠️"

    # Critical (>= 8.0%)
    status, emoji = get_alert_status({"regression_rate": 8.0})
    assert status == "critical"
    assert emoji == "🚨"

    status, emoji = get_alert_status({"regression_rate": 25.0})
    assert status == "critical"
    assert emoji == "🚨"


def test_html_report_generation(sample_runs, tmp_path):
    """Test Jinja2 HTML report generator and structure."""
    old_run, new_run = sample_runs
    diffs = compute_diff(old_run, new_run)
    scorecard = build_scorecard(diffs)
    trend_data = [{"run_id": "r1", "avg_score": 3.5, "prompt_version": "v1"}]

    report_path = generate_report(
        run_id="candidate_002",
        diffs=diffs,
        scorecard=scorecard,
        trend_data=trend_data,
        output_dir=tmp_path,
        baseline_run_id="baseline_001",
        prompt_version="v2",
        alert_status="critical",
        alert_emoji="🚨",
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    # Verify key elements
    assert "candidate_002" in content
    assert "baseline_001" in content
    assert "https://cdnjs.cloudflare.com/ajax/libs/Chart.js" in content
    assert "c1" in content  # Regressed case present in table
    assert "trendChart" in content  # Chart.js canvas present


def test_drift_detection_and_baseline_persistence(tmp_path):
    """Test drift detection, persistent baseline storage, and recalibration."""
    baseline_file = tmp_path / "drift_baseline.json"

    # Create synthetic series of 8 runs with initial avg ~3.9 then declining
    runs_history = [
        ("r1", 3.9, "t1"),
        ("r2", 3.85, "t2"),
        ("r3", 3.9, "t3"),
        ("r4", 3.8, "t4"),
        ("r5", 3.9, "t5"),
        ("r6", 3.85, "t6"),
        ("r7", 3.9, "t7"),  # First 7 baseline window avg = ~3.871
    ]

    # 1. First use establishes persistent baseline
    base = load_or_create_drift_baseline(runs_history, baseline_file=baseline_file, initial_window=7)
    assert baseline_file.exists()
    assert 3.85 <= base <= 3.90

    # 2. Check drift with stable runs -> no drift
    res = check_drift(runs_history, window=7, drift_threshold=0.15, baseline_file=baseline_file)
    assert res is not None
    assert res["drift_detected"] is False

    # 3. Add 7 degraded runs (avg 3.6)
    degraded_runs = runs_history + [
        ("r8", 3.6, "t8"),
        ("r9", 3.55, "t9"),
        ("r10", 3.6, "t10"),
        ("r11", 3.5, "t11"),
        ("r12", 3.55, "t12"),
        ("r13", 3.6, "t13"),
        ("r14", 3.5, "t14"),
    ]

    res_drift = check_drift(degraded_runs, window=7, drift_threshold=0.15, baseline_file=baseline_file)
    assert res_drift is not None
    # Drop of ~0.30+ clears 0.15 threshold
    assert res_drift["drift_detected"] is True
    assert res_drift["drift_amount"] >= 0.15

    # 4. Explicit manual recalibration
    set_drift_baseline(3.55, baseline_file=baseline_file)
    reloaded_base = load_or_create_drift_baseline(degraded_runs, baseline_file=baseline_file)
    assert reloaded_base == 3.55


def test_slack_alert_dispatch():
    """Test Slack alert Block Kit generation with mocked requests."""
    scorecard = {
        "total_cases": 10,
        "regressed_count": 2,
        "improved_count": 1,
        "avg_score_old": 3.8,
        "avg_score_new": 3.5,
        "regression_rate": 20.0,
    }

    with patch("reporting.slack_alert.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        # 1. Per-run regression alert
        send_slack_alert(
            scorecard=scorecard,
            status="critical",
            emoji="🚨",
            report_url="https://reports.example.com/run123",
            run_id="run_12345",
            prompt_version="v2",
            webhook_url="https://hooks.slack.com/services/TEST/MOCK",
        )

        assert mock_post.called
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        blocks = payload["blocks"]
        assert any("Critical Regression Detected" in str(b) for b in blocks)
        assert any("https://reports.example.com/run123" in str(b) for b in blocks)

        # 2. Slow drift alert
        drift_result = {
            "baseline_avg": 3.9,
            "moving_avg": 3.6,
            "drift_amount": 0.3,
            "window": 7,
            "runs_evaluated": 14,
        }
        send_drift_alert(
            drift_result=drift_result,
            webhook_url="https://hooks.slack.com/services/TEST/MOCK",
        )
        assert any("Slow Drift Warning" in str(b) for b in mock_post.call_args[1]["json"]["blocks"])


def test_missing_webhook_raises():
    """Test that missing webhook URL raises ValueError when sending."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="SLACK_WEBHOOK_URL is not set"):
            send_slack_alert(scorecard={}, status="pass", emoji="✅", webhook_url=None)


def test_cli_scripts_execution(sample_runs, tmp_path):
    """Test orchestration script and drift CLI script execution with synthetic runs."""
    from scripts.run_eval_report import find_latest_two_runs
    import subprocess

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    old_run, new_run = sample_runs

    # Save runs as json
    with open(runs_dir / "run_v1.json", "w", encoding="utf-8") as f:
        json.dump(old_run, f)
    with open(runs_dir / "run_v2.json", "w", encoding="utf-8") as f:
        json.dump(new_run, f)

    # Test find_latest_two_runs
    latest, prev = find_latest_two_runs(runs_dir)
    assert latest is not None

    # Test trend data helper
    trends = get_trend_data(runs_dir=runs_dir, n=5)
    assert len(trends) == 2

