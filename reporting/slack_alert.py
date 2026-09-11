"""
Slack alerting integration using Slack Incoming Webhooks and Block Kit formatting.

Features:
- Sends per-run regression alerts with status, headline scorecard metrics, and HTML report link.
- Sends distinct Slow Drift alerts with separate formatting and drift metrics.
- Enforces resp.raise_for_status() to surface API/network failures.
"""

import os
from typing import Any, Dict, Optional
import requests

from .config import SLACK_WEBHOOK_URL


def _get_webhook_url(webhook_url: Optional[str] = None) -> str:
    url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL") or SLACK_WEBHOOK_URL
    if not url:
        raise ValueError(
            "SLACK_WEBHOOK_URL is not set. Please provide a webhook URL via environment "
            "variable `SLACK_WEBHOOK_URL` or pass it directly."
        )
    return url


def send_slack_alert(
    scorecard: Dict[str, Any],
    status: str,
    emoji: str,
    report_url: Optional[str] = None,
    run_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> requests.Response:
    """
    Sends a per-run regression alert using Slack Block Kit.

    Payload format:
    - Header: Status line with emoji (e.g. '🚨 Critical Regression Alert')
    - Section: Headline numbers (regressions, average score change, regression rate)
    - Section / Context: Link to full HTML report (or hosting placeholder note)
    """
    target_url = _get_webhook_url(webhook_url)

    status_title = {
        "critical": "Critical Regression Detected",
        "warning": "Regression Warning",
        "pass": "Evaluation Passed",
    }.get(status.lower(), "Evaluation Update")

    regressed_count = scorecard.get("regressed_count", 0)
    improved_count = scorecard.get("improved_count", 0)
    total_cases = scorecard.get("total_cases", 0)
    avg_old = scorecard.get("avg_score_old", 0.0)
    avg_new = scorecard.get("avg_score_new", 0.0)
    reg_rate = scorecard.get("regression_rate", 0.0)

    headline = (
        f"*{regressed_count} regressions* detected out of {total_cases} cases "
        f"({reg_rate}% regression rate).\n"
        f"Average Score: *{avg_old}* → *{avg_new}* | Improvements: *{improved_count}*"
    )

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {status_title}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": headline,
            },
        },
    ]

    # Context block for run metadata
    context_elements = []
    if prompt_version:
        context_elements.append({"type": "mrkdwn", "text": f"*Prompt:* `{prompt_version}`"})
    if run_id:
        context_elements.append({"type": "mrkdwn", "text": f"*Run ID:* `{run_id[:12]}`"})
    if context_elements:
        blocks.append({"type": "context", "elements": context_elements})

    # Link to full HTML report
    if report_url and not report_url.startswith("TODO"):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"👉 <{report_url}|*View Full HTML Regression Report*>",
            },
        })
    else:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "_HTML report generated locally (Cloud report hosting pending configuration)._",
                }
            ],
        })

    payload = {"blocks": blocks}
    response = requests.post(target_url, json=payload, timeout=10)
    response.raise_for_status()
    return response


def send_drift_alert(
    drift_result: Dict[str, Any],
    report_url: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> requests.Response:
    """
    Sends a dedicated Slow Drift Alert with visually distinct styling.

    Used when baseline_avg - moving_avg exceeds drift threshold across a historical window,
    even if individual runs passed per-PR diff thresholds.
    """
    target_url = _get_webhook_url(webhook_url)

    baseline_avg = drift_result.get("baseline_avg", 0.0)
    moving_avg = drift_result.get("moving_avg", 0.0)
    drift_amount = drift_result.get("drift_amount", 0.0)
    window = drift_result.get("window", 7)
    runs_evaluated = drift_result.get("runs_evaluated", window)

    drift_text = (
        f"*Slow Drift Detected across the last {runs_evaluated} evaluation runs.*\n"
        f"• Baseline Score: *{baseline_avg:.2f}*\n"
        f"• Moving Average ({window}-run window): *{moving_avg:.2f}*\n"
        f"• Total Score Decline: *{drift_amount:.2f}* points\n\n"
        f"_Individual PR runs may have passed binary thresholds, but continuous baseline tracking "
        f"indicates cumulative performance degradation._"
    )

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "⚠️ Slow Drift Warning (Performance Degradation)",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": drift_text,
            },
        },
    ]

    if report_url and not report_url.startswith("TODO"):
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"🔍 <{report_url}|*Inspect Run History & Trend Data*>",
            },
        })

    payload = {"blocks": blocks}
    response = requests.post(target_url, json=payload, timeout=10)
    response.raise_for_status()
    return response
