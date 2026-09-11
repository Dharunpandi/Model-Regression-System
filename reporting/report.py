"""
HTML diff report generator using Jinja2 and Chart.js.

Produces a self-contained static HTML report containing:
1. Run Metadata (run_id, prompt_version, model, timestamp, baseline_run_id)
2. Visual Scorecard (total cases, regressions in red, improvements in green, avg scores)
3. Table of ONLY regressed cases (side-by-side old vs new model outputs & scores)
4. Historical trend line chart (Chart.js via CDN across the last N runs)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from jinja2 import Template

from .config import DEFAULT_TREND_N_RUNS, REPORTS_DIR, RUNS_DIR


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Regression Report — {{ run_id }}</title>
  <!-- Chart.js from cdnjs -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-card: #1e293b;
      --bg-hover: #334155;
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --border-color: #334155;
      --red-accent: #ef4444;
      --red-bg: rgba(239, 68, 68, 0.15);
      --green-accent: #22c55e;
      --green-bg: rgba(34, 197, 94, 0.15);
      --yellow-accent: #eab308;
      --blue-accent: #38bdf8;
      --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-primary);
      color: var(--text-primary);
      font-family: var(--font-family);
      line-height: 1.5;
      padding: 2rem;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border-color);
    }
    .header h1 { font-size: 1.8rem; font-weight: 700; color: var(--text-primary); }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.4rem 1rem;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.9rem;
    }
    .status-pass { background: var(--green-bg); color: var(--green-accent); border: 1px solid var(--green-accent); }
    .status-warning { background: rgba(234, 179, 8, 0.15); color: var(--yellow-accent); border: 1px solid var(--yellow-accent); }
    .status-critical { background: var(--red-bg); color: var(--red-accent); border: 1px solid var(--red-accent); }
    
    /* Metadata Grid */
    .section-title { font-size: 1.25rem; font-weight: 600; margin: 1.5rem 0 1rem; color: var(--blue-accent); }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .meta-card {
      background: var(--bg-card);
      padding: 1rem 1.25rem;
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }
    .meta-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
    .meta-val { font-size: 1.1rem; font-weight: 600; margin-top: 0.25rem; word-break: break-all; }

    /* Scorecard */
    .scorecard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 2.5rem;
    }
    .scorecard-card {
      background: var(--bg-card);
      padding: 1.25rem;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      text-align: center;
    }
    .scorecard-num { font-size: 2rem; font-weight: 700; margin-top: 0.25rem; }
    .card-regression { border-color: var(--red-accent); background: var(--red-bg); }
    .card-regression .scorecard-num { color: var(--red-accent); }
    .card-improvement { border-color: var(--green-accent); background: var(--green-bg); }
    .card-improvement .scorecard-num { color: var(--green-accent); }

    /* Regressions Table */
    .table-container {
      background: var(--bg-card);
      border-radius: 8px;
      border: 1px solid var(--border-color);
      overflow-x: auto;
      margin-bottom: 2.5rem;
    }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th {
      background: rgba(15, 23, 42, 0.6);
      padding: 0.85rem 1rem;
      font-size: 0.85rem;
      text-transform: uppercase;
      color: var(--text-secondary);
      border-bottom: 1px solid var(--border-color);
    }
    td { padding: 1rem; border-bottom: 1px solid var(--border-color); vertical-align: top; font-size: 0.9rem; }
    tr:hover { background: var(--bg-hover); }
    .output-box {
      background: rgba(15, 23, 42, 0.4);
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      border: 1px solid var(--border-color);
      font-size: 0.85rem;
    }
    .cat-tag { display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 0.1rem 0.4rem; border-radius: 4px; margin-bottom: 0.3rem; }
    .cat-old { background: #475569; color: #f8fafc; }
    .cat-new { background: #3b82f6; color: #f8fafc; }
    .score-badge { font-weight: 700; font-size: 1rem; padding: 0.2rem 0.6rem; border-radius: 4px; text-align: center; }
    .score-drop { background: var(--red-bg); color: var(--red-accent); border: 1px solid var(--red-accent); }
    .no-regressions { padding: 2rem; text-align: center; color: var(--green-accent); font-weight: 600; font-size: 1.1rem; }

    /* Chart Container */
    .chart-container {
      background: var(--bg-card);
      border-radius: 8px;
      border: 1px solid var(--border-color);
      padding: 1.5rem;
      margin-bottom: 2.5rem;
    }
  </style>
</head>
<body>
  <div class="container">
    
    <!-- 1. Header & Run Metadata -->
    <div class="header">
      <div>
        <h1>LLM Regression Evaluation Report</h1>
        <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">
          Generated on {{ generated_at }}
        </div>
      </div>
      <div>
        <span class="status-badge status-{{ alert_status }}">
          {{ alert_emoji }} {{ alert_status | upper }}
        </span>
      </div>
    </div>

    <h2 class="section-title">1. Run Metadata</h2>
    <div class="meta-grid">
      <div class="meta-card">
        <div class="meta-label">New Run ID</div>
        <div class="meta-val">{{ run_id }}</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Baseline Run ID</div>
        <div class="meta-val">{{ baseline_run_id or "N/A (Initial Run)" }}</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Prompt Version</div>
        <div class="meta-val">{{ prompt_version }}</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Model</div>
        <div class="meta-val">{{ model }}</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Timestamp</div>
        <div class="meta-val">{{ timestamp }}</div>
      </div>
    </div>

    <!-- 2. Scorecard -->
    <h2 class="section-title">2. Scorecard</h2>
    <div class="scorecard-grid">
      <div class="scorecard-card">
        <div class="meta-label">Total Cases</div>
        <div class="scorecard-num">{{ scorecard.total_cases }}</div>
      </div>
      <div class="scorecard-card card-regression">
        <div class="meta-label" style="color: var(--red-accent);">Regressions</div>
        <div class="scorecard-num">{{ scorecard.regressed_count }}</div>
      </div>
      <div class="scorecard-card card-improvement">
        <div class="meta-label" style="color: var(--green-accent);">Improvements</div>
        <div class="scorecard-num">{{ scorecard.improved_count }}</div>
      </div>
      <div class="scorecard-card">
        <div class="meta-label">Avg Score (Old &rarr; New)</div>
        <div class="scorecard-num" style="font-size: 1.5rem;">
          {{ scorecard.avg_score_old }} &rarr; {{ scorecard.avg_score_new }}
        </div>
      </div>
      <div class="scorecard-card {% if scorecard.regression_rate >= 8.0 %}card-regression{% elif scorecard.regression_rate >= 3.0 %}scorecard-card{% endif %}">
        <div class="meta-label">Regression Rate</div>
        <div class="scorecard-num">{{ scorecard.regression_rate }}%</div>
      </div>
    </div>

    <!-- 3. Table of Regressed Cases ONLY -->
    <h2 class="section-title">3. Regressed Cases ({{ regressed_cases | length }} cases)</h2>
    <div class="table-container">
      {% if regressed_cases %}
      <table>
        <thead>
          <tr>
            <th style="width: 120px;">Case ID</th>
            <th style="width: 250px;">Customer Input</th>
            <th>Old Model Output</th>
            <th>New Model Output</th>
            <th style="width: 90px; text-align: center;">Old Score</th>
            <th style="width: 90px; text-align: center;">New Score</th>
            <th style="width: 90px; text-align: center;">Delta</th>
          </tr>
        </thead>
        <tbody>
          {% for c in regressed_cases %}
          <tr>
            <td><strong>{{ c.case_id }}</strong></td>
            <td style="color: var(--text-secondary); font-size: 0.85rem;">{{ c.input }}</td>
            <td>
              <div class="output-box">
                <span class="cat-tag cat-old">{{ c.old_output.category }}</span>
                <div>{{ c.old_output.summary }}</div>
              </div>
            </td>
            <td>
              <div class="output-box" style="border-color: var(--red-accent);">
                <span class="cat-tag cat-new">{{ c.new_output.category }}</span>
                <div>{{ c.new_output.summary }}</div>
              </div>
            </td>
            <td style="text-align: center;"><span class="score-badge">{{ c.old_score }}</span></td>
            <td style="text-align: center;"><span class="score-badge score-drop">{{ c.new_score }}</span></td>
            <td style="text-align: center;"><strong style="color: var(--red-accent);">{{ c.delta }}</strong></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <div class="no-regressions">
        🎉 No regressed cases detected! All evaluated cases matched or improved on baseline performance.
      </div>
      {% endif %}
    </div>

    <!-- 4. Historical Trend Line Chart -->
    <h2 class="section-title">4. Historical Average Score Trend (Last {{ trend_data | length }} Runs)</h2>
    <div class="chart-container">
      <canvas id="trendChart" style="max-height: 320px;"></canvas>
    </div>

  </div>

  <script>
    const trendLabels = {{ trend_labels | tojson }};
    const trendScores = {{ trend_scores | tojson }};

    const ctx = document.getElementById('trendChart').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: trendLabels,
        datasets: [{
          label: 'Average Run Score (1-4 Scale)',
          data: trendScores,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          fill: true,
          tension: 0.3,
          pointBackgroundColor: '#38bdf8',
          pointRadius: 5,
          pointHoverRadius: 7,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } },
          tooltip: {
            callbacks: {
              label: function(context) {
                return 'Avg Score: ' + context.parsed.y.toFixed(2) + ' / 4.0';
              }
            }
          }
        },
        scales: {
          y: {
            min: 1.0,
            max: 4.0,
            ticks: { color: '#94a3b8', stepSize: 0.5 },
            grid: { color: '#334155' }
          },
          x: {
            ticks: { color: '#94a3b8' },
            grid: { color: '#334155' }
          }
        }
      }
    });
  </script>
</body>
</html>
"""


def get_trend_data(
    db: Any = None,
    n: int = DEFAULT_TREND_N_RUNS,
    runs_dir: Path = RUNS_DIR,
) -> List[Dict[str, Any]]:
    """
    Retrieves historical run averages for the last N runs, ordered oldest to newest.

    Returns a list of dicts:
      [
        {"run_id": str, "avg_score": float, "timestamp": str, "prompt_version": str},
        ...
      ]
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []

    run_files = sorted(runs_path.glob("*.json"), key=os.path.getmtime)
    
    # Filter out non-run files (like drift baselines or non-run artifacts)
    history: List[Dict[str, Any]] = []
    for p in run_files:
        if p.name.startswith("."):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "case_scores" in data or "cases" in data:
                cases = data.get("case_scores", data.get("cases", []))
                scores = [
                    int(c.get("summary_relevance_score", c.get("score", 1)))
                    for c in cases
                ]
                avg = round(sum(scores) / len(scores), 2) if scores else 0.0
                history.append({
                    "run_id": data.get("run_id", p.stem),
                    "avg_score": avg,
                    "timestamp": str(data.get("timestamp", "")),
                    "prompt_version": data.get("prompt_version", "v1"),
                })
        except Exception:
            continue

    # Take the last n runs, ordered oldest to newest
    return history[-n:]


def generate_report(
    run_id: str,
    diffs: List[Dict[str, Any]],
    scorecard: Dict[str, Any],
    trend_data: List[Dict[str, Any]],
    output_dir: Path = REPORTS_DIR,
    baseline_run_id: Optional[str] = None,
    prompt_version: str = "v1",
    model: str = "llama-3.3-70b-versatile",
    timestamp: Optional[str] = None,
    alert_status: str = "pass",
    alert_emoji: str = "✅",
) -> Path:
    """
    Renders and saves the HTML diff report.

    Returns the Path to the generated HTML file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regressed_cases = [d for d in diffs if d.get("regressed", False)]

    # Prepare trend arrays for Chart.js
    trend_labels = [
        f"{t.get('prompt_version', '')} ({t.get('run_id', '')[:8]})"
        for t in trend_data
    ]
    trend_scores = [t.get("avg_score", 0.0) for t in trend_data]

    # If current run is not in trend_data, append it for visual completeness
    if run_id not in [t.get("run_id") for t in trend_data]:
        trend_labels.append(f"{prompt_version} ({run_id[:8]})")
        trend_scores.append(scorecard.get("avg_score_new", 0.0))

    template = Template(HTML_REPORT_TEMPLATE)
    html_content = template.render(
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        prompt_version=prompt_version,
        model=model,
        timestamp=timestamp or "Recent",
        generated_at=timestamp or "Recent",
        alert_status=alert_status,
        alert_emoji=alert_emoji,
        scorecard=scorecard,
        regressed_cases=regressed_cases,
        trend_data=trend_data,
        trend_labels=trend_labels,
        trend_scores=trend_scores,
    )

    report_path = out_dir / f"regression_report_{run_id}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path
