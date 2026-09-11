# Model Regression System

An automated regression testing and evaluation framework for LLM classification pipelines.

## Features
- **Async Eval Engine**: Runs golden dataset cases through LLM classifiers in batches.
- **LLM-as-a-Judge**: Multi-metric evaluation including category accuracy, summary relevance, latency, and token consumption.
- **Regression Detection**: Compares evaluation runs against baselines and flags regressions with warning and critical thresholds.
- **Versioned Prompts**: Supports YAML-based prompt configuration and historical diff tracking.

## Project Structure
```
├── data/              # Golden datasets for evaluation
├── prompts/           # Versioned prompt templates (YAML)
├── src/               # Core engine, schemas, and classifiers
│   ├── classifier.py
│   ├── eval_engine.py
│   └── schema.py
├── tests/             # Unit tests (pytest)
├── .env.example       # Sample environment configuration
├── requirements.txt   # Python dependencies
└── pytest.ini         # Pytest configuration
```

## Setup & Installation

1. Clone repository:
   ```bash
   git clone <REPO_URL>
   cd "model regression system"
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY / OPENAI_API_KEY to .env
   ```

5. (Optional for SLM Jury) Pull local Ollama model:
   ```bash
   ollama pull qwen3:4b
   ```

## SLM Jury Judge System

The repository includes a multi-agent **SLM Jury Judge** (using `qwen3:4b` via Ollama) as a cost-effective, local alternative to cloud-based LLM judges like GPT-4o.

### Architecture & Capabilities
- **Three Profiled Personas**:
  - *Category Precisionist*: Enforces taxonomy rules and checks category match as a hard gate.
  - *Summary Fidelity Checker*: Verifies factual completeness and summary precision against golden references.
  - *Edge Case Skeptic*: Scrutinizes terse, sarcastic, multilingual, and noisy inputs to prevent shallow overconfidence.
- **Independent Parallel Judgments**: Each agent evaluates the case using a deterministic 1–4 rubric with schema-enforced reasoning-first generation.
- **Bias-Mitigated Debate Round**: Agents review anonymized peer reasoning (`Judge 1`, `Judge 2`) in randomized order, instructed to make independent final determinations without peer deference.
- **Ordinal Aggregation**: Final scores are aggregated using `statistics.mode()` with `int(statistics.median())` fallback (no continuous averaging).

### Why SLM Jury vs. GPT-4o?
- **Cost**: Eliminates per-eval API costs for high-frequency CI/CD runs.
- **Privacy & Offline Execution**: Runs 100% locally on workstation or self-hosted CI runner.
- **Ensemble Quality**: Profiled specialization and multi-agent debate approximate frontier model evaluation quality.

### Validating Jury Agreement
Before trusting the SLM Jury in CI/CD pipelines, run the standalone agreement validation script:

```bash
python scripts/validate_jury_agreement.py --dataset data/golden_dataset_v1.json --output data/jury_agreement_results.json
```

#### Key Metrics to Inspect:
1. **Overall Agreement (`within_one_pct`)**: Should exceed **85%** (scores within $\pm 1$ point of GPT-4o).
2. **Hard Failure Agreement (`wrong_cases_within_one_pct`)**: Agreement on cases where GPT-4o score $\le 2$. **Critical metric**: verifies the jury detects real errors rather than rubber-stamping outputs as correct.
3. **Reproducibility (`reproducibility_pct`)**: Verifies that 3 repeated runs with `temperature=0.0` and `seed=42` produce 100% deterministic identical scores.

> [!IMPORTANT]
> **Production Safety Notice**:
> The system defaults to `JUDGE_MODE="gpt4o"`. **TODO:** Keep `JUDGE_MODE="gpt4o"` in production CI until `validate_jury_agreement.py` has been executed on your expanded golden dataset and agreement numbers meet your team's quality threshold (recommended: $\ge 85\%$ overall agreement, $\ge 80\%$ failure agreement, $\ge 90\%$ determinism).

## Phase 4: Alerting & Reporting Layer

The Alerting and Reporting Layer consumes evaluation run outputs to detect regressions, render visual HTML reports with historical trend charts, dispatch Slack alerts, and monitor slow model score drift.

### Core Modules
- **Diff & Scorecard (`reporting/diff.py`)**: Computes case-by-case score diffs (1–4 integer scale), handles newly added test cases gracefully (`new_case: true`, `delta: 0`), and calculates aggregate scorecard metrics (regression rate, average score shift).
- **HTML Report Generator (`reporting/report.py`)**: Uses Jinja2 and Chart.js to produce a self-contained static HTML report featuring run metadata, scorecard metrics, a side-by-side table of **only regressed cases**, and a historical trend line chart across the last N runs.
- **Slack Alerting (`reporting/slack_alert.py`)**: Dispatches Slack Block Kit notifications for per-run regression alerts (with status emojis, headline metrics, and report links) and distinct **Slow Drift Warnings**.
- **Drift Detection (`reporting/drift.py`)**: Computes moving average scores across a configurable window (default: 7 runs) and compares against a stable, persistent baseline (`data/drift_baseline.json`) to detect cumulative performance degradation.

### Environment Variables & Configuration
| Variable | Default | Description |
| :--- | :--- | :--- |
| `SLACK_WEBHOOK_URL` | *None* | Slack Incoming Webhook URL for alerting |
| `WARN_THRESHOLD` | `0.03` (3%) | Regression rate threshold for warnings |
| `CRITICAL_THRESHOLD` | `0.08` (8%) | Regression rate threshold for CI blocking (exit code 1) |
| `DRIFT_WINDOW` | `7` | Number of recent runs for moving average computation |
| `DRIFT_THRESHOLD` | `0.15` | Score drop below baseline that triggers drift alert |
| `TREND_N_RUNS` | `10` | Number of historical runs plotted in the trend chart |
| `REPORT_URL` | *None* | Publicly accessible URL for generated HTML reports |

### Running Per-Run Diff & Report Generation
Run following an evaluation in CI or locally:
```bash
python scripts/run_eval_report.py
```

Options:
- `--new-run <id_or_path>`: Target evaluation run (defaults to latest in `/runs`).
- `--baseline-run <id_or_path>`: Baseline run for comparison (defaults to previous run or main baseline).
- `--slack`: Force sending Slack alert.
- `--report-url <url>`: URL of the hosted HTML report.

*CI Note:* If the regression rate meets or exceeds the critical threshold (`>= 8%`), `run_eval_report.py` exits with status code `1`, blocking CI merge steps.

### Running Scheduled Drift Detection
Run on a separate schedule (e.g., daily cron or nightly GitHub Action):
```bash
python scripts/check_drift.py --window 7 --threshold 0.15
```

Options:
- `--recalibrate`: Recalibrates the persistent baseline from the earliest window of runs.
- `--set-baseline <float>`: Manually sets a specific baseline score (e.g. `3.85`).
- `--slack`: Dispatches Slack alert if drift is detected.

### HTML Report Hosting Options
> [!NOTE]
> **TODO: Report Hosting Setup**: The Slack alert links to `report_url`. Public hosting for the static HTML reports is currently pending configuration. Recommended options:
> 1. **GitHub Actions Artifacts**: Upload HTML reports as workflow artifacts with retention policies.
> 2. **GitHub Pages / Cloudflare Pages**: Publish `reports/` artifacts to a protected internal dashboard site.
> 3. **AWS S3 / Google Cloud Storage Bucket**: Sync HTML reports to an S3/GCS bucket with presigned URLs or CloudFront/Cloud CDN.

## Running Tests
```bash
pytest
```


