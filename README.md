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

## Running Tests
```bash
pytest
```

