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
   # Add your GROQ_API_KEY to .env
   ```

## Running Tests
```bash
pytest
```
