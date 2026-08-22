"""
Evaluation engine — Phase 3.

Three jobs, in order:
1. run_eval()     — run every golden case through the classifier, async/batched.
2. score_case()    — grade each case on category match, summary relevance
                      (LLM-as-judge), latency, and tokens.
3. compare_runs()  — diff the current run against the previous one and flag
                      regressions/improvements with a configurable
                      warning/critical threshold.

Each run is saved to /runs as its own JSON file, named by prompt version +
timestamp, so you always have history to diff against.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .schema import (
    CaseScore,
    ClassificationInput,
    EvalRun,
    FlippedCase,
    GoldenDataset,
    PromptConfig,
    RunComparison,
)

load_dotenv()

_client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# How many requests fire at once. Groq's free tier caps at ~30 req/min, so
# keep this conservative rather than blasting all 50-100 cases at once.
MAX_CONCURRENCY = 5

# A case only counts as an overall "pass" if the summary judge score meets
# this bar. Category match alone isn't enough — a right category with a
# nonsense summary shouldn't pass.
SUMMARY_PASS_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Step 1: run every case through the classifier (async, batched)
# ---------------------------------------------------------------------------

async def _classify_one(email_text: str, prompt_config: PromptConfig, model: str):
    """One call to the classifier feature. Mirrors classify_email() from
    classifier.py but async, since the eval engine needs to fire many
    requests concurrently rather than one at a time."""
    from .classifier import _build_messages

    messages = _build_messages(prompt_config, email_text)

    start = time.perf_counter()
    response = await _client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw = json.loads(response.choices[0].message.content)
    usage = response.usage

    return {
        "category": raw.get("category"),
        "summary": raw.get("summary", ""),
        "latency_ms": latency_ms,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


async def _judge_summary(candidate_summary: str, expected_summary: str, model: str) -> int:
    """LLM-as-judge: rates how well the candidate summary captures the same
    meaning as the human-written expected summary, 1-5. This is separate
    from category matching because a summary can be badly worded even when
    the category is correct."""

    judge_prompt = (
        "You are grading a customer-support email summary against a "
        "reference summary written by a human.\n\n"
        f"Reference summary: {expected_summary}\n"
        f"Candidate summary: {candidate_summary}\n\n"
        "Rate how well the candidate captures the same core issue as the "
        "reference, on a scale of 1-5:\n"
        "5 = same issue, same key details\n"
        "3 = same general issue, missing or vague on details\n"
        "1 = different issue entirely, or nonsensical\n\n"
        'Respond ONLY with JSON: {"score": <1-5>}'
    )

    response = await _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content)
    return int(parsed.get("score", 1))


async def _run_single_case(case, prompt_config: PromptConfig, model: str, semaphore) -> CaseScore:
    async with semaphore:
        result = await _classify_one(case.input, prompt_config, model)
        judge_score = await _judge_summary(result["summary"], case.expected_summary, model)

    category_match = result["category"] == case.expected_category.value
    passed = category_match and judge_score >= SUMMARY_PASS_THRESHOLD

    return CaseScore(
        case_id=case.id,
        difficulty=case.difficulty,
        input=case.input,
        actual_category=result["category"],
        expected_category=case.expected_category,
        category_match=category_match,
        actual_summary=result["summary"],
        expected_summary=case.expected_summary,
        summary_relevance_score=judge_score,
        latency_ms=result["latency_ms"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        passed=passed,
    )


async def run_eval(
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    model: str = "llama-3.3-70b-versatile",
) -> EvalRun:
    """Runs every case in the golden dataset through the classifier,
    scores each one, and returns a full EvalRun record."""

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        _run_single_case(case, prompt_config, model, semaphore)
        for case in dataset.cases
    ]
    case_scores = await asyncio.gather(*tasks)

    pass_rate = sum(c.passed for c in case_scores) / len(case_scores)

    category_totals: dict = {}
    for c in case_scores:
        cat = c.expected_category.value
        category_totals.setdefault(cat, {"total": 0, "correct": 0})
        category_totals[cat]["total"] += 1
        if c.category_match:
            category_totals[cat]["correct"] += 1
    category_accuracy = {
        cat: (v["correct"] / v["total"]) for cat, v in category_totals.items()
    }

    run = EvalRun(
        run_id=f"{prompt_config.version_id}_{uuid.uuid4().hex[:8]}",
        prompt_version=prompt_config.version_id,
        dataset_version=dataset.dataset_version,
        model=model,
        timestamp=datetime.now(timezone.utc),
        case_scores=case_scores,
        pass_rate=pass_rate,
        category_accuracy=category_accuracy,
        avg_latency_ms=sum(c.latency_ms for c in case_scores) / len(case_scores),
        total_input_tokens=sum(c.input_tokens for c in case_scores),
        total_output_tokens=sum(c.output_tokens for c in case_scores),
    )
    return run


def save_run(run: EvalRun) -> Path:
    path = RUNS_DIR / f"{run.run_id}.json"
    with open(path, "w") as f:
        f.write(run.model_dump_json(indent=2))
    return path


def load_latest_run(exclude_run_id: str | None = None) -> EvalRun | None:
    """Finds the most recent saved run, for diffing against. Excludes the
    run you just created so you don't diff a run against itself."""
    runs = sorted(RUNS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    for path in runs:
        run = EvalRun.from_json(path)
        if run.run_id != exclude_run_id:
            return run
    return None


# ---------------------------------------------------------------------------
# Step 3 & 4: compare two runs, flag regressions, apply significance thresholds
# ---------------------------------------------------------------------------

def compare_runs(
    current: EvalRun,
    previous: EvalRun,
    warning_threshold: float = 0.03,
    critical_threshold: float = 0.08,
) -> RunComparison:
    """Diffs two runs case-by-case. This is the core value of the whole
    system — not just "what's the score" but "exactly what changed and
    why should I care.\""""

    previous_by_id = {c.case_id: c for c in previous.case_scores}

    regressions = []
    improvements = []

    for c in current.case_scores:
        prev = previous_by_id.get(c.case_id)
        if prev is None:
            continue  # case is new since the last run, nothing to diff

        if prev.passed and not c.passed:
            regressions.append(
                FlippedCase(
                    case_id=c.case_id,
                    input=c.input,
                    previous_passed=True,
                    current_passed=False,
                    previous_category=prev.actual_category,
                    current_category=c.actual_category,
                )
            )
        elif not prev.passed and c.passed:
            improvements.append(
                FlippedCase(
                    case_id=c.case_id,
                    input=c.input,
                    previous_passed=False,
                    current_passed=True,
                    previous_category=prev.actual_category,
                    current_category=c.actual_category,
                )
            )

    pass_rate_delta = current.pass_rate - previous.pass_rate

    category_accuracy_delta = {
        cat: current.category_accuracy.get(cat, 0) - previous.category_accuracy.get(cat, 0)
        for cat in set(current.category_accuracy) | set(previous.category_accuracy)
    }

    # Significance: a drop is only "signal" if it clears the threshold.
    # This stops a 1-2 case flip out of 80 from triggering a false alarm.
    abs_delta = abs(pass_rate_delta)
    if abs_delta >= critical_threshold:
        severity = "critical"
    elif abs_delta >= warning_threshold:
        severity = "warning"
    else:
        severity = "ok"

    return RunComparison(
        current_run_id=current.run_id,
        previous_run_id=previous.run_id,
        pass_rate_delta=pass_rate_delta,
        category_accuracy_delta=category_accuracy_delta,
        regressions=regressions,
        improvements=improvements,
        severity=severity,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def main():
        prompt_config = PromptConfig.from_yaml("prompts/v1.yaml")
        dataset = GoldenDataset.from_json("data/golden_dataset_v1.json")

        print(f"Running eval: prompt {prompt_config.version_id}, "
              f"{len(dataset.cases)} cases...")
        run = await run_eval(prompt_config, dataset)
        path = save_run(run)
        print(f"Saved run to {path}")
        print(f"Pass rate: {run.pass_rate:.1%}")
        print(f"Category accuracy: {run.category_accuracy}")

        previous = load_latest_run(exclude_run_id=run.run_id)
        if previous:
            comparison = compare_runs(run, previous)
            print(f"\nCompared to previous run ({previous.run_id}):")
            print(f"Pass rate delta: {comparison.pass_rate_delta:+.1%}")
            print(f"Severity: {comparison.severity}")
            print(f"Regressions: {len(comparison.regressions)}")
            print(f"Improvements: {len(comparison.improvements)}")
        else:
            print("\nNo previous run found — this is your baseline.")

    asyncio.run(main())