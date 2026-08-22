"""
Interface contract for the email classifier feature.

This is the "shape" your eval pipeline is built against. Keep it stable —
if you change these fields later, every downstream eval script and the
golden dataset format needs to change with it. That's exactly the kind of
break this system is meant to catch.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    general = "general"


class FewShotExample(BaseModel):
    input: str
    output: dict


class PromptConfig(BaseModel):
    """Loaded from a versioned YAML file in /prompts."""

    version_id: str
    created_at: datetime
    description: Optional[str] = None
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ClassificationInput(BaseModel):
    """What goes into the feature under test."""

    email_text: str


class ClassificationOutput(BaseModel):
    """What the feature must return. This is what LLM structured output
    gets validated against, and what the eval engine compares to ground
    truth in the golden dataset."""

    category: Category
    summary: str = Field(..., max_length=300)


class ClassificationResult(BaseModel):
    """A single run's full record — used by the eval engine, not just the
    raw feature call. Keeps latency/token/model metadata alongside the
    output so scoring and diffing have everything they need."""

    input: ClassificationInput
    output: ClassificationOutput
    prompt_version: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class GoldenCase(BaseModel):
    """One hand-labeled test case in the golden dataset."""

    id: str
    input: str
    expected_category: Category
    expected_summary: str
    difficulty: Difficulty
    notes: str


class GoldenDataset(BaseModel):
    """The full versioned golden dataset file."""

    dataset_version: str
    created_at: datetime
    description: Optional[str] = None
    cases: list[GoldenCase]

    @classmethod
    def from_json(cls, path: str | Path) -> "GoldenDataset":
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class CaseScore(BaseModel):
    """Every scoring dimension for a single case, in a single run."""

    case_id: str
    difficulty: Difficulty
    input: str

    actual_category: Category
    expected_category: Category
    category_match: bool  # binary, exact match

    actual_summary: str
    expected_summary: str
    summary_relevance_score: int = Field(..., ge=1, le=5)  # LLM-as-judge

    latency_ms: float
    input_tokens: int
    output_tokens: int

    passed: bool  # overall pass/fail for this case — category_match AND
    # summary_relevance_score >= threshold (engine decides threshold)


class EvalRun(BaseModel):
    """One full evaluation run — every case scored, plus run-level metadata.
    Saved to disk so future runs can diff against it."""

    run_id: str
    prompt_version: str
    dataset_version: str
    model: str
    timestamp: datetime

    case_scores: list[CaseScore]

    pass_rate: float  # fraction of cases that passed, 0.0-1.0
    category_accuracy: dict  # {"billing": 0.95, "technical": 0.88, ...}
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalRun":
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class FlippedCase(BaseModel):
    """A case whose pass/fail status changed between two runs."""

    case_id: str
    input: str
    previous_passed: bool
    current_passed: bool
    previous_category: Category
    current_category: Category


class RunComparison(BaseModel):
    """The diff between two eval runs — this is the core output the
    Slack alert and HTML report are both built from."""

    current_run_id: str
    previous_run_id: str

    pass_rate_delta: float  # current - previous, can be negative
    category_accuracy_delta: dict  # per-category delta

    regressions: list[FlippedCase]  # passed before, now fails
    improvements: list[FlippedCase]  # failed before, now passes

    severity: str  # "ok" | "warning" | "critical"
    warning_threshold: float
    critical_threshold: float