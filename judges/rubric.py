"""
Shared scoring rubric, output schemas, and few-shot calibration examples for LLM/SLM judges.

The scoring rubric uses a deterministic 1-4 scale where category_match acts as a hard gate.
Output schemas strictly enforce that reasoning/comparison notes are generated before
the numeric score is determined.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Deterministic 1-4 Scoring Rubric Constant
# ---------------------------------------------------------------------------

SCORING_RUBRIC = """### Deterministic 1-4 Scoring Rubric

**Hard Gate**: `category_match` MUST be evaluated first. If `category_match` is False, the score is automatically 1, regardless of summary quality.

- **Score 1 (Incorrect)**: `category_match` is False, regardless of summary. OR `category_match` is True, but the summary is completely wrong/nonsensical.
- **Score 2 (Partially Accurate)**: `category_match` is True, but the summary omits or misstates the core issue.
- **Score 3 (Mostly Matching)**: `category_match` is True, core issue is accurately captured, but a secondary detail is missing or emphasis differs slightly.
- **Score 4 (Exact Match)**: `category_match` is True, core issue AND all material secondary details are present (wording may differ, content and semantics must match).
"""


# ---------------------------------------------------------------------------
# Structured Output Schemas (Field ordering enforces reasoning before score)
# ---------------------------------------------------------------------------

class InitialJudgment(BaseModel):
    """
    Schema for initial independent judgment.
    Field order MUST be: category_match -> reasoning -> score.
    """
    category_match: bool = Field(
        ...,
        description="Boolean indicating whether the model's assigned category matches the golden category exactly."
    )
    reasoning: str = Field(
        ...,
        description="Detailed step-by-step reasoning written completely before deciding the score. Must evaluate category match first as a hard gate, then inspect summary fidelity against the rubric."
    )
    score: int = Field(
        ...,
        ge=1,
        le=4,
        description="Final integer score on the 1-4 rubric scale (1=Incorrect, 2=Partially Accurate, 3=Mostly Matching, 4=Exact Match)."
    )


class DebateJudgment(BaseModel):
    """
    Schema for debate round judgment.
    Field order MUST be: comparison_notes -> revised -> final_reasoning -> final_score.
    """
    comparison_notes: str = Field(
        ...,
        description="Comparative analysis evaluating the anonymized peer judges' scores and reasoning against own initial stance."
    )
    revised: bool = Field(
        ...,
        description="True if revising initial score/position after considering peer perspectives; False if maintaining original score."
    )
    final_reasoning: str = Field(
        ...,
        description="Final independent reasoning grounded in the actual email and candidate output, rather than blind deference to peers."
    )
    final_score: int = Field(
        ...,
        ge=1,
        le=4,
        description="Final post-debate integer score on the 1-4 rubric scale."
    )


# JSON Schemas for Ollama structured output parameter `format`
INITIAL_JUDGMENT_SCHEMA: Dict[str, Any] = InitialJudgment.model_json_schema()
DEBATE_JUDGMENT_SCHEMA: Dict[str, Any] = DebateJudgment.model_json_schema()


# ---------------------------------------------------------------------------
# Few-Shot Calibration Examples (Hand-labeled calibration data)
# ---------------------------------------------------------------------------

FEW_SHOT_CALIBRATION_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "calib_001",
        "email": "Hi, I was charged $49.99 twice on my card this month for the Pro plan. Can you refund the duplicate charge? My account email is on file.",
        "golden_category": "billing",
        "golden_summary": "Customer was double-charged for their Pro plan subscription and wants a refund.",
        "model_category": "billing",
        "model_summary": "Customer was charged twice for their Pro plan and is asking for a refund.",
        "category_match": True,
        "reasoning": "Category matches 'billing' exactly. The summary correctly captures the core issue (double charge for Pro plan) and the specific request (refund). All material details are present.",
        "score": 4,
        "notes": "Score 4 example: Exact semantic match on both category and summary."
    },
    {
        "id": "calib_002",
        "email": "the app keeps crashing when i open the reports tab, happens every single time, using android",
        "golden_category": "technical",
        "golden_summary": "The app consistently crashes when the customer opens the reports tab on Android.",
        "model_category": "technical",
        "model_summary": "The mobile application crashes when opening reports.",
        "category_match": True,
        "reasoning": "Category matches 'technical'. The summary captures the core issue (crash when opening reports tab), but omits the secondary detail that it occurs specifically on Android and happens every time.",
        "score": 3,
        "notes": "Score 3 example: Category match + core issue captured, minor secondary detail (Android OS) omitted."
    },
    {
        "id": "calib_003",
        "email": "cant login again. also billed wrong last week too",
        "golden_category": "account",
        "golden_summary": "Customer cannot log in and separately mentions a billing issue from last week.",
        "model_category": "account",
        "model_summary": "Customer has an inquiry about their monthly bill payment.",
        "category_match": True,
        "reasoning": "Category matches 'account'. However, the candidate summary completely misstates the core issue by focusing only on billing and failing to mention the primary issue that the customer cannot log in.",
        "score": 2,
        "notes": "Score 2 example: Category match is True, but summary misstates or omits the core issue."
    },
    {
        "id": "calib_004",
        "email": "great, another update that breaks everything. thanks a lot",
        "golden_category": "technical",
        "golden_summary": "Customer is frustrated that a recent update introduced new problems.",
        "model_category": "general",
        "model_summary": "Customer expressing frustration about a recent product update.",
        "category_match": False,
        "reasoning": "Category match failed: model predicted 'general' instead of golden 'technical'. Per the rubric hard gate, when category_match is False, the score must be 1 regardless of summary quality.",
        "score": 1,
        "notes": "Score 1 example: Category mismatch triggers automatic hard gate failure."
    },
    {
        "id": "calib_005",
        "email": "hola, no puedo cambiar mi contraseña, el enlace de recuperación no funciona",
        "golden_category": "account",
        "golden_summary": "Customer cannot change their password because the recovery link isn't working.",
        "model_category": "account",
        "model_summary": "Customer is unable to reset their password due to a broken recovery link.",
        "category_match": True,
        "reasoning": "Category matches 'account'. The summary accurately translates and captures both the core issue (password reset failure) and root cause (broken recovery link).",
        "score": 4,
        "notes": "Score 4 multilingual example: Spanish input correctly parsed and summarized in English."
    },
    {
        "id": "calib_006",
        "email": "just wanted to say the new dashboard redesign looks really clean, nice work",
        "golden_category": "general",
        "golden_summary": "Customer is giving positive feedback about the new dashboard redesign.",
        "model_category": "general",
        "model_summary": "Customer sent general feedback.",
        "category_match": True,
        "reasoning": "Category matches 'general'. The summary captures the broad intent (feedback) but is overly vague and omits the key detail that it is positive praise for the dashboard redesign.",
        "score": 3,
        "notes": "Score 3 example: Vague summary missing specific focus area."
    },
    # --- Placeholders for additional user-provided calibration examples ---
    # {
    #     "id": "calib_007_placeholder",
    #     "email": "...",
    #     "golden_category": "...",
    #     "golden_summary": "...",
    #     "model_category": "...",
    #     "model_summary": "...",
    #     "category_match": True,
    #     "reasoning": "...",
    #     "score": 4,
    #     "notes": "TODO: Add custom golden edge case."
    # },
]
