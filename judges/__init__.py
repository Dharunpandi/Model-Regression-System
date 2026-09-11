"""
Judges module: Multi-agent SLM jury and reference GPT-4o evaluation judges.
"""

from .rubric import (
    SCORING_RUBRIC,
    INITIAL_JUDGMENT_SCHEMA,
    DEBATE_JUDGMENT_SCHEMA,
    InitialJudgment,
    DebateJudgment,
    FEW_SHOT_CALIBRATION_EXAMPLES,
)
from .profiles import JURY_PERSONAS, JuryPersona
from .slm_jury import slm_jury_judge, aggregate_jury_scores
from .gpt4o import gpt4o_judge

__all__ = [
    "SCORING_RUBRIC",
    "INITIAL_JUDGMENT_SCHEMA",
    "DEBATE_JUDGMENT_SCHEMA",
    "InitialJudgment",
    "DebateJudgment",
    "FEW_SHOT_CALIBRATION_EXAMPLES",
    "JURY_PERSONAS",
    "JuryPersona",
    "slm_jury_judge",
    "aggregate_jury_scores",
    "gpt4o_judge",
]
