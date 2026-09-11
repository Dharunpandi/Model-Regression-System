"""
Jury agent personas and prompt builders for the SLM Jury Judge system.

Defines three distinct system personas:
1. Category Precisionist: Focuses on classification boundaries, taxomony fit, and cross-category ambiguity.
2. Summary Fidelity Checker: Focuses on factual consistency, core issue extraction, and omission/addition detection.
3. Edge Case Skeptic: Focuses on terse, sarcastic, multilingual, or deceptive inputs, resisting shallow overconfidence.
"""

from typing import List, Dict, Any
from .rubric import SCORING_RUBRIC, FEW_SHOT_CALIBRATION_EXAMPLES


class JuryPersona:
    def __init__(self, name: str, role_title: str, system_prompt: str, description: str):
        self.name = name
        self.role_title = role_title
        self.system_prompt = system_prompt
        self.description = description


# ---------------------------------------------------------------------------
# Persona 1: Category Precisionist
# ---------------------------------------------------------------------------

CATEGORY_PRECISIONIST_PROMPT = """You are a senior evaluation judge specializing as a Category Precisionist.
Your primary expertise is taxonomy enforcement, semantic boundaries, and multi-intent triage.

Your core duties:
1. HARD GATE: Strictly check if the candidate model's predicted category matches the golden reference category.
   If category_match is False, you MUST score the case 1 immediately, regardless of summary quality.
2. Carefully analyze ambiguous cross-category cases (e.g., billing vs. account, technical vs. general).
   If the customer email contains multiple issues, verify whether the candidate adhered to the primary/first-stated category policy.
3. When category matches, verify that the summary aligns with that category's focus without diluting the core issue.
4. You must write out your step-by-step reasoning completely BEFORE outputting your final score.
"""


# ---------------------------------------------------------------------------
# Persona 2: Summary Fidelity Checker
# ---------------------------------------------------------------------------

SUMMARY_FIDELITY_PROMPT = """You are a senior evaluation judge specializing as a Summary Fidelity Checker.
Your primary expertise is linguistic precision, factual consistency, and information preservation.

Your core duties:
1. First verify category_match. If category_match is False, the score MUST be 1.
2. When category matches, rigorously evaluate the candidate summary against the golden reference summary:
   - Score 4: The candidate summary captures the exact core issue AND all material secondary details.
   - Score 3: The core issue is accurately captured, but a secondary detail is omitted or wording differs in subtle emphasis.
   - Score 2: The summary misstates or omits the primary core issue (e.g. mentions only minor trivia or wrong aspect).
   - Score 1: Completely incorrect, hallucinated, or contradictory summary.
3. Watch out for extraneous hallucinations or claims not present in the customer's email.
4. You must write out your step-by-step reasoning completely BEFORE outputting your final score.
"""


# ---------------------------------------------------------------------------
# Persona 3: Edge Case Skeptic
# ---------------------------------------------------------------------------

EDGE_CASE_SKEPTIC_PROMPT = """You are a senior evaluation judge specializing as an Edge Case Skeptic.
Your primary expertise is critical evaluation of tricky, sarcastic, terse, noisy, or non-English inputs.

Your core duties:
1. First verify category_match. If category_match is False, the score MUST be 1.
2. Be highly skeptical of confident, fluent answers that miss subtext, irony, or language nuances.
   - For sarcastic inputs: Verify if the model understood genuine complaint vs literal words.
   - For multilingual inputs: Verify whether the translation and core meaning are fully preserved.
   - For short/empty/noisy inputs: Check if the model hallucinated details that were never stated.
   - For multi-issue emails: Check if the model got distracted by secondary complaints.
3. Apply the 1-4 rubric strictly. Do not give the candidate the benefit of the doubt on ambiguous failures.
4. You must write out your step-by-step reasoning completely BEFORE outputting your final score.
"""


JURY_PERSONAS: List[JuryPersona] = [
    JuryPersona(
        name="Category Precisionist",
        role_title="Category Precisionist",
        system_prompt=CATEGORY_PRECISIONIST_PROMPT,
        description="Focuses on category boundaries, multi-intent triage, and strict hard-gate enforcement."
    ),
    JuryPersona(
        name="Summary Fidelity Checker",
        role_title="Summary Fidelity Checker",
        system_prompt=SUMMARY_FIDELITY_PROMPT,
        description="Focuses on factual accuracy, core issue extraction, and omission/hallucination detection."
    ),
    JuryPersona(
        name="Edge Case Skeptic",
        role_title="Edge Case Skeptic",
        system_prompt=EDGE_CASE_SKEPTIC_PROMPT,
        description="Focuses on nuanced, sarcastic, terse, multilingual, or ambiguous edge cases."
    ),
]


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_initial_prompt(
    email: str,
    golden_category: str,
    golden_summary: str,
    model_category: str,
    model_summary: str,
    calibration_examples: List[Dict[str, Any]] = None,
) -> str:
    """
    Constructs the initial evaluation prompt for a jury agent.
    Includes the rubric, calibration examples, and the target test case.
    """
    if calibration_examples is None:
        calibration_examples = FEW_SHOT_CALIBRATION_EXAMPLES

    few_shot_text = ""
    for idx, ex in enumerate(calibration_examples, 1):
        few_shot_text += f"""
--- Calibration Example {idx} ({ex.get('notes', '')}) ---
Customer Email: {ex['email']}
Golden Category: {ex['golden_category']}
Golden Summary: {ex['golden_summary']}
Candidate Output Category: {ex['model_category']}
Candidate Output Summary: {ex['model_summary']}
Expected Evaluation:
{{
  "category_match": {str(ex['category_match']).lower()},
  "reasoning": "{ex['reasoning']}",
  "score": {ex['score']}
}}
"""

    prompt = f"""You are evaluating an AI email classifier output against a golden reference.

{SCORING_RUBRIC}

### Hand-Labeled Calibration Examples
{few_shot_text}

### Test Case to Evaluate
Customer Email: {email}
Golden Category: {golden_category}
Golden Summary: {golden_summary}

Candidate Model Output Category: {model_category}
Candidate Model Output Summary: {model_summary}

### Evaluation Instructions
1. First, check if Candidate Category matches Golden Category (`category_match`).
2. If `category_match` is False, the score MUST be 1.
3. If `category_match` is True, evaluate the candidate summary against the golden summary using the 1-4 scale.
4. Output valid JSON with keys in this EXACT order:
   - "category_match": boolean
   - "reasoning": detailed reasoning string (written completely BEFORE deciding score)
   - "score": integer 1-4
"""
    return prompt.strip()


def build_debate_prompt(
    email: str,
    golden_category: str,
    golden_summary: str,
    model_category: str,
    model_summary: str,
    own_initial: Dict[str, Any],
    anonymized_peers: List[Dict[str, Any]],
) -> str:
    """
    Constructs the debate prompt for a jury agent.
    Presents anonymized peer evaluations (e.g. Judge 1, Judge 2) and reminds the agent
    to form an independent final judgment grounded in the source data.
    """
    peers_text = ""
    for peer in anonymized_peers:
        peers_text += f"""
--- {peer['label']} Evaluation ---
Category Match: {peer.get('category_match')}
Reasoning: {peer.get('reasoning')}
Proposed Score: {peer.get('score')}
"""

    prompt = f"""You are participating in the final debate round for evaluating an AI email classifier.

{SCORING_RUBRIC}

### Test Case Under Evaluation
Customer Email: {email}
Golden Category: {golden_category}
Golden Summary: {golden_summary}

Candidate Model Output Category: {model_category}
Candidate Model Output Summary: {model_summary}

### Your Initial Evaluation
Category Match: {own_initial.get('category_match')}
Your Initial Reasoning: {own_initial.get('reasoning')}
Your Initial Score: {own_initial.get('score')}

### Anonymized Peer Judges' Input
{peers_text}

### Debate & Final Judgment Instructions
1. Review the peer judges' arguments and compare them with your initial evaluation.
2. CRITICAL INSTRUCTION: Your final score MUST be based on the actual test case and golden standard, NOT blind deference to peer judges. Their evaluations are for reference only.
3. If a peer noticed a factual detail or category constraint you overlooked, you may revise your score; otherwise, hold your ground.
4. Output valid JSON with keys in this EXACT order:
   - "comparison_notes": comparative analysis of peer perspectives vs your initial thoughts
   - "revised": boolean (true if changing score, false if keeping original)
   - "final_reasoning": independent justification grounded in the test case
   - "final_score": integer 1-4
"""
    return prompt.strip()
