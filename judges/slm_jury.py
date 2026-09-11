"""
SLM Jury Judge implementation using local Small Language Models (Qwen3-4B via Ollama).

Features:
- Three profiled agents (Category Precisionist, Summary Fidelity Checker, Edge Case Skeptic).
- Parallel independent initial judgment with schema-enforced reasoning-first constraint.
- Bias-mitigated debate stage with anonymized and randomly shuffled peer judgments.
- Deterministic mode/median ordinal aggregation (no continuous averaging).
- Strict temperature=0.0 and seed=42 for reproducibility.
"""

import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import ollama
except ImportError:
    ollama = None

from .profiles import JURY_PERSONAS, JuryPersona, build_debate_prompt, build_initial_prompt
from .rubric import (
    DEBATE_JUDGMENT_SCHEMA,
    FEW_SHOT_CALIBRATION_EXAMPLES,
    INITIAL_JUDGMENT_SCHEMA,
    DebateJudgment,
    InitialJudgment,
)


def _extract_model_output(model_output: Any) -> Tuple[str, str]:
    """
    Normalizes candidate model output from dict, Pydantic model, or raw string/object
    into (category, summary) tuple.
    """
    if isinstance(model_output, dict):
        category = str(model_output.get("category", "")).strip()
        summary = str(model_output.get("summary", "")).strip()
    elif hasattr(model_output, "output"):  # ClassificationResult
        output_obj = model_output.output
        category = str(getattr(output_obj, "category", "")).strip()
        summary = str(getattr(output_obj, "summary", "")).strip()
    elif hasattr(model_output, "category") and hasattr(model_output, "summary"):  # ClassificationOutput
        category = str(getattr(model_output, "category", "")).strip()
        summary = str(getattr(model_output, "summary", "")).strip()
    else:
        category = "unknown"
        summary = str(model_output)
        
    # In case category is an Enum or has value attribute
    if hasattr(category, "value"):
        category = str(category.value)

    return category, summary


def _call_ollama(
    messages: List[Dict[str, str]],
    format_schema: Dict[str, Any],
    model_name: str = "qwen3:4b",
    client: Any = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes a structured chat completion against Ollama with temperature=0.0 and seed=42.
    """
    if options is None:
        options = {"temperature": 0.0, "seed": 42}

    if client is not None:
        response = client.chat(
            model=model_name,
            messages=messages,
            format=format_schema,
            options=options,
        )
    else:
        if ollama is None:
            raise ImportError(
                "The 'ollama' package is required to run the SLM Jury Judge. "
                "Install it via `pip install ollama`."
            )
        response = ollama.chat(
            model=model_name,
            messages=messages,
            format=format_schema,
            options=options,
        )

    # Ollama returns a mapping with message content as JSON string
    if isinstance(response, dict):
        content = response["message"]["content"]
    else:
        content = response.message.content

    if isinstance(content, str):
        parsed = json.loads(content)
    else:
        parsed = content

    return parsed


# ---------------------------------------------------------------------------
# Stage 1: Independent Initial Judgment
# ---------------------------------------------------------------------------

def run_agent_initial_judgment(
    persona: JuryPersona,
    email: str,
    golden_category: str,
    golden_summary: str,
    model_category: str,
    model_summary: str,
    calibration_examples: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "qwen3:4b",
    client: Any = None,
) -> Dict[str, Any]:
    """
    Runs the initial evaluation for a single jury agent.
    """
    user_prompt = build_initial_prompt(
        email=email,
        golden_category=golden_category,
        golden_summary=golden_summary,
        model_category=model_category,
        model_summary=model_summary,
        calibration_examples=calibration_examples or FEW_SHOT_CALIBRATION_EXAMPLES,
    )

    messages = [
        {"role": "system", "content": persona.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = _call_ollama(
        messages=messages,
        format_schema=INITIAL_JUDGMENT_SCHEMA,
        model_name=model_name,
        client=client,
        options={"temperature": 0.0, "seed": 42},
    )

    # Validate with Pydantic model to enforce structure and field types
    validated = InitialJudgment(**raw_output)

    # Hard gate post-validation: if category_match is False, force score to 1 as invariant
    enforced_score = 1 if not validated.category_match else int(validated.score)

    return {
        "agent": persona.name,
        "category_match": validated.category_match,
        "reasoning": validated.reasoning,
        "score": enforced_score,
    }


# ---------------------------------------------------------------------------
# Stage 2: Bias-Mitigated Debate Round
# ---------------------------------------------------------------------------

def run_agent_debate(
    persona: JuryPersona,
    email: str,
    golden_category: str,
    golden_summary: str,
    model_category: str,
    model_summary: str,
    own_initial: Dict[str, Any],
    all_initial_judgments: List[Dict[str, Any]],
    model_name: str = "qwen3:4b",
    client: Any = None,
) -> Dict[str, Any]:
    """
    Runs the debate stage for a single agent.
    Peer judgments are anonymized ('Judge 1', 'Judge 2') and randomly shuffled
    to prevent positional bias and deference to persona labels.
    """
    # 1. Filter out own judgment to obtain peer evaluations
    peers = [j for j in all_initial_judgments if j["agent"] != persona.name]

    # 2. Make a shallow copy and randomly shuffle the peers list
    peers_shuffled = list(peers)
    random.shuffle(peers_shuffled)

    # 3. Anonymize peer identities
    anonymized_peers = []
    for idx, peer in enumerate(peers_shuffled, 1):
        anonymized_peers.append({
            "label": f"Judge {idx}",
            "category_match": peer.get("category_match"),
            "reasoning": peer.get("reasoning"),
            "score": peer.get("score"),
        })

    # 4. Build prompt and invoke LLM
    user_prompt = build_debate_prompt(
        email=email,
        golden_category=golden_category,
        golden_summary=golden_summary,
        model_category=model_category,
        model_summary=model_summary,
        own_initial=own_initial,
        anonymized_peers=anonymized_peers,
    )

    messages = [
        {"role": "system", "content": persona.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output = _call_ollama(
        messages=messages,
        format_schema=DEBATE_JUDGMENT_SCHEMA,
        model_name=model_name,
        client=client,
        options={"temperature": 0.0, "seed": 42},
    )

    validated = DebateJudgment(**raw_output)

    # Hard gate invariant: if initial category_match was False and no revision occurred, score stays 1
    final_score = int(validated.final_score)
    if not own_initial.get("category_match", True) and not validated.revised:
        final_score = 1

    return {
        "agent": persona.name,
        "comparison_notes": validated.comparison_notes,
        "revised": validated.revised,
        "final_reasoning": validated.final_reasoning,
        "final_score": final_score,
    }


# ---------------------------------------------------------------------------
# Stage 3: Aggregation (Mode with Median Fallback)
# ---------------------------------------------------------------------------

def aggregate_jury_scores(scores: List[int]) -> int:
    """
    Aggregates the ordinal 1-4 scores from jury members.
    Uses statistics.mode() when a single unique mode exists.
    If there is no unique mode (e.g., all 3 judges differ [2, 3, 4]),
    falls back to int(statistics.median()).
    Does NOT use continuous averaging.
    """
    if not scores:
        return 1

    # In Python 3.8+, statistics.mode returns the first item on ties without raising StatisticsError.
    # To strictly detect when no single unique mode exists, inspect multimode:
    modes = statistics.multimode(scores)
    if len(modes) == 1:
        return int(modes[0])

    # No unique mode: fall back to median
    try:
        return int(statistics.median(scores))
    except statistics.StatisticsError:
        return int(scores[0])



# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def slm_jury_judge(
    email: str,
    golden_category: str,
    golden_summary: str,
    model_output: Union[Dict[str, Any], Any],
    model_name: str = "qwen3:4b",
    client: Any = None,
    calibration_examples: Optional[List[Dict[str, Any]]] = None,
    max_workers: int = 3,
) -> Dict[str, Any]:
    """
    Main entry point for SLM Jury Judge.
    Runs 3 profiled agents independently, conducts a bias-mitigated debate round,
    and aggregates final scores using statistics.mode (fallback to median).

    Matches the signature and return format required for swappable evaluation:
    Returns:
    {
        "score": <1-4 int>,
        "trail": {
            "initial": [ ... ],
            "debate": [ ... ]
        }
    }
    """
    model_cat, model_sum = _extract_model_output(model_output)

    # --- Phase 1: Parallel Independent Initial Judgments ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        initial_futures = [
            executor.submit(
                run_agent_initial_judgment,
                persona=persona,
                email=email,
                golden_category=golden_category,
                golden_summary=golden_summary,
                model_category=model_cat,
                model_summary=model_sum,
                calibration_examples=calibration_examples,
                model_name=model_name,
                client=client,
            )
            for persona in JURY_PERSONAS
        ]
        initial_judgments = [f.result() for f in initial_futures]

    initial_by_agent = {j["agent"]: j for j in initial_judgments}

    # --- Phase 2: Parallel Bias-Mitigated Debate Round ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        debate_futures = [
            executor.submit(
                run_agent_debate,
                persona=persona,
                email=email,
                golden_category=golden_category,
                golden_summary=golden_summary,
                model_category=model_cat,
                model_summary=model_sum,
                own_initial=initial_by_agent[persona.name],
                all_initial_judgments=initial_judgments,
                model_name=model_name,
                client=client,
            )
            for persona in JURY_PERSONAS
        ]
        debate_judgments = [f.result() for f in debate_futures]

    # --- Phase 3: Aggregation ---
    final_scores = [d["final_score"] for d in debate_judgments]
    consensus_score = aggregate_jury_scores(final_scores)

    return {
        "score": consensus_score,
        "trail": {
            "initial": initial_judgments,
            "debate": debate_judgments,
        }
    }
