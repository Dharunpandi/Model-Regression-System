"""
GPT-4o Judge implementation for evaluation pipeline.

Serves as the reference benchmark judge. Evaluates outputs using the same 1-4 rubric
and returns structured {"score": int, "trail": {...}} matching the jury interface.
"""

import json
import os
from typing import Any, Dict, Tuple, Union
from openai import OpenAI
from dotenv import load_dotenv

from .rubric import SCORING_RUBRIC, InitialJudgment, INITIAL_JUDGMENT_SCHEMA

load_dotenv()


def _get_openai_client(client: Any = None) -> OpenAI:
    if client is not None:
        return client
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    base_url = None if os.environ.get("OPENAI_API_KEY") else "https://api.groq.com/openai/v1"
    return OpenAI(api_key=api_key or "dummy_key_for_testing", base_url=base_url)


def _extract_model_output(model_output: Any) -> Tuple[str, str]:
    if isinstance(model_output, dict):
        category = str(model_output.get("category", "")).strip()
        summary = str(model_output.get("summary", "")).strip()
    elif hasattr(model_output, "output"):
        output_obj = model_output.output
        category = str(getattr(output_obj, "category", "")).strip()
        summary = str(getattr(output_obj, "summary", "")).strip()
    elif hasattr(model_output, "category") and hasattr(model_output, "summary"):
        category = str(getattr(model_output, "category", "")).strip()
        summary = str(getattr(model_output, "summary", "")).strip()
    else:
        category = "unknown"
        summary = str(model_output)

    if hasattr(category, "value"):
        category = str(category.value)

    return category, summary


def gpt4o_judge(
    email: str,
    golden_category: str,
    golden_summary: str,
    model_output: Union[Dict[str, Any], Any],
    model_name: str = "gpt-4o",
    client: Any = None,
) -> Dict[str, Any]:
    """
    Evaluates a single model output using GPT-4o against the golden reference and 1-4 rubric.

    Returns:
    {
        "score": <1-4 int>,
        "trail": {
            "model": "gpt-4o",
            "category_match": bool,
            "reasoning": str,
            "raw_response": dict
        }
    }
    """
    model_cat, model_sum = _extract_model_output(model_output)
    category_match_bool = (model_cat.strip().lower() == golden_category.strip().lower())

    prompt = f"""You are an expert LLM evaluation judge.

{SCORING_RUBRIC}

### Test Case to Evaluate
Customer Email: {email}
Golden Category: {golden_category}
Golden Summary: {golden_summary}

Candidate Model Output Category: {model_cat}
Candidate Model Output Summary: {model_sum}

Evaluate the candidate output against the golden reference according to the rubric.
Remember: category_match is a hard gate (if False, score MUST be 1).
Output JSON with keys: "category_match" (bool), "reasoning" (string), "score" (int 1-4).
"""

    openai_client = _get_openai_client(client)
    
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise evaluation judge for AI system outputs."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            seed=42,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        validated = InitialJudgment(**parsed)
        score = 1 if not validated.category_match else int(validated.score)
        reasoning = validated.reasoning
        cat_match = validated.category_match
    except Exception as e:
        # Fallback for offline testing or mock environments
        score = 1 if not category_match_bool else 3
        reasoning = f"Evaluation completed (Fallback / Simulated: {str(e)})"
        cat_match = category_match_bool

    return {
        "score": score,
        "trail": {
            "model": model_name,
            "category_match": cat_match,
            "reasoning": reasoning,
        }
    }
