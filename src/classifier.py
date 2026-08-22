"""
The LLM feature under test: a customer support email classifier.

This is intentionally a thin wrapper. All the "intelligence" (prompt
wording, few-shot examples) lives in the versioned YAML files under
/prompts, not hardcoded here — that's what lets the eval pipeline swap
prompt versions in and out without touching this code.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from .schema import (
    ClassificationInput,
    ClassificationOutput,
    ClassificationResult,
    PromptConfig,
)

_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def _build_messages(prompt_config: PromptConfig, email_text: str) -> list[dict]:
    messages = [{"role": "system", "content": prompt_config.system_prompt}]
    for ex in prompt_config.few_shot_examples:
        messages.append({"role": "user", "content": ex.input})
        messages.append({"role": "assistant", "content": json.dumps(ex.output)})
    messages.append({"role": "user", "content": email_text})
    return messages


def classify_email(
    input_data: ClassificationInput,
    prompt_config: PromptConfig,
    model: str = "llama-3.3-70b-versatile",
) -> ClassificationResult:
    """Runs one email through the classifier feature and returns a fully
    typed, structured result — including the metadata (latency, tokens,
    prompt version) the eval engine needs for scoring and diffing."""

    messages = _build_messages(prompt_config, input_data.email_text)

    start = time.perf_counter()
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw = response.choices[0].message.content
    parsed = ClassificationOutput.model_validate_json(raw)

    usage = response.usage

    return ClassificationResult(
        input=input_data,
        output=parsed,
        prompt_version=prompt_config.version_id,
        model=model,
        latency_ms=latency_ms,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


if __name__ == "__main__":
    # Quick manual smoke test — run this once Phase 1 is wired up, before
    # you move on to building the golden dataset in Phase 2.
    cfg = PromptConfig.from_yaml("prompts/v1.yaml")
    test_input = ClassificationInput(
        email_text="I've been trying to reset my password for an hour and the reset email never arrives."
    )
    result = classify_email(test_input, cfg)
    print(result.model_dump_json(indent=2))
