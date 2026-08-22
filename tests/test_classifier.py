import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.schema import (
    PromptConfig,
    ClassificationInput,
    ClassificationOutput,
    ClassificationResult,
    Category,
)
from src.classifier import _build_messages, classify_email


def test_load_prompt_config():
    """Test loading prompt configuration from YAML."""
    yaml_path = Path(__file__).parent.parent / "prompts" / "v1.yaml"
    cfg = PromptConfig.from_yaml(yaml_path)
    assert cfg.version_id == "v1"
    assert "triage assistant" in cfg.system_prompt
    assert len(cfg.few_shot_examples) >= 3


def test_build_messages():
    """Test prompt message formatting with system prompt and few-shot examples."""
    yaml_path = Path(__file__).parent.parent / "prompts" / "v1.yaml"
    cfg = PromptConfig.from_yaml(yaml_path)
    messages = _build_messages(cfg, "Test email content")
    
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Test email content"


def test_classification_schema():
    """Test Pydantic output model validation."""
    valid = ClassificationOutput(category=Category.billing, summary="Billing issue")
    assert valid.category == "billing"
    assert valid.summary == "Billing issue"


@patch("src.classifier._client.chat.completions.create")
def test_classify_email_mocked(mock_create):
    """Test classification flow with mocked OpenAI API client."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"category": "account", "summary": "Password reset failure."}'))
    ]
    mock_response.usage = MagicMock(prompt_tokens=150, completion_tokens=25)
    mock_create.return_value = mock_response

    yaml_path = Path(__file__).parent.parent / "prompts" / "v1.yaml"
    cfg = PromptConfig.from_yaml(yaml_path)
    test_input = ClassificationInput(email_text="Cannot reset my password")

    result = classify_email(test_input, cfg)

    assert result.output.category == Category.account
    assert result.output.summary == "Password reset failure."
    assert result.prompt_version == "v1"
    assert result.input_tokens == 150
    assert result.output_tokens == 25
    assert result.latency_ms > 0
