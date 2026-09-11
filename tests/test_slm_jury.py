import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from judges.rubric import (
    SCORING_RUBRIC,
    INITIAL_JUDGMENT_SCHEMA,
    DEBATE_JUDGMENT_SCHEMA,
    InitialJudgment,
    DebateJudgment,
    FEW_SHOT_CALIBRATION_EXAMPLES,
)
from judges.profiles import (
    JURY_PERSONAS,
    build_initial_prompt,
    build_debate_prompt,
)
from judges.slm_jury import (
    aggregate_jury_scores,
    run_agent_initial_judgment,
    run_agent_debate,
    slm_jury_judge,
)
from judges.gpt4o import gpt4o_judge
from src.schema import (
    Category,
    CaseScore,
    Difficulty,
    EvalRun,
    ScoreRegressionCase,
)
from src.eval_engine import compare_runs


def test_rubric_schema_and_ordering():
    """Verify that rubric exists and schemas enforce reasoning before score."""
    assert "Deterministic 1-4 Scoring Rubric" in SCORING_RUBRIC
    assert "category_match" in SCORING_RUBRIC

    # Initial judgment keys order
    initial_props = list(INITIAL_JUDGMENT_SCHEMA["properties"].keys())
    assert initial_props == ["category_match", "reasoning", "score"]

    # Debate judgment keys order
    debate_props = list(DEBATE_JUDGMENT_SCHEMA["properties"].keys())
    assert debate_props == ["comparison_notes", "revised", "final_reasoning", "final_score"]

    # Few-shot examples
    assert len(FEW_SHOT_CALIBRATION_EXAMPLES) >= 4
    for ex in FEW_SHOT_CALIBRATION_EXAMPLES:
        assert 1 <= ex["score"] <= 4


def test_personas_and_prompts():
    """Verify all 3 personas exist and build prompts correctly."""
    assert len(JURY_PERSONAS) == 3
    persona_names = {p.name for p in JURY_PERSONAS}
    assert persona_names == {
        "Category Precisionist",
        "Summary Fidelity Checker",
        "Edge Case Skeptic",
    }

    initial_prompt = build_initial_prompt(
        email="My account is locked.",
        golden_category="account",
        golden_summary="Customer cannot access account.",
        model_category="account",
        model_summary="Customer cannot access account.",
    )
    assert "Deterministic 1-4 Scoring Rubric" in initial_prompt
    assert "My account is locked." in initial_prompt

    debate_prompt = build_debate_prompt(
        email="My account is locked.",
        golden_category="account",
        golden_summary="Customer cannot access account.",
        model_category="account",
        model_summary="Customer cannot access account.",
        own_initial={"category_match": True, "reasoning": "Looks good", "score": 4},
        anonymized_peers=[
            {"label": "Judge 1", "category_match": True, "reasoning": "Peer reasoning 1", "score": 3},
            {"label": "Judge 2", "category_match": True, "reasoning": "Peer reasoning 2", "score": 4},
        ],
    )
    assert "Judge 1" in debate_prompt
    assert "Judge 2" in debate_prompt
    assert "not blind deference" in debate_prompt.lower() or "reference only" in debate_prompt.lower()
    # Confirm persona names are NOT hardcoded in peer labels
    assert "Category Precisionist Evaluation" not in debate_prompt
    assert "Edge Case Skeptic Evaluation" not in debate_prompt


def test_aggregation_mode_and_median_fallback():
    """Verify aggregation uses mode and falls back to median (no averaging)."""
    # Unanimous
    assert aggregate_jury_scores([4, 4, 4]) == 4
    assert aggregate_jury_scores([1, 1, 1]) == 1

    # Clear majority
    assert aggregate_jury_scores([3, 3, 2]) == 3
    assert aggregate_jury_scores([1, 4, 4]) == 4
    assert aggregate_jury_scores([2, 1, 2]) == 2

    # Tie / No unique mode -> fallback to median
    # [2, 3, 4] -> median is 3 (average is 3.0)
    assert aggregate_jury_scores([2, 3, 4]) == 3
    # [1, 2, 4] -> median is 2 (average would be 2.33)
    assert aggregate_jury_scores([1, 2, 4]) == 2
    # [1, 3, 4] -> median is 3 (average would be 2.66)
    assert aggregate_jury_scores([1, 3, 4]) == 3


def test_initial_judgment_category_hard_gate():
    """If category_match is False, score must be 1 even if LLM hallucinated score 3."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {
            "content": '{"category_match": false, "reasoning": "Wrong category predicted", "score": 3}'
        }
    }

    res = run_agent_initial_judgment(
        persona=JURY_PERSONAS[0],
        email="Need refund",
        golden_category="billing",
        golden_summary="Refund request",
        model_category="technical",
        model_summary="Refund request",
        client=mock_client,
    )
    assert res["category_match"] is False
    assert res["score"] == 1  # Hard gate enforced


def test_slm_jury_judge_end_to_end():
    """Test full slm_jury_judge call with mocked responses."""
    mock_client = MagicMock()
    
    # Return different responses for initial and debate rounds
    def fake_chat(model, messages, format, options):
        # Initial round format schema check
        if "category_match" in format.get("properties", {}):
            return {
                "message": {
                    "content": '{"category_match": true, "reasoning": "Valid category and summary match.", "score": 4}'
                }
            }
        else:
            return {
                "message": {
                    "content": '{"comparison_notes": "Agreed with peers.", "revised": false, "final_reasoning": "Grounded in email.", "final_score": 4}'
                }
            }

    mock_client.chat.side_effect = fake_chat

    res = slm_jury_judge(
        email="I need to reset my password.",
        golden_category="account",
        golden_summary="Customer needs password reset.",
        model_output={"category": "account", "summary": "Customer needs password reset."},
        client=mock_client,
    )

    assert res["score"] == 4
    assert "trail" in res
    assert len(res["trail"]["initial"]) == 3
    assert len(res["trail"]["debate"]) == 3
    for init in res["trail"]["initial"]:
        assert init["category_match"] is True
        assert init["score"] == 4


def test_gpt4o_judge_mocked():
    """Test gpt4o_judge function structure and output."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"category_match": true, "reasoning": "Exact match", "score": 4}'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    res = gpt4o_judge(
        email="Duplicate billing charge",
        golden_category="billing",
        golden_summary="Duplicate charge refund requested",
        model_output={"category": "billing", "summary": "Duplicate charge refund requested"},
        client=mock_client,
    )
    assert res["score"] == 4
    assert res["trail"]["category_match"] is True


def test_regression_diff_logic():
    """Test that score degradation (delta < 0) is flagged as regression even if passed."""
    now = datetime.now(timezone.utc)

    # Baseline run with score 4
    prev_score = CaseScore(
        case_id="case_001",
        difficulty=Difficulty.easy,
        input="Email text",
        actual_category=Category.billing,
        expected_category=Category.billing,
        category_match=True,
        actual_summary="Customer wants refund.",
        expected_summary="Customer wants refund.",
        summary_relevance_score=4,
        latency_ms=100.0,
        input_tokens=50,
        output_tokens=20,
        passed=True,
    )
    prev_run = EvalRun(
        run_id="run_v1_001",
        prompt_version="v1",
        dataset_version="v1",
        model="llama-3.3-70b",
        timestamp=now,
        case_scores=[prev_score],
        pass_rate=1.0,
        category_accuracy={"billing": 1.0},
        avg_latency_ms=100.0,
        total_input_tokens=50,
        total_output_tokens=20,
    )

    # Current run with score 3 (still passed, but degraded by 1)
    curr_score = CaseScore(
        case_id="case_001",
        difficulty=Difficulty.easy,
        input="Email text",
        actual_category=Category.billing,
        expected_category=Category.billing,
        category_match=True,
        actual_summary="Customer mentioned refund.",
        expected_summary="Customer wants refund.",
        summary_relevance_score=3,
        latency_ms=105.0,
        input_tokens=50,
        output_tokens=20,
        passed=True,
    )
    curr_run = EvalRun(
        run_id="run_v2_001",
        prompt_version="v2",
        dataset_version="v1",
        model="llama-3.3-70b",
        timestamp=now,
        case_scores=[curr_score],
        pass_rate=1.0,
        category_accuracy={"billing": 1.0},
        avg_latency_ms=105.0,
        total_input_tokens=50,
        total_output_tokens=20,
    )

    comparison = compare_runs(curr_run, prev_run)

    # Binary passed status did not flip
    assert len(comparison.regressions) == 0

    # Score regression should be captured (4 -> 3, delta = -1)
    assert len(comparison.score_regressions) == 1
    assert comparison.score_regressions[0].case_id == "case_001"
    assert comparison.score_regressions[0].previous_score == 4
    assert comparison.score_regressions[0].current_score == 3
    assert comparison.score_regressions[0].score_delta == -1
    assert comparison.avg_score_delta == -1.0


def test_validate_jury_agreement_flow(tmp_path):
    """Test validation script functions end-to-end with temporary outputs."""
    from scripts.validate_jury_agreement import (
        run_agreement_validation,
        run_reproducibility_check,
        print_summary_table,
        save_results,
    )

    test_cases = [
        {
            "id": "case_test_1",
            "input": "I need help with login",
            "expected_category": "account",
            "expected_summary": "Customer needs login help",
            "difficulty": "easy",
        },
        {
            "id": "case_test_2",
            "input": "Double charged on my card",
            "expected_category": "billing",
            "expected_summary": "Customer was double charged",
            "difficulty": "easy",
        },
    ]

    with patch("scripts.validate_jury_agreement.gpt4o_judge") as mock_gpt, \
         patch("scripts.validate_jury_agreement.slm_jury_judge") as mock_jury:
        
        mock_gpt.side_effect = [
            {"score": 4, "trail": {}},
            {"score": 2, "trail": {}},
        ]
        mock_jury.side_effect = [
            {"score": 4, "trail": {}},  # agreement 4==4
            {"score": 2, "trail": {}},  # agreement 2==2 (wrong case)
            # For reproducibility check: 2 cases x 3 repeats = 6 calls
            {"score": 4, "trail": {}},
            {"score": 4, "trail": {}},
            {"score": 4, "trail": {}},
            {"score": 2, "trail": {}},
            {"score": 2, "trail": {}},
            {"score": 2, "trail": {}},
        ]

        agreement_data = run_agreement_validation(test_cases)
        assert agreement_data["total_cases"] == 2
        assert agreement_data["exact_match_pct"] == 100.0
        assert agreement_data["within_one_pct"] == 100.0
        assert agreement_data["wrong_cases_count"] == 1
        assert agreement_data["wrong_cases_within_one_pct"] == 100.0

        repro_data = run_reproducibility_check(test_cases, sample_size=2, repeats=3)
        assert repro_data["sample_size"] == 2
        assert repro_data["reproducibility_pct"] == 100.0

        # Print summary table test
        print_summary_table(agreement_data, repro_data)

        # Save results test
        out_file = tmp_path / "jury_results"
        save_results(out_file, agreement_data, repro_data)
        assert (tmp_path / "jury_results.json").exists()
        assert (tmp_path / "jury_results.csv").exists()

