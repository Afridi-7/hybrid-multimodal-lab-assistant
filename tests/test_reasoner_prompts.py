"""Tests for what the Stage-3 reasoners are told about cell counts and CBC input."""

import pytest


def _vision_summary(n_wbc):
    return {
        "cell_counts": {"WBC": n_wbc, "RBC": 17, "Platelet": 0},
        "wbc_differential": {"ig": 50.0, "monocyte": 50.0},
        "wbc_classified": n_wbc,
        "min_wbc_for_differential": 100,
        "cbc_report": {
            "sex": "female",
            "abnormal_count": 1,
            "findings": [{
                "analyte": "wbc", "value": 18.0, "unit": "x10^9/L",
                "reference_range": [4.0, 11.0], "label": "leukocytosis",
                "severity": "moderate", "direction": "high",
            }],
        },
    }


def test_agent_prompt_states_cell_count_and_cbc():
    pytest.importorskip("langgraph")
    pytest.importorskip("sentence_transformers")  # imported by the retriever
    from src.rag.agent import ClinicalReasoningAgent

    prompt = ClinicalReasoningAgent._build_user_prompt(
        _vision_summary(2), {"flagged_count": 0, "total_samples": 2}
    )
    assert "Number of white cells classified: 2" in prompt
    assert "NOT sufficient" in prompt
    assert "leukocytosis" in prompt


def test_linear_prompt_states_cell_count():
    pytest.importorskip("openai")
    from src.rag.llm_reasoner import ClinicalReasoner

    reasoner = ClinicalReasoner.__new__(ClinicalReasoner)
    text = reasoner._format_vision_summary(_vision_summary(2))
    assert "White cells classified: 2" in text
    assert "NOT sufficient" in text
