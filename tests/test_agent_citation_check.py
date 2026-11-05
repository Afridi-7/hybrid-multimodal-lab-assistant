"""Tests for the agent's deterministic citation check (src/rag/agent.py::apply_citation_check)."""

import pytest


def _check():
    pytest.importorskip("langgraph")
    pytest.importorskip("sentence_transformers")  # imported by the retriever
    from src.rag.agent import apply_citation_check
    return apply_citation_check


def _refs(n):
    return [{"reference_id": i, "source": "essentials_haematology.pdf", "chunk_id": 100 + i} for i in range(1, n + 1)]


def test_citation_without_retrieval_forces_review():
    # The 5 Nov 2026 case-study run: no knowledge-base call, yet [Reference 1] and [Reference 2] cited.
    report = {"safety_flags": [], "requires_expert_review": False, "retrieved_references": []}
    _check()(report, '{"differential_diagnoses": ["Left shift [Reference 1]", "Monocytosis [Reference 2]"]}')
    assert report["requires_expert_review"] is True
    assert report["citation_check"]["unsupported"] == [1, 2]
    assert any(f.startswith("UNSUPPORTED_CITATION") for f in report["safety_flags"])


def test_citations_of_retrieved_passages_pass():
    report = {"safety_flags": [], "requires_expert_review": False, "retrieved_references": _refs(5)}
    _check()(report, "Left shift [Reference 2]; see also [Reference 5].")
    assert report["requires_expert_review"] is False
    assert report["citation_check"] == {"cited": [2, 5], "retrieved": [1, 2, 3, 4, 5], "unsupported": []}
    assert report["safety_flags"] == []


def test_check_never_lowers_the_review_flag():
    report = {"safety_flags": ["HIGH_UNCERTAINTY"], "requires_expert_review": True,
              "retrieved_references": _refs(5)}
    _check()(report, "No citations here.")
    assert report["requires_expert_review"] is True
    assert report["safety_flags"] == ["HIGH_UNCERTAINTY"]
