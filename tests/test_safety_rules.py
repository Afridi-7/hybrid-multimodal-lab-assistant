"""Tests for the deterministic minimum-cell safety rule."""

from src.utils.safety_rules import INSUFFICIENT_CELL_COUNT_FLAG, apply_minimum_cell_rule


def test_too_few_cells_forces_review_and_flag():
    out = apply_minimum_cell_rule(
        {"safety_flags": [], "requires_expert_review": False}, n_wbc=2, min_wbc=100
    )
    assert out["requires_expert_review"] is True
    assert any(f.startswith(INSUFFICIENT_CELL_COUNT_FLAG) for f in out["safety_flags"])
    assert out["cell_count_check"] == {
        "wbc_classified": 2, "minimum_required": 100, "sufficient": False,
    }


def test_enough_cells_leaves_decision_unchanged():
    out = apply_minimum_cell_rule(
        {"safety_flags": [], "requires_expert_review": False}, n_wbc=150, min_wbc=100
    )
    assert out["requires_expert_review"] is False
    assert out["safety_flags"] == []
    assert out["cell_count_check"]["sufficient"] is True


def test_rule_never_lowers_an_existing_review_flag():
    out = apply_minimum_cell_rule(
        {"safety_flags": ["HIGH_UNCERTAINTY"], "requires_expert_review": True},
        n_wbc=500, min_wbc=100,
    )
    assert out["requires_expert_review"] is True
    assert out["safety_flags"] == ["HIGH_UNCERTAINTY"]


def test_rule_disabled_with_zero_minimum():
    reasoning = {"safety_flags": [], "requires_expert_review": False}
    out = apply_minimum_cell_rule(dict(reasoning), n_wbc=1, min_wbc=0)
    assert out == reasoning


def test_flag_is_not_duplicated_and_bad_shapes_are_normalised():
    out = apply_minimum_cell_rule({"safety_flags": "free text"}, n_wbc=0, min_wbc=100)
    out = apply_minimum_cell_rule(out, n_wbc=0, min_wbc=100)
    assert out["safety_flags"][0] == "free text"
    assert sum(f.startswith(INSUFFICIENT_CELL_COUNT_FLAG) for f in out["safety_flags"]) == 1
    assert out["requires_expert_review"] is True
