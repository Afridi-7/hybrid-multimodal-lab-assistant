"""Deterministic safety rules applied after Stage-3 reasoning.

These rules are enforced in code rather than left to the language model,
because the model can mis-judge them (see the two-cell case study in the
thesis). A rule may only *raise* ``requires_expert_review``; it never lowers it.
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_MIN_WBC_FOR_DIFFERENTIAL = 100

INSUFFICIENT_CELL_COUNT_FLAG = "INSUFFICIENT_CELL_COUNT"


def apply_minimum_cell_rule(
    reasoning: Dict[str, Any],
    n_wbc: int,
    min_wbc: int = DEFAULT_MIN_WBC_FOR_DIFFERENTIAL,
) -> Dict[str, Any]:
    """Force expert review when too few white cells were classified.

    A WBC differential expresses each cell type as a percentage of the cells
    counted. Routine manual differentials count at least 100 cells, so a
    percentage computed from a handful of cells is not meaningful.

    Args:
        reasoning: Stage-3 result dictionary (modified in place and returned).
        n_wbc: Number of white cells classified by Stage 2.
        min_wbc: Minimum number of cells for a reliable differential.
            ``0`` or a negative value disables the rule.

    Returns:
        The same dictionary, with ``cell_count_check`` added and, if the count
        is insufficient, an ``INSUFFICIENT_CELL_COUNT`` safety flag and
        ``requires_expert_review = True``.
    """
    n_wbc = int(n_wbc or 0)
    min_wbc = int(min_wbc or 0)
    if min_wbc <= 0:
        return reasoning

    sufficient = n_wbc >= min_wbc
    reasoning["cell_count_check"] = {
        "wbc_classified": n_wbc,
        "minimum_required": min_wbc,
        "sufficient": sufficient,
    }
    if sufficient:
        return reasoning

    flags = reasoning.get("safety_flags")
    if not isinstance(flags, list):
        flags = [] if flags in (None, "") else [str(flags)]
    if not any(str(f).startswith(INSUFFICIENT_CELL_COUNT_FLAG) for f in flags):
        flags.append(
            f"{INSUFFICIENT_CELL_COUNT_FLAG}: only {n_wbc} white cell(s) classified; "
            f"at least {min_wbc} are needed for a reliable differential, so "
            "percentages are not meaningful."
        )
    reasoning["safety_flags"] = flags
    reasoning["requires_expert_review"] = True
    return reasoning
