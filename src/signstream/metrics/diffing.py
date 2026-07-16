"""Central hypothesis diffing: one Levenshtein alignment for the whole protocol.

Agents log full hypothesis snapshots, never edits (revisions must be
derivable, not self-reported). This module is the single place edits come
from: every stability metric consumes :func:`diff_hypotheses`, and the
quality metrics reuse the same alignment for their error counts — so the
protocol has exactly one, golden-tested edit-distance implementation.
(``jiwer`` is used in the test suite as an independent cross-check only.)

Math: for sequences :math:`a` (length :math:`m`) and :math:`b` (length
:math:`n`), the standard Levenshtein distance with unit costs

.. math::

    d_{i,j} = \\min\\bigl(d_{i-1,j-1} + [a_i \\neq b_j],\\;
                          d_{i-1,j} + 1,\\; d_{i,j-1} + 1\\bigr)

(Levenshtein, 1966). The backtrace of one optimal path yields the edit
script. Where several optimal paths exist, ties are broken deterministically
in the order **match/substitute, then delete, then insert**, so a given
``(prev, next)`` pair always produces the same script (goldens pin this).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, TypeAlias

EditOp: TypeAlias = Literal["substitute", "delete", "insert"]


class Edit(NamedTuple):
    """One token edit between consecutive hypotheses: ``(op, pos, old, new)``.

    Attributes:
        op: ``"substitute"``, ``"delete"``, or ``"insert"``.
        pos: Index into ``prev`` where the edit applies. For ``insert``, the
            token is inserted *before* ``prev[pos]`` (``pos == len(prev)``
            appends); several inserts may share a ``pos`` and are listed in
            the order they appear in ``next``.
        old: The replaced/deleted ``prev`` token; ``None`` for ``insert``.
        new: The inserted/substituting ``next`` token; ``None`` for
            ``delete``.
    """

    op: EditOp
    pos: int
    old: str | None
    new: str | None


def diff_hypotheses(prev: Sequence[str], next_: Sequence[str]) -> list[Edit]:
    """Minimal edit script turning ``prev`` into ``next_``.

    The single revision source for all stability metrics (ADR-5): the number
    of edits equals the Levenshtein distance, and applying the script to
    ``prev`` reconstructs ``next_`` exactly (property-tested).

    Args:
        prev: The earlier hypothesis (unit sequence).
        next_: The later hypothesis (unit sequence).

    Returns:
        Edits in left-to-right ``prev`` order (an empty list iff the
        hypotheses are equal). Deterministic under the documented
        tie-breaking rule.
    """
    m, n = len(prev), len(next_)
    # dp[i][j] = distance between prev[:i] and next_[:j].
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        row, above = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if prev[i - 1] == next_[j - 1] else 1
            row[j] = min(above[j - 1] + cost, above[j] + 1, row[j - 1] + 1)

    # Backtrace one optimal path; ties prefer match/substitute, then delete,
    # then insert. Walking back from (m, n) and reversing yields edits in
    # left-to-right order with same-pos inserts in next_'s order.
    edits: list[Edit] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if prev[i - 1] == next_[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    edits.append(Edit("substitute", i - 1, prev[i - 1], next_[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            edits.append(Edit("delete", i - 1, prev[i - 1], None))
            i -= 1
            continue
        edits.append(Edit("insert", i, None, next_[j - 1]))
        j -= 1
    edits.reverse()
    return edits
