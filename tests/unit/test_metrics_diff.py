"""Tests of diff_hypotheses: golden edit scripts, reconstruction, jiwer cross-check.

The differ is the protocol's single revision source, so it is pinned three
ways: exact golden scripts (including tie-break cases), the diff-then-apply
reconstruction property on random sequences, and an edit-count cross-check
against jiwer's independent Levenshtein implementation (the spec's stated
role for jiwer: cross-check only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jiwer
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from signstream.metrics import Edit, diff_hypotheses

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "golden_logs" / "metrics" / "diff_cases.yaml"
)

with FIXTURE_PATH.open(encoding="utf-8") as _handle:
    DIFF_CASES: list[dict[str, Any]] = yaml.safe_load(_handle)["cases"]


def apply_edits(prev: tuple[str, ...], edits: list[Edit]) -> tuple[str, ...]:
    """Apply an edit script (pos in prev coordinates) — test-side inverse."""
    out = list(prev)
    offset = 0
    for op, pos, old, new in edits:
        index = pos + offset
        if op == "substitute":
            assert out[index] == old
            assert new is not None
            out[index] = new
        elif op == "delete":
            assert out[index] == old
            del out[index]
            offset -= 1
        else:  # insert
            assert new is not None
            out.insert(index, new)
            offset += 1
    return tuple(out)


@pytest.mark.parametrize("case", DIFF_CASES, ids=lambda c: c["name"])
def test_diff_golden_scripts(case: dict[str, Any]) -> None:
    prev, next_ = tuple(case["prev"]), tuple(case["next"])
    edits = diff_hypotheses(prev, next_)
    assert [tuple(e) for e in edits] == [tuple(e) for e in case["edits"]]
    # Every golden script must also reconstruct next from prev.
    assert apply_edits(prev, edits) == next_


TOKENS = st.lists(st.sampled_from(["A", "B", "C", "D"]), max_size=8).map(tuple)


@given(prev=TOKENS, next_=TOKENS)
def test_diff_then_apply_reconstructs(prev: tuple[str, ...], next_: tuple[str, ...]) -> None:
    assert apply_edits(prev, diff_hypotheses(prev, next_)) == next_


@given(seq=TOKENS)
def test_equal_sequences_diff_to_nothing(seq: tuple[str, ...]) -> None:
    assert diff_hypotheses(seq, seq) == []


@given(prev=TOKENS, next_=TOKENS)
def test_edit_count_is_minimal_vs_jiwer(prev: tuple[str, ...], next_: tuple[str, ...]) -> None:
    """Total edit count equals jiwer's independent S+D+I (cross-check only).

    Restricted to non-empty sequences: jiwer rejects empty references, and
    the empty cases are covered by the goldens and the reconstruction
    property.
    """
    if not prev or not next_:
        return
    ours = len(diff_hypotheses(prev, next_))
    measures = jiwer.process_words(" ".join(prev), " ".join(next_))
    theirs = measures.substitutions + measures.deletions + measures.insertions
    assert ours == theirs
