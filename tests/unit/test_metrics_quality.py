"""Golden and API tests of the quality metrics (quality/wer, quality/cer).

Golden fixtures under ``tests/fixtures/golden_logs/metrics/`` carry a
complete log, a reference, and hand-computed expected values; the tests
assert exact agreement (per-utterance values, ``None`` sentinels, aggregate
key sets, coverage). API tests pin the registry, purity, unit gating, and
the corpus aggregate against jiwer's independent implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jiwer
import pytest
import yaml

from signstream.metrics import REGISTRY, Metric, MetricResult, register
from signstream.schema import EmissionLog, Reference, RunMeta, event_from_json_obj

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden_logs" / "metrics"
METRIC_FIXTURES = sorted(p for p in FIXTURE_DIR.glob("*.yaml") if p.stem != "diff_cases")


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        fixture = yaml.safe_load(handle)
    assert isinstance(fixture, dict)
    return fixture


def build_pair(fixture: dict[str, Any]) -> tuple[Reference, EmissionLog]:
    """In-memory (reference, log) pair from a golden fixture."""
    log = EmissionLog(
        run_meta=RunMeta.from_json_obj(fixture["run_meta"]),
        events=tuple(event_from_json_obj(event) for event in fixture["events"]),
    )
    assert log.validate().ok, "golden fixtures must be schema-valid logs"
    return Reference.from_json_obj(fixture["reference"]), log


def fixture_expectations() -> list[tuple[str, str, dict[str, Any]]]:
    cases = []
    for path in METRIC_FIXTURES:
        fixture = load_fixture(path)
        for metric_name in fixture["expect"]["metrics"]:
            cases.append((path.stem, metric_name, fixture))
    return cases


def test_fixture_directory_is_populated() -> None:
    """Guard against silently running zero golden cases; the spec requires
    >= 3 goldens per metric (typical, edge, undefined)."""
    names = [name for _, name, _ in fixture_expectations()]
    assert names.count("quality/wer") >= 3
    assert names.count("quality/cer") >= 3


@pytest.mark.parametrize(
    ("stem", "metric_name", "fixture"),
    fixture_expectations(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_metric_golden(stem: str, metric_name: str, fixture: dict[str, Any]) -> None:
    ref, log = build_pair(fixture)
    expected = fixture["expect"]["metrics"][metric_name]

    result = REGISTRY[metric_name].compute(ref, log)

    assert result.name == metric_name
    assert set(result.per_utt) == set(expected["per_utt"])
    for utt_id, value in expected["per_utt"].items():
        actual = result.per_utt[utt_id]
        if value is None:
            assert actual is None, f"{utt_id}: expected undefined, got {actual}"
        else:
            assert actual == pytest.approx(value, rel=1e-12), utt_id
    assert set(result.aggregate) == set(expected["aggregate"])
    for key, value in expected["aggregate"].items():
        assert result.aggregate[key] == pytest.approx(value, rel=1e-12), key


def test_registry_contents() -> None:
    assert {"quality/wer", "quality/cer"} <= set(REGISTRY)
    metric: Metric = REGISTRY["quality/wer"]  # structural protocol check
    assert metric.requires == frozenset()


def test_register_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register
        class DuplicateWer:
            name: str = "quality/wer"
            requires: frozenset[str] = frozenset()

            def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
                raise NotImplementedError


def test_unit_mismatch_is_a_pairing_error() -> None:
    fixture = load_fixture(FIXTURE_DIR / "wer_typical.yaml")
    _, log = build_pair(fixture)  # gloss-unit log
    char_ref = Reference(unit="char", utterances={"u1": ("A",)})
    with pytest.raises(ValueError, match="does not match"):
        REGISTRY["quality/wer"].compute(char_ref, log)


def test_compute_is_deterministic_and_pure() -> None:
    ref, log = build_pair(load_fixture(FIXTURE_DIR / "wer_typical.yaml"))
    metric = REGISTRY["quality/wer"]
    first = metric.compute(ref, log)
    second = metric.compute(ref, log)
    assert first == second
    # Inputs are untouched: frozen dataclasses plus identical revalidation.
    assert log.validate().ok


def test_corpus_wer_matches_jiwer() -> None:
    """Corpus aggregate cross-check against jiwer (the spec's stated role
    for jiwer). wer_typical has no empty references, which jiwer rejects."""
    fixture = load_fixture(FIXTURE_DIR / "wer_typical.yaml")
    ref, log = build_pair(fixture)
    result = REGISTRY["quality/wer"].compute(ref, log)

    finals = {e.utt_id: e.hyp for e in log.events if getattr(e, "is_final", False)}
    references = [" ".join(units) for units in ref.utterances.values()]
    hypotheses = [" ".join(finals[utt_id]) for utt_id in ref.utterances]
    assert result.aggregate["wer"] == pytest.approx(jiwer.wer(references, hypotheses), rel=1e-12)
