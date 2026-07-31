"""API tests of the latency metrics (fel/gel/al) beyond the golden fixtures.

Covers the requires declarations (the scorer's skip-with-warning contract),
behavior when a run provides no alignment at all, the wait-k invariant of
the imported AL metric, unit-mismatch pairing errors, and purity.
"""

from __future__ import annotations

import pytest
from test_metrics_quality import FIXTURE_DIR, build_pair, fixture_expectations, load_fixture

from signstream.metrics import REGISTRY
from signstream.schema import (
    DatasetInfo,
    Emission,
    EmissionLog,
    MaskInfo,
    ModelInfo,
    Reference,
    RunMeta,
    Span,
    UttStart,
    WallclockProtocol,
)


def test_latency_goldens_are_populated() -> None:
    """>= 3 goldens per latency metric (typical, edge, undefined)."""
    names = [name for _, name, _ in fixture_expectations()]
    assert names.count("latency/fel_ms") >= 3
    assert names.count("latency/gel_ms") >= 3
    assert names.count("latency/al") >= 3


def test_requires_declarations() -> None:
    """The scorer skips alignment-requiring metrics on runs without spans."""
    assert REGISTRY["latency/fel_ms"].requires == frozenset()
    assert REGISTRY["latency/gel_ms"].requires == frozenset({"alignment"})
    assert REGISTRY["latency/al"].requires == frozenset({"alignment"})


@pytest.mark.parametrize("metric_name", ["latency/gel_ms", "latency/al"])
def test_alignment_metrics_are_total_without_alignment(metric_name: str) -> None:
    """compute() stays total if called anyway: everything undefined."""
    fixture = load_fixture(FIXTURE_DIR / "gel_typical.yaml")
    ref_with, log = build_pair(fixture)
    ref_without = Reference(unit=ref_with.unit, utterances=ref_with.utterances, alignment=None)

    result = REGISTRY[metric_name].compute(ref_without, log)

    assert set(result.per_utt) == set(ref_without.utterances)
    assert all(value is None for value in result.per_utt.values())
    assert result.aggregate == {"coverage": 0.0}


def _wait_k_log(n_units: int, k: int) -> tuple[Reference, EmissionLog]:
    """A revision-free wait-k agent over unit-aligned frames.

    Source: one frame per reference unit (unit i ends at frame i). The
    agent emits its i-th unit right after observing frame i + k - 1, so by
    Ma et al.'s definition AL must equal k exactly.
    """
    n_frames = n_units + k  # enough frames for the last unit's emission time
    units = tuple(f"G{i}" for i in range(n_units))
    spans = tuple(Span(unit=u, f_start=i, f_end=i) for i, u in enumerate(units))
    events: list[UttStart | Emission] = [UttStart(utt_id="u", n_frames=n_frames)]
    for i in range(n_units):
        events.append(
            Emission(
                utt_id="u",
                t_frame=i + k - 1,
                t_wall_ms=1.0,
                hyp=units[: i + 1],
                is_final=False,
            )
        )
    events.append(
        Emission(utt_id="u", t_frame=n_frames - 1, t_wall_ms=1.0, hyp=units, is_final=True)
    )
    run_meta = RunMeta(
        schema_version="1.0",
        run_id="synthetic/wait-k",
        created_utc="2026-08-14T09:31:07Z",
        git_sha="a1b2c3d",
        config_hash="9f2e77c1",
        experiment="golden",
        dataset=DatasetInfo(name="phoenix14t", split="test", fps=25.0, unit="gloss"),
        model=ModelInfo(
            name="transformer_ctc",
            params=1,
            mask=MaskInfo(mode="chunked", chunk_frames=1, lookahead_frames=k),
        ),
        seed=1,
        hardware={"cpu": "synthetic"},
        wallclock_protocol=WallclockProtocol(batch_size=1, warmup_utts=0, timer="perf_counter"),
    )
    ref = Reference(unit="gloss", utterances={"u": units}, alignment={"u": spans})
    return ref, EmissionLog(run_meta=run_meta, events=tuple(events))


@pytest.mark.parametrize(("n_units", "k"), [(2, 1), (4, 1), (4, 2), (5, 3), (6, 4)])
def test_al_wait_k_invariant(n_units: int, k: int) -> None:
    """For a revision-free wait-k policy on unit-aligned frames, AL == k
    (Ma et al., 2019) — the imported formula's defining property."""
    ref, log = _wait_k_log(n_units, k)
    assert log.validate().ok
    result = REGISTRY["latency/al"].compute(ref, log)
    assert result.per_utt["u"] == pytest.approx(k)


def test_unit_mismatch_is_a_pairing_error() -> None:
    _, log = build_pair(load_fixture(FIXTURE_DIR / "fel_typical.yaml"))
    char_ref = Reference(unit="char", utterances={"u1": ("A",)})
    for name in ("latency/fel_ms", "latency/gel_ms", "latency/al"):
        with pytest.raises(ValueError, match="does not match"):
            REGISTRY[name].compute(char_ref, log)


def test_compute_is_deterministic_and_pure() -> None:
    ref, log = build_pair(load_fixture(FIXTURE_DIR / "gel_typical.yaml"))
    for name in ("latency/fel_ms", "latency/gel_ms", "latency/al"):
        metric = REGISTRY[name]
        assert metric.compute(ref, log) == metric.compute(ref, log)
    assert log.validate().ok
