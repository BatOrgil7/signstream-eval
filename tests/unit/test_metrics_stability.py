"""Property and API tests of the stability metrics (upgr/upsr/ne/swe).

Golden fixtures run through the generic runner in test_metrics_quality.
Here: the spec's property tests (hypothesis, small budgets) — ne <=
swe/w_min, utterance-order permutation invariance, constant-delay
invariance, and zero-on-monotonic-runs — plus the SWE configuration
surface (weight sweep, custom class-A lists) and the usual API contracts.
"""

from __future__ import annotations

from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_metrics_quality import FIXTURE_DIR, build_pair, fixture_expectations, load_fixture

from signstream.metrics import (
    DEFAULT_CLASS_A_TOKENS,
    DEFAULT_SWE_WEIGHTS,
    REGISTRY,
    SeverityWeightedErasure,
)
from signstream.schema import (
    DatasetInfo,
    Emission,
    EmissionLog,
    MaskInfo,
    ModelInfo,
    Reference,
    RunMeta,
    UttStart,
    WallclockProtocol,
)

STABILITY_METRICS = ("stability/upgr", "stability/upsr", "stability/ne", "stability/swe")


def build_log(
    chains: dict[str, list[tuple[str, ...]]],
    unit: Literal["gloss", "char"] = "gloss",
    delay: int = 0,
) -> tuple[Reference, EmissionLog]:
    """A log whose utterances emit the given hypothesis chains (last = final).

    ``delay`` shifts every emission (and each utterance's length) by a
    constant number of frames — timing that stability metrics must ignore.
    """
    events: list[UttStart | Emission] = []
    for utt_id, chain in chains.items():
        n_frames = len(chain) + 10 + delay
        events.append(UttStart(utt_id=utt_id, n_frames=n_frames))
        for index, hyp in enumerate(chain[:-1]):
            events.append(
                Emission(
                    utt_id=utt_id, t_frame=index + delay, t_wall_ms=1.0, hyp=hyp, is_final=False
                )
            )
        events.append(
            Emission(
                utt_id=utt_id, t_frame=n_frames - 1, t_wall_ms=1.0, hyp=chain[-1], is_final=True
            )
        )
    run_meta = RunMeta(
        schema_version="1.0",
        run_id="synthetic/stability",
        created_utc="2026-08-14T09:31:07Z",
        git_sha="a1b2c3d",
        config_hash="9f2e77c1",
        experiment="golden",
        dataset=DatasetInfo(name="synthetic", split="test", fps=25.0, unit=unit),
        model=ModelInfo(
            name="transformer_ctc",
            params=1,
            mask=MaskInfo(mode="chunked", chunk_frames=1, lookahead_frames=0),
        ),
        seed=1,
        hardware={"cpu": "synthetic"},
        wallclock_protocol=WallclockProtocol(batch_size=1, warmup_utts=0, timer="perf_counter"),
    )
    ref = Reference(unit=unit, utterances={u: chain[-1] for u, chain in chains.items()})
    return ref, EmissionLog(run_meta=run_meta, events=tuple(events))


TOKEN = st.sampled_from(["A", "B", "NICHT", "KEIN", "NICHT-GUT", "X"])
HYP = st.lists(TOKEN, max_size=5).map(tuple)
CHAIN = st.lists(HYP, min_size=1, max_size=5)


def test_builder_produces_valid_logs() -> None:
    _, log = build_log({"u1": [("A",), ("A", "B")], "u2": [("X",)]}, delay=3)
    assert log.validate().ok


def test_stability_goldens_are_populated() -> None:
    """>= 3 goldens per stability metric (typical, edge, undefined)."""
    names = [name for _, name, _ in fixture_expectations()]
    for metric_name in STABILITY_METRICS:
        assert names.count(metric_name) >= 3, metric_name


def test_requires_declarations() -> None:
    """Recovered spec table: the Needs cell is an em-dash for all four."""
    for metric_name in STABILITY_METRICS:
        assert REGISTRY[metric_name].requires == frozenset(), metric_name


@settings(max_examples=60)
@given(chain=CHAIN)
def test_ne_bounded_by_swe_over_wmin(chain: list[tuple[str, ...]]) -> None:
    """The spec's named invariant: ne <= swe / w_min."""
    ref, log = build_log({"u": chain})
    ne = REGISTRY["stability/ne"].compute(ref, log).per_utt["u"]
    swe = REGISTRY["stability/swe"].compute(ref, log).per_utt["u"]
    w_min = min(DEFAULT_SWE_WEIGHTS.values())
    if ne is not None and swe is not None:
        assert ne <= swe / w_min + 1e-12


@settings(max_examples=30)
@given(
    chains=st.tuples(CHAIN, CHAIN, CHAIN),
    order=st.permutations([0, 1, 2]),
)
def test_utterance_order_never_changes_aggregates(
    chains: tuple[list[tuple[str, ...]], ...], order: list[int]
) -> None:
    base = {f"u{i}": chain for i, chain in enumerate(chains)}
    permuted = {f"u{i}": chains[i] for i in order}
    for metric_name in STABILITY_METRICS:
        ref_a, log_a = build_log(base)
        ref_b, log_b = build_log(permuted)
        agg_a = REGISTRY[metric_name].compute(ref_a, log_a).aggregate
        agg_b = REGISTRY[metric_name].compute(ref_b, log_b).aggregate
        assert agg_a == agg_b, metric_name


@settings(max_examples=30)
@given(chain=CHAIN, delay=st.integers(min_value=0, max_value=50))
def test_constant_delay_leaves_stability_invariant(
    chain: list[tuple[str, ...]], delay: int
) -> None:
    ref_a, log_a = build_log({"u": chain})
    ref_b, log_b = build_log({"u": chain}, delay=delay)
    for metric_name in STABILITY_METRICS:
        assert (
            REGISTRY[metric_name].compute(ref_a, log_a).per_utt["u"]
            == REGISTRY[metric_name].compute(ref_b, log_b).per_utt["u"]
        ), metric_name


@settings(max_examples=40)
@given(final=HYP, cuts=st.lists(st.integers(min_value=0, max_value=5), max_size=4))
def test_monotonic_runs_score_zero(final: tuple[str, ...], cuts: list[int]) -> None:
    """A revision-free (append-only) run yields 0 for every defined value."""
    prefixes = sorted(min(c, len(final)) for c in cuts)
    chain = [final[:c] for c in prefixes] + [final]
    ref, log = build_log({"u": chain})
    for metric_name in STABILITY_METRICS:
        result = REGISTRY[metric_name].compute(ref, log)
        value = result.per_utt["u"]
        assert value in (None, 0.0), (metric_name, value)
        short = metric_name.split("/")[1]
        if short in result.aggregate:
            assert result.aggregate[short] == 0.0, metric_name


def test_swe_weight_sweep() -> None:
    """w_A is configuration (§15 sweeps {3, 5, 10}); the class-A fixture's
    pooled SWE is 3·w_A/4 by hand."""
    fixture = load_fixture(FIXTURE_DIR / "stability_class_a.yaml")
    ref, log = build_pair(fixture)
    for w_a, expected in ((3.0, 2.25), (5.0, 3.75), (10.0, 7.5)):
        metric = SeverityWeightedErasure(weights={"A": w_a, "B": 2.0, "C": 1.0})
        result = metric.compute(ref, log)
        assert result.aggregate["swe"] == pytest.approx(expected), w_a


def test_swe_custom_class_a_tokens() -> None:
    """With an empty class-A list the same edits weigh as class B."""
    fixture = load_fixture(FIXTURE_DIR / "stability_class_a.yaml")
    ref, log = build_pair(fixture)
    metric = SeverityWeightedErasure(class_a_tokens=())
    # Hand: the three class-A edits become B (2 each): pooled (2+2+2)/4 = 1.5
    assert metric.compute(ref, log).aggregate["swe"] == pytest.approx(1.5)
    # And a list targeting a different token flips only that utterance:
    metric = SeverityWeightedErasure(class_a_tokens=("REGEN",))
    # u1's sub touches REGEN (new side) -> A:5; u2, u3 become B:2
    assert metric.compute(ref, log).aggregate["swe"] == pytest.approx((5 + 2 + 2) / 4)


def test_swe_rejects_incomplete_weights() -> None:
    with pytest.raises(ValueError, match="must define class"):
        SeverityWeightedErasure(weights={"A": 5.0, "B": 2.0})  # type: ignore[arg-type]


def test_default_class_a_list_is_the_phoenix_example() -> None:
    assert DEFAULT_CLASS_A_TOKENS == ("NICHT", "KEIN", "NICHT-*")
    assert DEFAULT_SWE_WEIGHTS == {"A": 5.0, "B": 2.0, "C": 1.0}


def test_unit_mismatch_is_a_pairing_error() -> None:
    _, log = build_pair(load_fixture(FIXTURE_DIR / "stability_stable.yaml"))
    char_ref = Reference(unit="char", utterances={"u1": ("A",)})
    for metric_name in STABILITY_METRICS:
        with pytest.raises(ValueError, match="does not match"):
            REGISTRY[metric_name].compute(char_ref, log)


def test_compute_is_deterministic_and_pure() -> None:
    ref, log = build_pair(load_fixture(FIXTURE_DIR / "stability_revisions.yaml"))
    for metric_name in STABILITY_METRICS:
        metric = REGISTRY[metric_name]
        assert metric.compute(ref, log) == metric.compute(ref, log)
    assert log.validate().ok
