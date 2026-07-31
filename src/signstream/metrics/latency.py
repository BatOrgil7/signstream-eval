"""Latency metrics: ``latency/fel_ms``, ``latency/gel_ms``, ``latency/al``.

**Clocks.** All three are *algorithmic* latency metrics, defined on the
logical frame clock — hardware-independent, in frames at the corpus fps
(PHOENIX: 25 fps, 1 frame = 40 ms). Millisecond values are frame quantities
converted via ``1000 / fps``. Wall-clock (computational) latency is a
different family entirely (``compute/step_ms``, ``compute/rtf``) and is
never mixed in here.

**Timing conventions.** An emission at ``t_frame`` means "hypothesis
available after observing frame ``t_frame``", i.e. at video time
``(t_frame + 1) / fps`` — so absolute times from the utterance start carry
the ``+ 1`` while differences of frame indices do not. Content latency uses
*stable* emission times: the earliest emission from which a final-hypothesis
position holds its final value through to the final emission ("content
first stably emitted").

**Aggregation.** Latency distributions are non-normal, so aggregates are
percentiles (p50/p90, linear interpolation on the inclusive scale) plus
``coverage``; percentile keys are omitted when nothing is defined.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence

from signstream.metrics.base import (
    MetricResult,
    check_units_match,
    emissions_by_utt,
    final_hypotheses,
    register,
)
from signstream.metrics.diffing import diff_hypotheses
from signstream.schema import Emission, EmissionLog, Reference, Span


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile on the inclusive scale.

    The ``numpy`` "linear" / ``statistics`` "inclusive" convention:
    rank ``h = (n - 1) * fraction``, interpolating between the neighbouring
    order statistics. A single value is every percentile of itself.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def _stable_times(emissions: Sequence[Emission], final_hyp: tuple[str, ...]) -> list[int]:
    """Per final-hypothesis position: ``t_frame`` of its stable emission.

    Position ``p`` is *stable* from emission ``k`` when every emission from
    ``k`` through the final one contains ``final_hyp[p]`` at position ``p``.
    The final emission itself always qualifies, so a stable time exists for
    every position.
    """
    stable: list[int] = []
    for position, token in enumerate(final_hyp):
        stable_index = len(emissions) - 1
        for index in range(len(emissions) - 1, -1, -1):
            hyp = emissions[index].hyp
            if position < len(hyp) and hyp[position] == token:
                stable_index = index
            else:
                break
        stable.append(emissions[stable_index].t_frame)
    return stable


def _matched_positions(
    reference: tuple[str, ...], hypothesis: tuple[str, ...]
) -> list[tuple[int, int]]:
    """Matched (reference index, hypothesis index) pairs of the central alignment.

    Derived by replaying :func:`diff_hypotheses`' edit script — the
    positions it never touches are the matches — so latency shares the one
    Levenshtein alignment with every other metric family.
    """
    matches: list[tuple[int, int]] = []
    ref_index = 0
    hyp_index = 0
    for edit in diff_hypotheses(reference, hypothesis):
        while ref_index < edit.pos:
            matches.append((ref_index, hyp_index))
            ref_index += 1
            hyp_index += 1
        if edit.op == "substitute":
            ref_index += 1
            hyp_index += 1
        elif edit.op == "delete":
            ref_index += 1
        else:  # insert
            hyp_index += 1
    while ref_index < len(reference):
        matches.append((ref_index, hyp_index))
        ref_index += 1
        hyp_index += 1
    return matches


def _percentile_aggregate(defined: Sequence[float], coverage: float) -> dict[str, float]:
    """The p50/p90 + coverage aggregate block shared by fel and gel."""
    aggregate: dict[str, float] = {"coverage": coverage}
    if defined:
        aggregate["p50"] = _percentile(defined, 0.5)
        aggregate["p90"] = _percentile(defined, 0.9)
    return aggregate


@register
class FirstEmissionLatency:
    r"""``latency/fel_ms`` — first-emission latency on the logical frame clock.

    Time of the first non-empty emission minus the utterance start. With the
    protocol's emission-timing convention (an emission at :math:`t` is
    available at video time :math:`(t+1)/\mathrm{fps}`):

    .. math::

        \mathrm{FEL}(u) = \frac{t_{\mathrm{first}}(u) + 1}{\mathrm{fps}}
        \cdot 1000 \;\mathrm{ms},

    where :math:`t_{\mathrm{first}}(u)` is the ``t_frame`` of the
    utterance's earliest emission with a non-empty hypothesis. Undefined
    (``None``) when the utterance never emits content (no non-empty
    emission) or is absent from the log. Aggregates: p50/p90 over defined
    utterances plus coverage.

    Provenance: the initial-response delay reported for streaming
    recognizers and simultaneous translation (cf. first-token delay in Ma
    et al., 2019); measured here on the hardware-free logical clock.
    Golden fixtures: ``tests/fixtures/golden_logs/metrics/fel_*.yaml``.
    """

    name: str = "latency/fel_ms"
    requires: frozenset[str] = frozenset()

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""
        check_units_match(ref, log)
        fps = log.run_meta.dataset.fps
        emissions = emissions_by_utt(log)

        per_utt: dict[str, float | None] = {}
        defined: list[float] = []
        for utt_id in ref.utterances:
            first = next((e for e in emissions.get(utt_id, []) if e.hyp), None)
            if first is None:
                per_utt[utt_id] = None
                continue
            value = (first.t_frame + 1) * 1000.0 / fps
            per_utt[utt_id] = value
            defined.append(value)

        coverage = len(defined) / len(ref.utterances) if ref.utterances else 0.0
        return MetricResult(
            name=self.name,
            per_utt=per_utt,
            aggregate=_percentile_aggregate(defined, coverage),
        )


@register
class GlossEmissionLag:
    r"""``latency/gel_ms`` — per-unit emission lag against aligned ends.

    For each reference unit :math:`g` with aligned end frame :math:`\tau(g)`:

    .. math::

        \mathrm{GEL}(g) = \frac{\mathrm{stab}(g) - \tau(g)}{\mathrm{fps}}
        \cdot 1000 \;\mathrm{ms},

    where :math:`\mathrm{stab}(g)` is the earliest ``t_frame`` from which
    :math:`g` is present at its final aligned position in every hypothesis
    through to the final one. Negative values mean the unit was committed
    before its aligned end. Frame-index differences need no ``+1`` offset.

    Defined only for units that are *correct* in the final hypothesis
    (matched, not substituted/deleted, under the central Levenshtein
    alignment) and that have an aligned span. Per-utterance value: mean of
    the utterance's defined unit lags (``None`` when none). Aggregates:
    p50/p90 over the pooled per-unit lags of the whole set, plus coverage =
    defined units / all reference units. Requires alignment; without it
    (``Reference.alignment`` is ``None``) everything is undefined — the
    scorer skips this metric with a warning instead of calling it.

    Provenance: protocol-native; adapts per-token emission latency from
    streaming ASR to CSLR with CTC-forced-aligned reference ends (all
    latency claims are stated relative to CTC-aligned ends by protocol
    design). Golden fixtures:
    ``tests/fixtures/golden_logs/metrics/gel_*.yaml``.
    """

    name: str = "latency/gel_ms"
    requires: frozenset[str] = frozenset({"alignment"})

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""
        check_units_match(ref, log)
        fps = log.run_meta.dataset.fps
        emissions = emissions_by_utt(log)
        finals = final_hypotheses(log)

        per_utt: dict[str, float | None] = {}
        pooled: list[float] = []
        total_units = 0
        for utt_id, ref_units in ref.utterances.items():
            total_units += len(ref_units)
            spans = _spans_for(ref, utt_id)
            final_hyp = finals.get(utt_id)
            if spans is None or final_hyp is None:
                per_utt[utt_id] = None
                continue
            stable = _stable_times(emissions[utt_id], final_hyp)
            matched = dict(_matched_positions(ref_units, final_hyp))
            lags: list[float] = []
            for ref_index in range(len(ref_units)):
                hyp_index = matched.get(ref_index)
                span = spans[ref_index] if ref_index < len(spans) else None
                if hyp_index is None or span is None:
                    continue
                lags.append((stable[hyp_index] - span.f_end) * 1000.0 / fps)
            per_utt[utt_id] = sum(lags) / len(lags) if lags else None
            pooled.extend(lags)

        coverage = len(pooled) / total_units if total_units else 0.0
        return MetricResult(
            name=self.name,
            per_utt=per_utt,
            aggregate=_percentile_aggregate(pooled, coverage),
        )


@register
class AverageLagging:
    r"""``latency/al`` — Average Lagging (Ma et al., 2019), in dataset units.

    **Imported metric**, reimplemented from the published formula and
    reported for cross-paper comparability; it is not a contribution of
    this protocol. For final hypothesis :math:`y` and a source of
    :math:`|x|` reference units:

    .. math::

        \mathrm{AL} = \frac{1}{\tau} \sum_{i=1}^{\tau}
        \Bigl[ g(i) - \frac{i-1}{\gamma} \Bigr],
        \qquad \gamma = \frac{|y|}{|x|},
        \qquad \tau = \min\{\, i : g(i) = |x| \,\}

    (:math:`\tau = |y|` when no prefix observes the full source). Protocol
    instantiation (the source axis is measured in *dataset units*, hence
    the alignment requirement and the unit-denominated result): :math:`g(i)`
    is the number of reference units fully observed — aligned end
    :math:`\le` the stable ``t_frame`` of final-hypothesis position
    :math:`i` — when unit :math:`i` is stably emitted. For a
    revision-free incremental system this reduces exactly to Ma et al.'s
    definition. AL is content-agnostic on the target side (like the
    original): correctness is the quality family's job.

    Undefined (``None``) when the final hypothesis is empty, the reference
    is empty, the utterance is absent from the log, or the utterance lacks
    a complete set of aligned spans. Aggregates: mean (the number papers
    report) plus p50/p90 over defined utterances and coverage.

    Provenance: Ma et al., 2019 — "STACL: Simultaneous Translation with
    Implicit Anticipation and Controllable Latency" (ACL 2019). Golden
    fixtures: ``tests/fixtures/golden_logs/metrics/al_*.yaml``.
    """

    name: str = "latency/al"
    requires: frozenset[str] = frozenset({"alignment"})

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""
        check_units_match(ref, log)
        emissions = emissions_by_utt(log)
        finals = final_hypotheses(log)

        per_utt: dict[str, float | None] = {}
        defined: list[float] = []
        for utt_id, ref_units in ref.utterances.items():
            spans = _spans_for(ref, utt_id)
            final_hyp = finals.get(utt_id)
            if (
                spans is None
                or final_hyp is None
                or not final_hyp
                or not ref_units
                or len(spans) != len(ref_units)
                or any(span is None for span in spans)
            ):
                per_utt[utt_id] = None
                continue
            ends = sorted(span.f_end for span in spans if span is not None)
            stable = _stable_times(emissions[utt_id], final_hyp)
            source_len = len(ref_units)
            gamma = len(final_hyp) / source_len
            g = [bisect_right(ends, t_frame) for t_frame in stable]
            cutoff = next(
                (i for i, observed in enumerate(g, start=1) if observed == source_len),
                len(final_hyp),
            )
            lag = sum(g[i - 1] - (i - 1) / gamma for i in range(1, cutoff + 1)) / cutoff
            per_utt[utt_id] = lag
            defined.append(lag)

        coverage = len(defined) / len(ref.utterances) if ref.utterances else 0.0
        aggregate: dict[str, float] = {"coverage": coverage}
        if defined:
            aggregate["mean"] = sum(defined) / len(defined)
            aggregate["p50"] = _percentile(defined, 0.5)
            aggregate["p90"] = _percentile(defined, 0.9)
        return MetricResult(name=self.name, per_utt=per_utt, aggregate=aggregate)


def _spans_for(ref: Reference, utt_id: str) -> tuple[Span | None, ...] | None:
    """The utterance's spans, or ``None`` when the run provides no alignment
    or the aligner produced nothing for this utterance."""
    if ref.alignment is None:
        return None
    return ref.alignment.get(utt_id)
