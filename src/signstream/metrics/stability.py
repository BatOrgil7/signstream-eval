"""Stability metrics: ``stability/upgr``, ``stability/upsr``, ``stability/ne``, ``stability/swe``.

How much do earlier hypotheses get revised as more video arrives? Agents log
full snapshots, never edits; every revision here is *derived* by
:func:`signstream.metrics.diffing.diff_hypotheses` over **consecutive**
emissions — the protocol's single revision source (an edit between
consecutive emissions is the protocol's definition of a revision). The one
exception is ``upsr``, whose spec definition is a plain is-prefix-of-final
predicate and involves no diffing at all.

Composition sensitivity: edit *distance* is tie-break invariant, but edit
*composition* (substitute vs delete+insert) is not, and ``ne``/``swe``/
``upgr`` depend on composition. The differ's frozen, golden-pinned
tie-break (match/substitute, then delete, then insert) is therefore part of
these metrics' operational definition — e.g. a swapped pair scores as two
substitutions (counted by ``swe``/``upgr``, invisible to ``ne``), never as
delete+insert.

None of these metrics read the clock (``t_frame``/``t_wall_ms``) or the
reference content; per-utterance values are keyed by the reference split
and aggregates are corpus-level pooled ratios with ``coverage``, matching
the quality family's convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import Final, Literal, TypeAlias

from signstream.metrics.base import MetricResult, check_units_match, emissions_by_utt, register
from signstream.metrics.diffing import Edit, diff_hypotheses
from signstream.schema import Emission, EmissionLog, Reference, Unit

Severity: TypeAlias = Literal["A", "B", "C"]

#: Default severity weights (spec section 10.2): class A (polarity /
#: sentence-type-bearing units) 5, class B (other unit substitutions /
#: deletions) 2, class C (intra-unit character edits) 1. Configuration, not
#: a constant: the w_A sensitivity sweep passes {3, 5, 10}.
DEFAULT_SWE_WEIGHTS: Final[Mapping[Severity, float]] = {"A": 5.0, "B": 2.0, "C": 1.0}

#: Default class-A token list — the spec's PHOENIX example: the negation
#: glosses NICHT and KEIN plus every NICHT-* compound. A trailing ``*``
#: marks a prefix pattern; per-corpus lists are configuration.
DEFAULT_CLASS_A_TOKENS: Final[tuple[str, ...]] = ("NICHT", "KEIN", "NICHT-*")


def _revision_edits(hyps: Sequence[tuple[str, ...]]) -> list[Edit]:
    """All edits across successive hypothesis diffs, in emission order."""
    edits: list[Edit] = []
    for previous, current in pairwise(hyps):
        edits.extend(diff_hypotheses(previous, current))
    return edits


def _pooled_ratio(
    name: str,
    key: str,
    ref: Reference,
    log: EmissionLog,
    num_den: Callable[[list[Emission]], tuple[float, float]],
) -> MetricResult:
    """Shared per-utterance-ratio + pooled-corpus-aggregate skeleton.

    ``num_den`` maps one utterance's ordered emissions to a (numerator,
    denominator) pair. Per-utterance value is their ratio (``None`` when the
    denominator is zero — the metric is undefined there); the corpus
    aggregate pools both sums, so a numerator arising in an
    undefined-per-utterance case (e.g. deletions before an empty final
    hypothesis) still counts, mirroring ``quality/wer``'s convention.

    Raises:
        ValueError: If the reference and log declare different units.
    """
    check_units_match(ref, log)
    emissions = emissions_by_utt(log)

    per_utt: dict[str, float | None] = {}
    numerator_total = 0.0
    denominator_total = 0.0
    defined = 0
    for utt_id in ref.utterances:
        utt_emissions = emissions.get(utt_id)
        if not utt_emissions:
            per_utt[utt_id] = None
            continue
        numerator, denominator = num_den(utt_emissions)
        numerator_total += numerator
        denominator_total += denominator
        if denominator:
            per_utt[utt_id] = numerator / denominator
            defined += 1
        else:
            per_utt[utt_id] = None

    coverage = defined / len(ref.utterances) if ref.utterances else 0.0
    aggregate: dict[str, float] = {"coverage": coverage}
    if denominator_total:
        aggregate[key] = numerator_total / denominator_total
    return MetricResult(name=name, per_utt=per_utt, aggregate=aggregate)


@register
class UnstablePartialGlossRatio:
    r"""``stability/upgr`` — fraction of emitted partial tokens later revised.

    Per utterance :math:`u`, over successive hypothesis diffs
    :math:`h_0 \to h_1 \to \dots \to h_K` (all emissions in order,
    :math:`h_K` final):

    .. math::

        \mathrm{UPGR}(u) =
        \frac{\sum_{k<K} \bigl( S_{k} + D_{k} \bigr)}
             {\sum_{k<K} |h_k|}

    where :math:`S_k`/:math:`D_k` are substitution/deletion counts of
    ``diff_hypotheses``\ (:math:`h_k`, :math:`h_{k+1}`), and the denominator
    counts every token of every *partial* (non-final) hypothesis snapshot.
    Undefined (``None``) when no partial token was ever emitted (final-only
    utterances, or all partials empty). Corpus aggregate pools both sums;
    coverage reported alongside. Composition-sensitive: revisions the
    frozen tie-break scores as substitutions count here.

    Provenance: the unstable partial word ratio (UPWR) of streaming ASR
    stability analysis (Shangguan et al., 2020), with glosses/dataset units
    in place of words. Golden fixtures:
    ``tests/fixtures/golden_logs/metrics/stability_*.yaml``.
    """

    name: str = "stability/upgr"
    requires: frozenset[str] = frozenset()

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""

        def num_den(utt_emissions: list[Emission]) -> tuple[float, float]:
            hyps = [e.hyp for e in utt_emissions]
            changed = sum(1 for e in _revision_edits(hyps) if e.op in ("substitute", "delete"))
            partial_tokens = sum(len(e.hyp) for e in utt_emissions if not e.is_final)
            return float(changed), float(partial_tokens)

        return _pooled_ratio(self.name, "upgr", ref, log, num_den)


@register
class UnstablePartialSuffixRatio:
    r"""``stability/upsr`` — fraction of partial hypotheses not confirmed by the final.

    Per utterance :math:`u` with final hypothesis :math:`h_K`:

    .. math::

        \mathrm{UPSR}(u) =
        \frac{\#\{\, k < K : h_k \neq h_K[{:}|h_k|] \,\}}{K}

    i.e. the share of partial (non-final) hypotheses that are **not** a
    prefix of the final hypothesis. Undefined (``None``) when the utterance
    has no partials. Defined directly by the spec's prefix predicate — no
    diffing involved, hence fully tie-break insensitive. An empty partial
    is a prefix of anything (stable).

    Provenance: prefix-stability rates for incremental/streaming output
    (cf. the prefix-based stability analyses of re-translation,
    Arivazhagan et al., 2020); protocol instantiation over unit-sequence
    snapshots. Golden fixtures:
    ``tests/fixtures/golden_logs/metrics/stability_*.yaml``.
    """

    name: str = "stability/upsr"
    requires: frozenset[str] = frozenset()

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""

        def num_den(utt_emissions: list[Emission]) -> tuple[float, float]:
            final_hyp = utt_emissions[-1].hyp
            partials = [e.hyp for e in utt_emissions if not e.is_final]
            unstable = sum(1 for h in partials if final_hyp[: len(h)] != h)
            return float(unstable), float(len(partials))

        return _pooled_ratio(self.name, "upsr", ref, log, num_den)


@register
class NormalizedErasure:
    r"""``stability/ne`` — Normalized Erasure: deleted units per final unit.

    Per utterance :math:`u` over successive hypothesis diffs:

    .. math::

        \mathrm{NE}(u) = \frac{\sum_{k<K} D_k}{|h_K|}

    — total ``delete`` edits across consecutive snapshots, normalized by the
    final hypothesis length. Undefined (``None``) when the final hypothesis
    is empty (deletions on such utterances still enter the pooled corpus
    numerator). Composition-sensitive in the *conservative* direction: the
    frozen tie-break prefers substitutions, so a revised-in-place token is a
    substitution (not counted here), and ``ne`` counts only genuine
    retractions of sequence content.

    Provenance: the normalized erasure of re-translation evaluation
    (Arivazhagan et al., 2020), with the token diff supplied by the
    protocol's central Levenshtein alignment instead of prefix comparison.
    Golden fixtures:
    ``tests/fixtures/golden_logs/metrics/stability_*.yaml``.
    """

    name: str = "stability/ne"
    requires: frozenset[str] = frozenset()

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""

        def num_den(utt_emissions: list[Emission]) -> tuple[float, float]:
            hyps = [e.hyp for e in utt_emissions]
            deletes = sum(1 for e in _revision_edits(hyps) if e.op == "delete")
            return float(deletes), float(len(utt_emissions[-1].hyp))

        return _pooled_ratio(self.name, "ne", ref, log, num_den)


@register
class SeverityWeightedErasure:
    r"""``stability/swe`` — Severity-Weighted Erasure with MQM-style classes.

    Per utterance :math:`u`, over the substitution and deletion edits of
    successive hypothesis diffs (insertions add content and erase nothing,
    so they carry no severity class):

    .. math::

        \mathrm{SWE}(u) =
        \frac{\sum_{e \,\in\, \mathrm{sub} \cup \mathrm{del}}
              w\bigl(\mathrm{sev}(e)\bigr)}{|h_K|}

    Severity classes (spec section 10.2): **A** — the edit touches a
    polarity/sentence-type-bearing unit (its old *or* new token matches the
    per-corpus class-A list; trailing ``*`` entries are prefix patterns,
    e.g. PHOENIX ``NICHT``, ``KEIN``, ``NICHT-*``); **C** — intra-unit
    character edits, i.e. every sub/del on character-unit runs
    (fingerspelling); **B** — every other unit substitution/deletion.
    Default weights :math:`w = (A{:}5,\, B{:}2,\, C{:}1)`; weights and the
    class-A list are constructor configuration (the sensitivity study
    sweeps :math:`w_A \in \{3, 5, 10\}`), with the spec defaults registered.
    Undefined (``None``) when the final hypothesis is empty. Satisfies
    :math:`\mathrm{ne} \le \mathrm{swe} / w_{\min}` (property-tested).
    Composition-sensitive: the frozen tie-break's substitutions are
    weighted like deletions of the touched unit.

    Provenance: extends normalized erasure (Arivazhagan et al., 2020) with
    MQM-style severity weighting (Lommel et al., 2014; Freitag et al.,
    2021); the class-A/B/C scheme is this protocol's contribution. Golden
    fixtures: ``tests/fixtures/golden_logs/metrics/stability_*.yaml``.
    """

    name: str = "stability/swe"
    requires: frozenset[str] = frozenset()

    def __init__(
        self,
        weights: Mapping[Severity, float] = DEFAULT_SWE_WEIGHTS,
        class_a_tokens: Sequence[str] = DEFAULT_CLASS_A_TOKENS,
    ) -> None:
        """Configure the severity weighting.

        Args:
            weights: Weight per severity class; must cover A, B, and C.
            class_a_tokens: Per-corpus polarity/sentence-type-bearing token
                list; entries ending in ``*`` match as prefixes.
        """
        self._weights: dict[Severity, float] = dict(weights)
        for severity in ("A", "B", "C"):
            if severity not in self._weights:
                raise ValueError(f"swe weights must define class {severity!r}")
        self._exact = frozenset(t for t in class_a_tokens if not t.endswith("*"))
        self._prefixes = tuple(t[:-1] for t in class_a_tokens if t.endswith("*"))

    def _severity(self, edit: Edit, unit: Unit) -> Severity:
        """Severity class of one sub/del edit under this configuration."""
        if unit == "char":
            return "C"
        for token in (edit.old, edit.new):
            if token is not None and (token in self._exact or token.startswith(self._prefixes)):
                return "A"
        return "B"

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""
        unit = log.run_meta.dataset.unit

        def num_den(utt_emissions: list[Emission]) -> tuple[float, float]:
            hyps = [e.hyp for e in utt_emissions]
            weighted = sum(
                self._weights[self._severity(e, unit)]
                for e in _revision_edits(hyps)
                if e.op in ("substitute", "delete")
            )
            return float(weighted), float(len(utt_emissions[-1].hyp))

        return _pooled_ratio(self.name, "swe", ref, log, num_den)
