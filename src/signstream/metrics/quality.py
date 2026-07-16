"""Quality metrics: final-hypothesis error rates (``quality/wer``, ``quality/cer``).

Both metrics are the same computation — Levenshtein error counts between an
utterance's *final* hypothesis and its reference — parameterized by the
protocol's per-dataset token unit (ADR-7): ``quality/wer`` is defined on
gloss-unit runs, ``quality/cer`` on character-unit runs (fingerspelling,
where the logged units *are* characters). On a run with the other unit the
metric is undefined for every utterance (``per_utt`` all ``None``,
``coverage`` 0) — mirroring the scorer's skip-with-warning philosophy while
keeping ``compute`` total.

Edit counts come from :func:`signstream.metrics.diffing.diff_hypotheses`,
the protocol's single edit-distance implementation; ``jiwer`` is an
independent cross-check in the test suite only.
"""

from __future__ import annotations

from signstream.metrics.base import MetricResult, register
from signstream.metrics.diffing import diff_hypotheses
from signstream.schema import Emission, EmissionLog, Reference, Unit


def _final_hypotheses(log: EmissionLog) -> dict[str, tuple[str, ...]]:
    """The final hypothesis of every utterance in the log.

    Validated logs carry exactly one final emission per utterance; if a
    malformed log carries several, the last one wins (metrics stay total —
    validation, not metrics, is where malformed logs are rejected).
    """
    finals: dict[str, tuple[str, ...]] = {}
    for event in log.events:
        if isinstance(event, Emission) and event.is_final:
            finals[event.utt_id] = event.hyp
    return finals


def _error_rate_result(
    name: str,
    aggregate_key: str,
    expected_unit: Unit,
    ref: Reference,
    log: EmissionLog,
) -> MetricResult:
    """Shared WER/CER computation (see the metric docstrings for the math).

    Args:
        name: Metric key (``family/name``).
        aggregate_key: Key of the corpus-level value in ``aggregate``.
        expected_unit: Unit this metric is defined for; other units yield
            an all-undefined result.
        ref: Reference split (defines the utterance set).
        log: Emission log to score.

    Returns:
        Per-utterance rates plus ``{aggregate_key: corpus rate, "coverage":
        defined fraction}``. The corpus key is omitted when the summed
        reference length is zero (corpus rate undefined).

    Raises:
        ValueError: If the reference and the log declare different units —
            a pairing error by the caller, not a scoreable condition.
    """
    log_unit = log.run_meta.dataset.unit
    if ref.unit != log_unit:
        raise ValueError(
            f"reference unit {ref.unit!r} does not match the log's dataset unit "
            f"{log_unit!r}; refusing to score a mismatched pair"
        )

    per_utt: dict[str, float | None] = {}
    if ref.unit != expected_unit:
        for utt_id in ref.utterances:
            per_utt[utt_id] = None
        return MetricResult(name=name, per_utt=per_utt, aggregate={"coverage": 0.0})

    finals = _final_hypotheses(log)
    total_edits = 0
    total_ref_len = 0
    defined = 0
    for utt_id, ref_units in ref.utterances.items():
        hyp = finals.get(utt_id)
        if hyp is None:
            # No final hypothesis for this utterance: nothing to score.
            per_utt[utt_id] = None
            continue
        edits = len(diff_hypotheses(ref_units, hyp))
        total_edits += edits
        total_ref_len += len(ref_units)
        if ref_units:
            per_utt[utt_id] = edits / len(ref_units)
            defined += 1
        else:
            # Empty reference: the per-utterance ratio is undefined, but the
            # hypothesis's insertions still count in the corpus-level sums.
            per_utt[utt_id] = None

    coverage = defined / len(ref.utterances) if ref.utterances else 0.0
    aggregate: dict[str, float] = {"coverage": coverage}
    if total_ref_len > 0:
        aggregate[aggregate_key] = total_edits / total_ref_len
    return MetricResult(name=name, per_utt=per_utt, aggregate=aggregate)


@register
class WordErrorRate:
    r"""``quality/wer`` — word (gloss) error rate of the final hypothesis.

    Per utterance :math:`u` with reference length :math:`N_u`:

    .. math::

        \mathrm{WER}(u) = \frac{S_u + D_u + I_u}{N_u}

    where :math:`S_u`, :math:`D_u`, :math:`I_u` are the substitution,
    deletion, and insertion counts of a minimal Levenshtein alignment
    between the final hypothesis and the reference unit sequence.
    Test-set aggregate is corpus-level, the field's standard convention:

    .. math::

        \mathrm{WER} = \frac{\sum_u (S_u + D_u + I_u)}{\sum_u N_u}

    Undefined (``None``) for utterances with an empty reference
    (:math:`N_u = 0`; their insertions still count in the corpus sums) or
    with no final hypothesis in the log; ``coverage`` reports the defined
    fraction. Defined only on gloss-unit runs (ADR-7).

    Provenance: the standard word error rate of speech recognition
    (Levenshtein, 1966; see e.g. Jurafsky & Martin, *Speech and Language
    Processing*), as conventionally reported for CSLR gloss output on
    PHOENIX-2014/2014T (Koller et al., 2015). Golden fixtures:
    ``tests/fixtures/golden_logs/metrics/wer_*.yaml``.
    """

    name: str = "quality/wer"
    requires: frozenset[str] = frozenset()

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""
        return _error_rate_result(self.name, "wer", "gloss", ref, log)


@register
class CharacterErrorRate:
    r"""``quality/cer`` — character error rate of the final hypothesis.

    The same computation as ``quality/wer`` with characters as the symbol:
    on character-unit runs (ADR-7; e.g. FSboard fingerspelling) the logged
    unit sequences *are* character sequences, so

    .. math::

        \mathrm{CER}(u) = \frac{S_u + D_u + I_u}{N_u},
        \qquad
        \mathrm{CER} = \frac{\sum_u (S_u + D_u + I_u)}{\sum_u N_u}

    over minimal Levenshtein alignments of character sequences. Undefined
    (``None``) per utterance under the same rules as ``quality/wer``, and
    undefined for every utterance on non-character-unit runs.

    Provenance: the standard character error rate used for fingerspelling
    recognition (e.g. Shi et al., 2018, ChicagoFSWild) and character-level
    sequence transcription generally. Golden fixtures:
    ``tests/fixtures/golden_logs/metrics/cer_*.yaml``.
    """

    name: str = "quality/cer"
    requires: frozenset[str] = frozenset()

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Pure computation; see the class docstring for the math."""
        return _error_rate_result(self.name, "cer", "char", ref, log)
