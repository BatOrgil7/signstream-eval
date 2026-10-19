"""Metric engine: every protocol metric as a pure function of (reference, emission log).

Implemented so far: the quality family (``quality/wer``, ``quality/cer``),
the algorithmic-latency family (``latency/fel_ms``, ``latency/gel_ms``,
``latency/al`` — all on the logical frame clock; wall-clock metrics are the
separate compute family), and the stability family (``stability/upgr``,
``stability/upsr``, ``stability/ne``, ``stability/swe`` — revisions derived
from consecutive snapshots), plus the shared machinery every metric builds
on — the :class:`Metric` protocol, :class:`MetricResult`, the plain-dict
``REGISTRY`` with its :func:`register` decorator, and
:func:`diff_hypotheses`, the protocol's single Levenshtein-alignment
revision source (the compute family arrives in a later increment and must
reuse it, never reimplement it).

Purity rules: no I/O, no randomness, no global state inside ``compute``.
The scorer loads/validates logs and checks each metric's ``requires``
against what a run provides. ``None`` is the documented sentinel for
"metric undefined for this utterance"; aggregates are reported alongside
``coverage``. Metric docstrings carry their math and provenance citation,
and every metric ships golden fixtures under
``tests/fixtures/golden_logs/``.

Dependency policy: stdlib only here (numpy permitted, torch never) — this
package must run without the modeling stack so third parties can score
logs. ``jiwer`` appears exclusively in the test suite as an independent
cross-check of the in-repo edit distance. Checked with mypy in strict mode;
highest coverage requirement in the repository (>= 90%).
"""

from signstream.metrics.base import REGISTRY, Metric, MetricResult, register
from signstream.metrics.diffing import Edit, EditOp, diff_hypotheses
from signstream.metrics.latency import AverageLagging, FirstEmissionLatency, GlossEmissionLag
from signstream.metrics.quality import CharacterErrorRate, WordErrorRate
from signstream.metrics.stability import (
    DEFAULT_CLASS_A_TOKENS,
    DEFAULT_SWE_WEIGHTS,
    NormalizedErasure,
    Severity,
    SeverityWeightedErasure,
    UnstablePartialGlossRatio,
    UnstablePartialSuffixRatio,
)

__all__ = [
    "DEFAULT_CLASS_A_TOKENS",
    "DEFAULT_SWE_WEIGHTS",
    "REGISTRY",
    "AverageLagging",
    "CharacterErrorRate",
    "Edit",
    "EditOp",
    "FirstEmissionLatency",
    "GlossEmissionLag",
    "Metric",
    "MetricResult",
    "NormalizedErasure",
    "Severity",
    "SeverityWeightedErasure",
    "UnstablePartialGlossRatio",
    "UnstablePartialSuffixRatio",
    "WordErrorRate",
    "diff_hypotheses",
    "register",
]
