"""Metric engine: every protocol metric as a pure function of (reference, emission log).

Implemented so far (quality family): ``quality/wer`` and ``quality/cer``
over final hypotheses, plus the shared machinery every later metric builds
on — the :class:`Metric` protocol, :class:`MetricResult`, the plain-dict
``REGISTRY`` with its :func:`register` decorator, and
:func:`diff_hypotheses`, the protocol's single Levenshtein-alignment
revision source (latency, stability, and compute families arrive in later
increments and must reuse it, never reimplement it).

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
from signstream.metrics.quality import CharacterErrorRate, WordErrorRate

__all__ = [
    "REGISTRY",
    "CharacterErrorRate",
    "Edit",
    "EditOp",
    "Metric",
    "MetricResult",
    "WordErrorRate",
    "diff_hypotheses",
    "register",
]
