"""Metric engine: every protocol metric as a pure function of (reference, emission log).

Metric families: quality (WER/CER), latency (first-emission lag, per-gloss
emission lag, average lagging), stability (unstable-partial ratios,
normalized erasure, severity-weighted erasure), and compute (per-step wall
ms, real-time factor).

Planned public interface: the ``Metric`` Protocol (``name``, ``requires``,
``compute(ref, log) -> MetricResult``), the metric ``REGISTRY``,
``score_run(run_dir, metric_cfg)``, and ``diff_hypotheses(prev, next)`` —
the single Levenshtein-based revision source shared by all stability
metrics.

Purity rules: no I/O, no randomness, no global state inside ``compute``.
``None`` is the documented sentinel for "metric undefined for this
utterance"; coverage is reported alongside aggregates. Metric docstrings
carry the math (LaTeX) and the provenance citation for every adapted metric.

Dependency policy: numpy/stdlib only — this package must run without torch
so third parties can score logs. Checked with mypy in strict mode; highest
test-coverage requirement in the repository (>= 90%).
"""
