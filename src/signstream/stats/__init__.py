"""Statistics engine: from per-utterance metrics to defensible claims.

Turns ``metrics.parquet`` files across runs (indexed by variant x k x seed)
into reported numbers: utterance-level paired bootstrap CIs (B = 10,000,
fixed seed), Wilcoxon signed-rank tests on paired per-utterance differences,
Holm-Bonferroni correction within each metric family, Cliff's delta effect
sizes, seed aggregation, and generated LaTeX tables.

Planned public interface: ``paired_bootstrap_ci(a, b, stat, B, seed)``,
``wilcoxon_paired(a, b)``, ``holm(pvals)``, ``cliffs_delta(a, b)``, and
``make_main_table(...)``.

Pairing discipline: comparisons are within-utterance across conditions —
strict inner join on ``utt_id`` with an assertion that utterance sets are
identical, otherwise abort. Tables are generated, never hand-edited.

Dependency policy: numpy/scipy/pandas only, no torch. Checked with mypy in
strict mode.
"""
