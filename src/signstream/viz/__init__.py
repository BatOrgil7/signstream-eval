"""Figure engine: publication figures, each regenerable by one command.

Planned figures: the latency-quality Pareto curve (per-variant curves over
look-ahead k, bidirectional topline at k = infinity, seed+bootstrap CI
bands), stability vs look-ahead, and the hypothesis-evolution timeline strip
for a single utterance.

Planned public interface: ``fig_pareto(df)``, ``fig_stability(df)``,
``fig_timeline(log, utt_id)``, and ``style.apply()``.

Rules: matplotlib only; colorblind-safe Okabe-Ito palette; variant identity
encoded by marker + linestyle, not color alone; vector PDF + 300-dpi PNG
exports. Every figure function is pure (DataFrame -> Figure) and contains no
statistics — it plots precomputed values only, so numbers in figures always
equal numbers in tables.

Depends on matplotlib (``full`` extra).
"""
