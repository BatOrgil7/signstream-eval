# Reproducing the paper's numbers

> Skeleton; completed when the score/stats/figures stages and the released
> emission logs exist.

Reproduction is two-level by design — the emission logs released with the
paper are the protocol's public artifact, so every number is recomputable
without access to the licensed videos.

## Level 1 — no data needed

Released emission logs -> `make reproduce` reruns score -> stats -> figures;
every paper number regenerates (target: < 10 min, CPU-only).
*Instructions to be completed.*

## Level 2 — with licensed data

Request the corpora per [dataset_access.md](dataset_access.md), then rerun
the full pipeline (`experiment=e1_pareto -m`); GPU wall-time will be
documented here. *Instructions to be completed.*

## Determinism

Seeded runs with deterministic torch algorithms; any documented exceptions
will be listed here before the CI determinism test may be relaxed.
