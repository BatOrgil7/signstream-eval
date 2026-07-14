# SignStream-Eval: Streaming Evaluation Protocol for CSLR

**Protocol version:** 1.0 (draft)
**Schema version:** 1.0

This document is the human-readable, normative specification of the
streaming evaluation protocol: anyone should be able to adopt the protocol
from this file alone, by producing emission logs in the documented schema —
without using this repository's code.

> Skeleton. The normative prose is written incrementally as the
> corresponding modules land; the Metrics section is generated from the
> metric docstrings in `src/signstream/metrics/` (single source of truth).

## 1. Definitions

Emission, revision, algorithmic latency, computational latency, look-ahead
(k), topline, unit (gloss/character). *To be written.*

## 2. Clocks

The logical frame clock (algorithmic latency, hardware-independent) and the
wall clock around each agent step (computational latency,
hardware-dependent) are always recorded together. *To be written.*

## 3. Logging schema (v1.0)

`run_meta.json` (one object per run) + `emissions.jsonl` (one JSON object
per line; full hypothesis snapshots, never diffs). The schema is
implemented and frozen: the formal field rules live in
[`src/signstream/schema/emission_log.schema.json`](../src/signstream/schema/emission_log.schema.json)
(usable from any language), the reference validator — including the
cross-line ordering and coverage rules JSON Schema cannot express — in
`signstream.schema`. *Normative prose to be written.*

## 4. Metric suite

Quality / latency / stability / compute families. *Generated from metric
docstrings once `src/signstream/metrics/` lands.*

## 5. Statistical reporting rules

Paired bootstrap CIs, Wilcoxon signed-rank, Holm-Bonferroni, Cliff's delta,
seed aggregation. *To be written; see `src/signstream/stats/`.*

## 6. Protocol validation

Construct-validity injections (V1-V5) and their pass criteria. *To be
written.*
