# Architecture Decision Records

One file per decision, numbered `NNNN-<slug>.md`. Any deviation from the
frozen design spec during implementation requires a new ADR here explaining
what changed and why.

Numbering convention: IDs 0001–0007 are reserved for the design spec's
pre-made architecture decisions ADR-1 through ADR-7 (logs-as-contract,
custom simulator, dual clocks, matched variants via mask, central
hypothesis diffing, config/tracking provenance, per-dataset unit), so that
file numbers match the spec's cross-references. Implementation-time
decisions start at 0008.
