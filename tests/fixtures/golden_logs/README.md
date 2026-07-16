# Golden log fixtures

Tiny handcrafted emission logs with hand-computed expectations, documented
inside each fixture file (YAML). Two layers share this directory:

- `schema/` — **contract validity goldens**: each file holds one complete
  log (`run_meta` + `events`, or `raw_emissions` for deliberately broken
  bytes), an optional `reference`, and an `expect` block
  (`valid`, `error_codes`, `warning_codes`). Valid fixtures cover the
  typical, edge, and reference-coverage cases; every validation rule has at
  least one invalid fixture that triggers it.
- `metrics/` — **metric goldens**: each file holds a complete schema-valid
  log, its reference, and an `expect.metrics` block mapping metric name to
  the hand-computed per-utterance values (`null` = undefined for that
  utterance) and aggregates; the derivation is documented in-file. At least
  three goldens per metric (typical, edge, undefined/None case).
  `diff_cases.yaml` pins the exact edit scripts of `diff_hypotheses`,
  including its deterministic tie-breaking rule.

Fixture files are the executable specification: when the contract and a
fixture disagree, one of them is wrong — investigate before editing either.
