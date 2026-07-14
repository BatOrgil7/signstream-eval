# Golden log fixtures

Tiny handcrafted emission logs with hand-computed expectations, documented
inside each fixture file (YAML). Two layers share this directory:

- `schema/` — **contract validity goldens**: each file holds one complete
  log (`run_meta` + `events`, or `raw_emissions` for deliberately broken
  bytes), an optional `reference`, and an `expect` block
  (`valid`, `error_codes`, `warning_codes`). Valid fixtures cover the
  typical, edge, and reference-coverage cases; every validation rule has at
  least one invalid fixture that triggers it.
- *(from the metrics increment)* metric goldens: logs plus the expected
  value of every metric, at least three per metric (typical, edge,
  undefined/None case).

Fixture files are the executable specification: when the contract and a
fixture disagree, one of them is wrong — investigate before editing either.
