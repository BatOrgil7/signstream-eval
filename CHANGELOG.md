# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The emission-log schema is versioned separately under semantic versioning; see
`src/signstream/schema/`.

## [Unreleased]

### Added

- Stability metrics (`signstream.metrics.stability`), revisions derived
  from consecutive hypothesis snapshots via the central differ:
  `stability/upgr` (unstable partial token ratio, UPWR analog),
  `stability/upsr` (non-prefix partial ratio), `stability/ne` (normalized
  erasure), and `stability/swe` (severity-weighted erasure with
  configurable weights and per-corpus class-A token list; PHOENIX defaults
  `NICHT`/`KEIN`/`NICHT-*`, w = A:5 B:2 C:1). Golden fixtures including a
  tie-break composition pin, plus hypothesis property tests
  (ne <= swe/w_min, order/delay invariance, zero-on-monotonic-runs).

- Latency metrics (`signstream.metrics.latency`), all on the logical frame
  clock: `latency/fel_ms` (first-emission latency), `latency/gel_ms`
  (per-unit emission lag against aligned ends; `requires={"alignment"}`),
  and `latency/al` (Average Lagging, Ma et al. 2019, imported; source axis
  in dataset units via aligned ends; `requires={"alignment"}`), with
  hand-computed golden fixtures and a wait-k invariant test.
- `signstream.schema.Span` and an optional `Reference.alignment` field
  carrying per-unit aligned spans to alignment-requiring metrics.

- Metric engine foundation (`signstream.metrics`): the `Metric` protocol,
  `MetricResult`, the metric registry, and `diff_hypotheses` — the
  protocol's single Levenshtein-alignment revision source. Quality metrics
  `quality/wer` and `quality/cer` over final hypotheses (corpus-level
  aggregates with coverage), with hand-computed golden fixtures under
  `tests/fixtures/golden_logs/metrics/` and a jiwer cross-check in the test
  suite.

- Emission-log schema v1.0 (`signstream.schema`): typed records (`RunMeta`,
  `UttStart`/`Emission` under the `EmissionEvent` alias, `Reference`),
  `EmissionLog` with `load`/`save`/`validate`, a formal JSON Schema
  (`emission_log.schema.json`) for language-independent validation,
  structured validation-error reports covering field rules and cross-line
  ordering (including reference-split coverage), and golden validity
  fixtures under `tests/fixtures/golden_logs/schema/`.

- Repository scaffold: src layout with the full package skeleton
  (`schema`, `data`, `alignment`, `models`, `streaming`, `metrics`, `stats`,
  `viz`, `runner`, `tracking`, `utils`), Hydra config tree, docs skeleton,
  CI workflows, packaging with `score`/`full`/`dev` extras, ruff/mypy/pytest
  configuration, and pre-commit hooks.
- MediaPipe Holistic landmark-extraction pipeline
  (`signstream.data.landmarks`, `scripts/extract_landmarks.py`): video-file
  and frame-folder input, resumable content-addressed `.npz` cache with
  atomic writes, failure manifest.
