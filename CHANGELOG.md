# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The emission-log schema is versioned separately under semantic versioning; see
`src/signstream/schema/`.

## [Unreleased]

### Added

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
