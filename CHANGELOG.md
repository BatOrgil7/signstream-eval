# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The emission-log schema is versioned separately under semantic versioning; see
`src/signstream/schema/`.

## [Unreleased]

### Added

- Repository scaffold: src layout with the full package skeleton
  (`schema`, `data`, `alignment`, `models`, `streaming`, `metrics`, `stats`,
  `viz`, `runner`, `tracking`, `utils`), Hydra config tree, docs skeleton,
  CI workflows, packaging with `score`/`full`/`dev` extras, ruff/mypy/pytest
  configuration, and pre-commit hooks.
- MediaPipe Holistic landmark-extraction pipeline
  (`signstream.data.landmarks`, `scripts/extract_landmarks.py`): video-file
  and frame-folder input, resumable content-addressed `.npz` cache with
  atomic writes, failure manifest.
