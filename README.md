# signstream-eval

[![CI](https://github.com/BatOrgil7/signstream-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/BatOrgil7/signstream-eval/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A standardized streaming evaluation protocol and open-source harness for
continuous sign language recognition (CSLR).

CSLR systems are conventionally evaluated *offline*: the model sees the whole
video, then reports an error rate. Real use is *streaming*: output is
produced live, mid-signing. Streaming behavior has two properties offline
evaluation ignores — **latency** (how long before the system commits to a
sign) and **stability** (how much earlier output gets revised as more video
arrives). This project provides:

1. **A protocol** defining how to measure them: a versioned emission-log
   schema, a metric suite (quality / latency / stability / compute), and
   statistical reporting rules.
2. **A harness** (this repository) that scores emission *logs* against the
   protocol. It scores logs, not models — any streaming SLR system can adopt
   the protocol by writing logs in the schema, without using our code.
3. **A demonstration study** on PHOENIX-2014T: matched causal / look-ahead /
   bidirectional variants of one landmark-based recognizer, reported as
   latency–quality Pareto curves.

> **Status:** v0.1 in development. The emission-log schema (v1.0) and the
> complete metric suite are implemented and tested: the scoring core
> (`schema`, `metrics`) is covered by **160 tests at 100% statement
> coverage**. The PHOENIX-2014T demonstration study is in progress.

## Metric suite

Every metric is a pure function of a reference transcription and an emission
log, so any streaming system can be scored without running its code.

| Family | Metric ID | What it measures |
|---|---|---|
| Quality | `quality/wer`, `quality/cer` | Word and character error rate of the final hypothesis |
| Latency | `latency/fel_ms` | First-emission latency: time until the system commits to any output |
| Latency | `latency/gel_ms` | Per-gloss emission lag against forced-aligned reference ends |
| Latency | `latency/al` | Average Lagging (Ma et al., 2019), adapted to sign units |
| Stability | `stability/upgr`, `stability/upsr` | Share of partial output later revised (token-level and non-prefix) |
| Stability | `stability/ne` | Normalized erasure: how much displayed output is taken back |
| Stability | `stability/swe` | Severity-weighted erasure: revisions of meaning-critical signs (e.g. negation) cost more |

Latency is measured on a hardware-independent frame clock; computational cost
is recorded separately on a wall clock, so results compare across machines.

## Verification

Each metric is checked three ways: hand-computed golden fixtures
(`tests/fixtures/golden_logs/`), property-based invariants with Hypothesis
(e.g. erasure is zero on monotonic output; `ne <= swe / w_min`), and
independent cross-checks (WER against `jiwer`).

```bash
pytest --cov=signstream.schema --cov=signstream.metrics
```

## Installation

Python ≥ 3.11 is required.

```bash
# Score emission logs only (torch-free — for third parties adopting the protocol)
pip install "signstream-eval[score] @ git+https://github.com/BatOrgil7/signstream-eval"

# Full pipeline: training, streaming simulation, landmark extraction, figures
pip install "signstream-eval[full] @ git+https://github.com/BatOrgil7/signstream-eval"

# Development (from a checkout)
pip install -e ".[dev]"
pre-commit install
```

Fully pinned environments are provided in `requirements.lock` (pip) and
`environment.yml` (conda).

## Repository layout

```
src/signstream/
├── schema/      emission-log contract: typed records, validation, versioning
├── data/        dataset adapters, Sample view, landmark cache builder
├── alignment/   CTC forced-alignment provider (reference timing)
├── models/      Transformer-CTC recognizer + attention-mask variant factory
├── streaming/   simulator, StreamingAgent protocol, dual-clock timing
├── metrics/     quality / latency / stability / compute metrics (pure functions)
├── stats/       paired bootstrap CIs, Wilcoxon, Holm–Bonferroni, effect sizes
├── viz/         publication figure generators
├── runner/      stage orchestration and CLI entrypoints
├── tracking/    experiment-tracker adapters (W&B / MLflow / none)
└── utils/       seeding, hashing, logging setup

configs/         Hydra configuration tree (dataset / model / streaming / …)
scripts/         one-time pipelines (landmark extraction, alignments, tinyset)
tests/           unit + integration suites, golden-log fixtures
docs/            protocol spec, dataset access, reproduction guide, ADRs
paper/           generated figures and tables (committed artifacts)
```

Core packages (`schema`, `metrics`, `stats`) import only
numpy/scipy/stdlib and run without torch; heavyweight dependencies
(torch, mediapipe, hydra, matplotlib) live behind the `full` extra.

## Landmark extraction

The one-time MediaPipe Holistic pass over a corpus (requires the `full`
extra):

```bash
# Corpora shipped as video files
python scripts/extract_landmarks.py --video-dir /data/corpus/videos \
    --cache-dir /data/cache --corpus mycorpus

# PHOENIX-2014T: per-utterance folders of PNG frames at a fixed 25 fps
python scripts/extract_landmarks.py --video-dir /data/phoenix14t/features/fullFrame-210x260px \
    --cache-dir /data/cache --corpus phoenix14t --input-mode frame-folder --fps 25.0
```

Extraction is resumable, content-addressed by extractor version, and writes
a failure manifest. See `docs/adr/0008-pin-mediapipe-version.md` for why
mediapipe is pinned to 0.10.14.

## Development

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy src/signstream     # types (strict on schema/metrics/stats)
pytest                  # tests
```

Datasets are licensed and never redistributed here; see
`docs/dataset_access.md`.

## License

Apache-2.0. See [LICENSE](LICENSE). To cite this work, see
[CITATION.cff](CITATION.cff).
