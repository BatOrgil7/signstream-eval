# ADR-0008: Pin mediapipe to 0.10.14 for Holistic landmark extraction

**Status:** Accepted
**Date:** 2026-07-06
**Author:** Bat-Orgil Erdenebat

## Context

The landmark extraction pipeline (`scripts/extract_landmarks.py`) needs
MediaPipe's `Holistic` solution, which returns 543 combined landmarks (33
pose + 468 face + 21 per hand) from a single call — this is the number the
project's design spec is built around (`frames: [T, 543, 3]`).

While setting this up, `pip install mediapipe` installed 0.10.33 by default,
and `mp.solutions.holistic` did not exist in that version —
`AttributeError: module 'mediapipe' has no attribute 'solutions'`. Verified
empirically, not assumed. MediaPipe removed the legacy `solutions` API
(which includes `Holistic`) starting in version 0.10.18, replacing it with a
newer "Tasks" API that does not expose a single combined Holistic model —
Tasks only provides separate PoseLandmarker, FaceLandmarker, and
HandLandmarker models, which would need to be run independently and merged
by hand.

## Decision

Pin `mediapipe==0.10.14` — the last release confirmed (via direct test) to
still expose `mp.solutions.holistic.Holistic`.

## Alternatives considered

1. **Migrate to the Tasks API (PoseLandmarker + FaceLandmarker +
   HandLandmarker run separately).** Rejected for now: this is materially
   more engineering work (three separate model lifecycles instead of one,
   manual landmark-index alignment across three outputs) for zero benefit to
   the thesis's actual contribution, which is the streaming evaluation
   protocol, not the landmark extractor. Revisit only if 0.10.14 becomes
   unavailable or a security/compatibility issue forces an upgrade.
2. **Use a different pose/hand library (OpenPose, MMPose).** Rejected:
   changes the landmark schema entirely, invalidates the 543-landmark
   contract the whole spec is built on, and is a much bigger scope change
   than a version pin.

## Consequences

- The `full` dependency extra in `pyproject.toml` (and `requirements.lock`)
  pins `mediapipe==0.10.14` exactly, not a range.
- If this version is ever yanked from PyPI or becomes incompatible with a
  future Python version, extraction will need this ADR revisited — check
  here first before re-deriving the reasoning from scratch.
- `schema.py`'s `EXTRACTOR_VERSION` constant encodes this version
  (`"mediapipe-0.10.14-holistic-v1"`) so the content-addressed cache
  correctly forces re-extraction if this decision ever changes.
