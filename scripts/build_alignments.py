#!/usr/bin/env python3
"""Build CTC forced-alignment spans from a trained bidirectional checkpoint.

Produces ``alignments/{corpus}/{utt_id}.json`` — the per-reference-unit
(start, end) frame spans that latency metrics require — plus an
alignment-confidence report. Uses the best offline (bidirectional)
checkpoint as the least-noisy teacher.

Not implemented yet: requires the trained recognizer from the model plane
(:mod:`signstream.models`) and the alignment provider
(:mod:`signstream.alignment`).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "build_alignments.py requires a trained bidirectional checkpoint; "
        "it is implemented together with signstream.alignment."
    )


if __name__ == "__main__":
    main()
