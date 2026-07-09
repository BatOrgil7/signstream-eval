"""Reference timing provider: CTC forced alignment of reference units to frame spans.

Latency metrics need to know *when* each reference unit ends. Those spans
come from CTC forced alignment (``torchaudio.functional.forced_align``)
using the trained bidirectional checkpoint — the best offline model is the
least-noisy teacher — cached to ``alignments/{corpus}/{utt_id}.json``.

Planned public interface: ``AlignmentProvider.get(utt_id) -> list[Span]``
and ``build_alignments(ckpt, split)``, plus an alignment-confidence report.

Alignment noise is a protocol-level caveat, not a hidden assumption: every
latency metric reports coverage, and the protocol includes a ±2-frame
span-perturbation sensitivity check.

Depends on torch (``full`` extra); never imported by the scoring core.
"""
