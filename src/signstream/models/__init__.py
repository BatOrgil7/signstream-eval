"""Recognizer and matched streaming variants: one Transformer-CTC model, one mask switch.

One ``Recognizer`` class (landmark subset -> linear projection -> pre-LN
Transformer encoder -> CTC head) whose attention mask is the *only*
difference between evaluation variants: bidirectional (offline topline),
causal, and chunked with look-ahead ``k``. Identical initialization across
variants is enforced by seeding before construction and verified by a
parameter-checksum unit test — the "matched variants" claim is true by
construction.

Planned public interface: ``Recognizer.forward(x, lengths, mask)``,
``build_mask(T, mode, chunk, lookahead)``, ``ctc_greedy_decode(logits)``,
``Recognizer.from_checkpoint(path)``, and the ``MODEL_REGISTRY``.

Depends on torch (``full`` extra); never imported by the scoring core.
"""
