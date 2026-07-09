"""Streaming simulator and agents: chunked replay on a logical frame clock.

Replays a ``Sample`` chunk-by-chunk, drives a ``StreamingAgent``, measures
per-step wall time, and writes schema-valid emission logs. Two clocks are
always recorded together: the logical frame clock (algorithmic latency,
hardware-independent) and the wall clock around each ``agent.step()`` call
(computational latency, hardware-dependent).

Planned public interface: the ``StreamingAgent`` Protocol
(``reset(meta)``, ``step(chunk, t_frame) -> Hypothesis``,
``finalize() -> Hypothesis``), ``Simulator.run(dataset, agent, cfg)``, and
``RecognizerAgent`` (recompute-full-prefix decoding each step).

Agents emit a full hypothesis snapshot every step, even if unchanged;
revisions are derived centrally by the metric engine, never self-reported.
Look-ahead is enforced both by the replay loop (frames physically withheld)
and by the attention mask.

Depends on torch via the model plane (``full`` extra).
"""
