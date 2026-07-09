"""Streaming evaluation protocol and harness for continuous sign language recognition.

The package is organized around one guiding decision: **the emission log is
the protocol**. Every subpackage is either a producer of emission logs
(:mod:`signstream.streaming` replaying a recognizer on a logical frame
clock) or a consumer of them (:mod:`signstream.metrics` ->
:mod:`signstream.stats` -> :mod:`signstream.viz`), fully decoupled by the
versioned, validated JSONL schema in :mod:`signstream.schema`.

Dependency policy: the scoring core (``schema``, ``metrics``, ``stats``)
imports only numpy/scipy/stdlib and runs without torch, so third parties can
score their own logs with ``pip install signstream-eval[score]``. Everything
that needs torch, mediapipe, hydra, or matplotlib lives behind the ``full``
extra.
"""

__version__ = "0.1.0"
