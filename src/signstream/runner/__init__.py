"""Experiment orchestration: the Hydra-driven stage graph and CLI.

Stage graph with explicit dependencies:
``extract -> train -> align -> simulate -> score -> stats -> figures``.
Each stage implements ``outputs_exist(cfg)`` (checks a ``DONE.<stage>``
marker plus expected files) and ``run(cfg)``; ``stage=all`` walks the graph
skipping satisfied stages, making runs idempotent and resumable. A failed
stage leaves no DONE marker and writes ``FAILED.<stage>.txt`` with the
traceback.

Planned public interface: the ``Stage`` Protocol and the CLI entrypoint
``python -m signstream.run stage=<name> experiment=<name> [overrides]``.

Every run directory is stamped with the composed config, git SHA, config
hash, and schema version; sweeps are Hydra multiruns only, so the sweep grid
itself is version-controlled.

Depends on hydra-core and the pipeline modules (``full`` extra).
"""
