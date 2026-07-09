"""Cross-cutting utilities: determinism, provenance, logging, atomic I/O.

Planned public interface: ``seed_everything(seed)`` (python/numpy/torch plus
deterministic cudnn flags), ``config_hash(cfg)`` (the short hash stamped on
every artifact), ``git_sha()``, structured logging setup (the ``logging``
module only — no ``print`` anywhere in ``src/``), and atomic file writes.

Kept dependency-light; torch is touched only inside ``seed_everything`` and
imported lazily so the scoring core stays torch-free.
"""
