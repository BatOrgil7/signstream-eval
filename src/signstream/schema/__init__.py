"""Emission-log contract: typed records, JSON Schema validation, versioned (de)serialization.

Single source of truth for the protocol's data contract. An emission log is
one ``run_meta.json`` object plus one ``emissions.jsonl`` file per run (one
JSON object per line: utterance-start records and emission events carrying
full hypothesis snapshots, never diffs).

Planned public interface: ``RunMeta``, ``EmissionEvent``,
``EmissionLog.load(path)`` / ``.save(path)`` / ``.validate()``, and
``SCHEMA_VERSION``.

This is a leaf module: stdlib + ``jsonschema`` only, no torch, no
project-internal imports — third parties must be able to validate and score
logs without the modeling stack. Checked with mypy in strict mode.

Schema versioning (semver): patch = additive optional fields; minor = new
line types; major = breaking.
"""

from typing import Final

#: Version of the emission-log contract — the protocol's public interface.
SCHEMA_VERSION: Final[str] = "1.0"

__all__ = ["SCHEMA_VERSION"]
