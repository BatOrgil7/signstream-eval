"""Emission-log contract: typed records, JSON Schema validation, versioned (de)serialization.

Single source of truth for the protocol's data contract (v1.0, frozen). An
emission log is one ``run_meta.json`` object plus one ``emissions.jsonl``
file per run — one JSON object per line, carrying full hypothesis snapshots,
never diffs. Any streaming system, in any language, adopts the protocol by
writing these two files; ``emission_log.schema.json`` (shipped inside this
package) encodes the structural field rules for third-party validators, and
:func:`validate_run_dir` is the reference validator including the cross-line
ordering rules JSON Schema cannot express.

Public interface: :data:`SCHEMA_VERSION`, the record types
(:class:`RunMeta` and its parts, :class:`UttStart` / :class:`Emission` under
the :data:`EmissionEvent` alias, :class:`Reference`), :class:`EmissionLog`
with ``load`` / ``save`` / ``validate``, and the validation types
(:class:`ValidationIssue`, :class:`ValidationReport`,
:class:`SchemaValidationError`).

This is a leaf package: stdlib + ``jsonschema`` only, no torch, no numpy, no
project-internal imports — third parties must be able to validate and score
logs without the modeling stack. Checked with mypy in strict mode.

Schema versioning (semver): patch = additive optional fields; minor = new
line types; major = breaking (not before v2).
"""

from signstream.schema.log import EmissionLog
from signstream.schema.records import (
    SCHEMA_VERSION,
    DatasetInfo,
    Emission,
    EmissionEvent,
    MaskInfo,
    MaskMode,
    ModelInfo,
    Reference,
    RunMeta,
    Unit,
    UttStart,
    WallclockProtocol,
    event_from_json_obj,
)
from signstream.schema.validation import (
    EMISSIONS_FILENAME,
    RUN_META_FILENAME,
    SchemaValidationError,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_event_rows,
    validate_run_dir,
    validate_run_meta_obj,
)

__all__ = [
    "EMISSIONS_FILENAME",
    "RUN_META_FILENAME",
    "SCHEMA_VERSION",
    "DatasetInfo",
    "Emission",
    "EmissionEvent",
    "EmissionLog",
    "MaskInfo",
    "MaskMode",
    "ModelInfo",
    "Reference",
    "RunMeta",
    "SchemaValidationError",
    "Severity",
    "Unit",
    "UttStart",
    "ValidationIssue",
    "ValidationReport",
    "WallclockProtocol",
    "event_from_json_obj",
    "validate_event_rows",
    "validate_run_dir",
    "validate_run_meta_obj",
]
