"""Validation of emission logs against the frozen v1.0 contract.

Two layers, matching what JSON Schema can and cannot express:

1. **Structural** — required fields, types, and value constraints, validated
   per JSON object against ``emission_log.schema.json`` (the same file a
   third party can use from any language).
2. **Semantic** — rules that span lines and are enforced here in code:
   ``utt_start`` precedes an utterance's emissions and appears once;
   ``t_frame`` never decreases within an utterance (equal values are legal —
   the replay loop's final emission may share ``t_frame`` with the last
   step); exactly one ``is_final: true`` per utterance and it is the
   utterance's last event; ``t_frame`` stays within ``0 .. n_frames - 1``;
   and, when a :class:`~signstream.schema.records.Reference` is supplied,
   every reference utterance appears in the log (the anti-cherry-picking
   rule).

All problems are reported as structured :class:`ValidationIssue` records —
with file, line, utterance, and JSON path where applicable — collected into
a :class:`ValidationReport`; nothing here raises on invalid *content*.
:class:`SchemaValidationError` exists for callers (like
``EmissionLog.load``) that need a validated object or nothing, and it
carries the full report.

Dependency policy: standard library + ``jsonschema`` only (leaf package).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaError
from jsonschema.protocols import Validator

from signstream.schema.records import Reference

#: File names of the two per-run log files (spec section 6.1).
RUN_META_FILENAME: Final[str] = "run_meta.json"
EMISSIONS_FILENAME: Final[str] = "emissions.jsonl"

#: The major.minor contract versions this validator understands.
_SUPPORTED_MAJOR_MINOR: Final[tuple[str, str]] = ("1", "0")

_SCHEMA_RESOURCE: Final[str] = "emission_log.schema.json"

Severity: TypeAlias = Literal["error", "warning"]

_EVENT_TYPES: Final[frozenset[str]] = frozenset({"utt_start", "emission"})

#: jsonschema keyword -> stable issue code.
_KEYWORD_CODES: Final[Mapping[str, str]] = {
    "required": "missing-required-field",
    "type": "wrong-type",
    "enum": "invalid-value",
    "const": "invalid-value",
    "minimum": "invalid-value",
    "exclusiveMinimum": "invalid-value",
    "minLength": "invalid-value",
    "pattern": "invalid-format",
}


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found while validating an emission log.

    Attributes:
        severity: ``"error"`` (contract violation) or ``"warning"``
            (suspicious but not forbidden by the contract).
        code: Stable machine-readable identifier of the violated rule.
        message: Human-readable, actionable description.
        file: Which log file the issue is in (``run_meta.json`` /
            ``emissions.jsonl``), or ``"events"`` for in-memory validation.
        line: 1-based line number in ``emissions.jsonl`` (or 1-based event
            index for in-memory validation); ``None`` for file-level issues.
        utt_id: Utterance the issue concerns, when applicable.
        path: JSON path of the offending field within its object,
            e.g. ``$.dataset.unit``.
    """

    severity: Severity
    code: str
    message: str
    file: str | None = None
    line: int | None = None
    utt_id: str | None = None
    path: str | None = None

    def render(self) -> str:
        """Format as a single human-readable report line."""
        location = self.file or "<log>"
        if self.line is not None:
            location += f":{self.line}"
        parts = [f"[{self.severity}] {location}"]
        if self.utt_id is not None:
            parts.append(f"utt '{self.utt_id}'")
        if self.path is not None:
            parts.append(self.path)
        return f"{' '.join(parts)}: {self.message} ({self.code})"


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of validating one emission log.

    Attributes:
        issues: All findings, in file order (``run_meta.json`` first).
    """

    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Issues that make the log invalid under the contract."""
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Suspicious findings that do not violate the contract."""
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        """True when the log has no errors (warnings allowed)."""
        return not self.errors

    def summary(self) -> str:
        """Multi-line human-readable report."""
        errors, warnings = self.errors, self.warnings
        if self.ok:
            head = "emission log is valid"
            if warnings:
                head += f" ({len(warnings)} warning(s))"
        else:
            head = f"emission log is INVALID: {len(errors)} error(s), {len(warnings)} warning(s)"
        lines = [head]
        lines.extend(issue.render() for issue in self.issues)
        return "\n".join(lines)


class SchemaValidationError(ValueError):
    """Raised when a caller requires a valid log and validation found errors.

    The full structured report is available as :attr:`report`; the exception
    message is the report's rendered summary.
    """

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.summary())
        self.report: Final[ValidationReport] = report


def _reject_nonfinite(constant: str) -> object:
    """``parse_constant`` hook: refuse Python's lenient non-JSON literals."""
    raise ValueError(f"non-finite number literal {constant!r} is not valid JSON")


def _strict_json_loads(text: str) -> object:
    """``json.loads`` that rejects ``NaN``/``Infinity``/``-Infinity``.

    JSON numbers are finite by definition; Python's parser accepts these
    extensions by default, which would let a log no conformant third-party
    parser can read slip through validation.
    """
    return json.loads(text, parse_constant=_reject_nonfinite)


@lru_cache(maxsize=1)
def _schema_document() -> dict[str, Any]:
    """Load and cache ``emission_log.schema.json`` from package data."""
    text = resources.files("signstream.schema").joinpath(_SCHEMA_RESOURCE).read_text("utf-8")
    return cast(dict[str, Any], json.loads(text))


@cache
def _validator_for(def_name: str) -> Validator:
    """A cached validator for one ``$defs`` entry of the schema document."""
    document = _schema_document()
    subschema: dict[str, Any] = {"$ref": f"#/$defs/{def_name}", "$defs": document["$defs"]}
    validator: Validator = Draft202012Validator(subschema)
    return validator


def _issue_from_schema_error(
    error: JsonSchemaError,
    *,
    file: str,
    line: int | None = None,
    utt_id: str | None = None,
) -> ValidationIssue:
    """Map a jsonschema error to a :class:`ValidationIssue`."""
    keyword = str(error.validator)
    return ValidationIssue(
        severity="error",
        code=_KEYWORD_CODES.get(keyword, "schema-violation"),
        message=error.message,
        file=file,
        line=line,
        utt_id=utt_id,
        path=error.json_path,
    )


def validate_run_meta_obj(obj: object, *, file: str = RUN_META_FILENAME) -> list[ValidationIssue]:
    """Structurally validate one parsed ``run_meta.json`` object.

    Args:
        obj: The parsed JSON value (any type; non-objects are reported).
        file: File label used in the issues.

    Returns:
        All structural issues, in JSON-path order.
    """
    if not isinstance(obj, Mapping):
        return [
            ValidationIssue(
                severity="error",
                code="wrong-type",
                message=(
                    f"{RUN_META_FILENAME} must contain exactly one JSON object, "
                    f"got {type(obj).__name__}"
                ),
                file=file,
                path="$",
            )
        ]
    errors = sorted(_validator_for("run_meta").iter_errors(obj), key=lambda e: e.json_path)
    issues = [_issue_from_schema_error(e, file=file) for e in errors]

    dataset = obj.get("dataset")
    if isinstance(dataset, Mapping):
        fps = dataset.get("fps")
        if isinstance(fps, float) and not math.isfinite(fps):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid-value",
                    message=f"dataset.fps must be a finite number, got {fps}",
                    file=file,
                    path="$.dataset.fps",
                )
            )

    version = obj.get("schema_version")
    if isinstance(version, str) and re.fullmatch(r"\d+\.\d+(\.\d+)?", version):
        major, minor = version.split(".")[:2]
        if (major, minor) != _SUPPORTED_MAJOR_MINOR:
            supported = ".".join(_SUPPORTED_MAJOR_MINOR)
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unsupported-schema-version",
                    message=(
                        f"schema_version '{version}' is not supported by this validator "
                        f"(supported: {supported}.x); a minor/major bump may add line types "
                        "or break fields this code does not know"
                    ),
                    file=file,
                    path="$.schema_version",
                )
            )
    return issues


def validate_event_rows(
    rows: Sequence[tuple[int, object]],
    *,
    file: str = EMISSIONS_FILENAME,
    reference: Reference | None = None,
) -> list[ValidationIssue]:
    """Validate parsed ``emissions.jsonl`` rows: structure, ordering, coverage.

    Args:
        rows: ``(line_number, parsed_json_value)`` pairs in file order.
        file: File label used in the issues.
        reference: When given, additionally enforce the anti-cherry-picking
            rule: every reference utterance must appear in the log (error);
            logged utterances missing from the reference are flagged as
            warnings.

    Returns:
        All issues in line order (semantic issues after the structural issue
        of the same line, reference-coverage issues last).
    """
    issues: list[ValidationIssue] = []
    valid_rows: list[tuple[int, Mapping[str, Any]]] = []

    for line, obj in rows:
        if not isinstance(obj, Mapping):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid-line",
                    message=(
                        f"each {EMISSIONS_FILENAME} line must be a JSON object, "
                        f"got {type(obj).__name__}"
                    ),
                    file=file,
                    line=line,
                )
            )
            continue
        utt_id = obj.get("utt_id")
        utt_label = utt_id if isinstance(utt_id, str) else None
        event_type = obj.get("type")
        if event_type is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing-required-field",
                    message="line is missing the required 'type' field",
                    file=file,
                    line=line,
                    utt_id=utt_label,
                    path="$.type",
                )
            )
            continue
        if event_type not in _EVENT_TYPES:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unknown-event-type",
                    message=(
                        f"unknown event type {event_type!r}; schema v1.0 defines "
                        "'utt_start' and 'emission' (new line types require a schema "
                        "minor bump)"
                    ),
                    file=file,
                    line=line,
                    utt_id=utt_label,
                    path="$.type",
                )
            )
            continue
        errors = sorted(_validator_for(str(event_type)).iter_errors(obj), key=lambda e: e.json_path)
        if errors:
            issues.extend(
                _issue_from_schema_error(e, file=file, line=line, utt_id=utt_label) for e in errors
            )
            continue
        valid_rows.append((line, obj))

    issues.extend(_validate_event_semantics(valid_rows, file=file, reference=reference))
    return issues


def _validate_event_semantics(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    file: str,
    reference: Reference | None,
) -> list[ValidationIssue]:
    """Cross-line ordering rules over structurally valid rows only."""
    issues: list[ValidationIssue] = []
    started_line: dict[str, int] = {}
    n_frames_by_utt: dict[str, int] = {}
    max_t_frame: dict[str, int] = {}
    final_line: dict[str, int] = {}
    has_emissions: set[str] = set()

    def error(code: str, message: str, line: int, utt_id: str) -> None:
        issues.append(
            ValidationIssue(
                severity="error", code=code, message=message, file=file, line=line, utt_id=utt_id
            )
        )

    for line, obj in rows:
        utt_id = str(obj["utt_id"])
        if obj["type"] == "utt_start":
            if utt_id in started_line:
                error(
                    "duplicate-utt-start",
                    f"utterance already started at {file}:{started_line[utt_id]}; "
                    "each utterance must have exactly one utt_start record",
                    line,
                    utt_id,
                )
            elif utt_id in has_emissions:
                error(
                    "utt-start-after-emission",
                    "utt_start must precede all of the utterance's emissions",
                    line,
                    utt_id,
                )
            else:
                started_line[utt_id] = line
                n_frames_by_utt[utt_id] = int(obj["n_frames"])
            continue

        t_frame = int(obj["t_frame"])
        is_final = bool(obj["is_final"])
        t_wall_ms = obj["t_wall_ms"]
        if isinstance(t_wall_ms, float) and not math.isfinite(t_wall_ms):
            error(
                "invalid-value",
                f"t_wall_ms must be a finite number, got {t_wall_ms}",
                line,
                utt_id,
            )
        if utt_id not in started_line and utt_id not in has_emissions:
            error(
                "emission-before-utt-start",
                "emission for an utterance with no preceding utt_start record",
                line,
                utt_id,
            )
        has_emissions.add(utt_id)
        if utt_id in final_line:
            if is_final:
                error(
                    "multiple-final-emissions",
                    f"utterance already finalized at {file}:{final_line[utt_id]}; "
                    "exactly one is_final=true emission is allowed per utterance",
                    line,
                    utt_id,
                )
            else:
                error(
                    "event-after-final",
                    f"emission after the utterance's final emission at "
                    f"{file}:{final_line[utt_id]}; the final emission must be the "
                    "utterance's last event",
                    line,
                    utt_id,
                )
        if utt_id in max_t_frame and t_frame < max_t_frame[utt_id]:
            error(
                "t-frame-decreasing",
                f"t_frame {t_frame} is earlier than a previous emission's t_frame "
                f"{max_t_frame[utt_id]}; t_frame must never decrease within an utterance",
                line,
                utt_id,
            )
        max_t_frame[utt_id] = max(max_t_frame.get(utt_id, t_frame), t_frame)
        n_frames = n_frames_by_utt.get(utt_id)
        if n_frames is not None and t_frame >= n_frames:
            error(
                "t-frame-out-of-range",
                f"t_frame {t_frame} out of range for an utterance with n_frames "
                f"{n_frames} (valid: 0..{n_frames - 1})",
                line,
                utt_id,
            )
        if is_final and utt_id not in final_line:
            final_line[utt_id] = line

    for utt_id, line in started_line.items():
        if utt_id not in final_line:
            error(
                "missing-final-emission",
                "utterance has no is_final=true emission; empty-output utterances "
                "must still log a final (possibly empty) hypothesis",
                line,
                utt_id,
            )

    if reference is not None:
        logged = set(started_line) | has_emissions
        for utt_id in sorted(reference.utt_ids - logged):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing-reference-utterance",
                    message=(
                        "utterance from the reference split does not appear in the log; "
                        "every reference utterance must be logged (prevents cherry-picking)"
                    ),
                    file=file,
                    utt_id=utt_id,
                )
            )
        for utt_id in sorted(logged - reference.utt_ids):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unexpected-utterance",
                    message="logged utterance does not appear in the reference split",
                    file=file,
                    utt_id=utt_id,
                )
            )
    return issues


def validate_run_dir(run_dir: Path, *, reference: Reference | None = None) -> ValidationReport:
    """Validate one run directory (``run_meta.json`` + ``emissions.jsonl``).

    This is the entry point for third-party logs: it never raises on invalid
    content — every problem, from a missing file to a single bad field,
    becomes an issue in the returned report.

    Args:
        run_dir: Directory containing the two log files.
        reference: Optional reference split for the coverage rule.

    Returns:
        The full validation report.
    """
    issues: list[ValidationIssue] = []

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing-file",
                message=f"{RUN_META_FILENAME} not found in {run_dir}",
                file=RUN_META_FILENAME,
            )
        )
    else:
        try:
            meta_obj: object = _strict_json_loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid-json",
                    message=f"file is not valid JSON: {exc.msg}",
                    file=RUN_META_FILENAME,
                    line=exc.lineno,
                )
            )
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid-json",
                    message=f"file is not valid JSON: {exc}",
                    file=RUN_META_FILENAME,
                )
            )
        else:
            issues.extend(validate_run_meta_obj(meta_obj))

    emissions_path = run_dir / EMISSIONS_FILENAME
    if not emissions_path.is_file():
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing-file",
                message=f"{EMISSIONS_FILENAME} not found in {run_dir}",
                file=EMISSIONS_FILENAME,
            )
        )
    else:
        rows: list[tuple[int, object]] = []
        for line_number, raw_line in enumerate(
            emissions_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                rows.append((line_number, _strict_json_loads(raw_line)))
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid-json",
                        message=f"line is not valid JSON: {exc.msg}",
                        file=EMISSIONS_FILENAME,
                        line=line_number,
                    )
                )
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid-json",
                        message=f"line is not valid JSON: {exc}",
                        file=EMISSIONS_FILENAME,
                        line=line_number,
                    )
                )
        issues.extend(validate_event_rows(rows, reference=reference))

    return ValidationReport(tuple(issues))
