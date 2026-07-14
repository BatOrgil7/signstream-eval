"""API-level tests of the schema package.

Covers the behaviors golden fixtures cannot express by themselves: the
load-or-raise contract (the exception carries the structured report),
missing-file reporting, programmatic construction, in-memory validation of
an object that was never on disk, and the reference-coverage semantics of
``EmissionLog.validate``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_schema_golden import FIXTURE_DIR, load_fixture, write_run_dir

from signstream.schema import (
    SCHEMA_VERSION,
    DatasetInfo,
    Emission,
    EmissionLog,
    MaskInfo,
    ModelInfo,
    Reference,
    RunMeta,
    SchemaValidationError,
    UttStart,
    WallclockProtocol,
    event_from_json_obj,
    validate_run_dir,
)


def make_log() -> EmissionLog:
    """A small, valid, programmatically constructed log."""
    run_meta = RunMeta(
        schema_version=SCHEMA_VERSION,
        run_id="E1/chunked-k4-s42",
        created_utc="2026-08-14T09:31:07Z",
        git_sha="a1b2c3d",
        config_hash="9f2e77c1",
        experiment="e1_pareto",
        dataset=DatasetInfo(name="phoenix14t", split="test", fps=25.0, unit="gloss"),
        model=ModelInfo(
            name="transformer_ctc",
            params=9_200_000,
            mask=MaskInfo(mode="chunked", chunk_frames=4, lookahead_frames=4),
        ),
        seed=42,
        hardware={"gpu": "NVIDIA RTX 4090 24GB", "deterministic": True},
        wallclock_protocol=WallclockProtocol(batch_size=1, warmup_utts=10, timer="perf_counter"),
    )
    events = (
        UttStart(utt_id="u1", n_frames=30),
        Emission(utt_id="u1", t_frame=10, t_wall_ms=9.1, hyp=("MORGEN",), is_final=False),
        Emission(utt_id="u1", t_frame=29, t_wall_ms=9.0, hyp=("MORGEN", "REGEN"), is_final=True),
        UttStart(utt_id="u2", n_frames=20),
        Emission(utt_id="u2", t_frame=19, t_wall_ms=8.7, hyp=(), is_final=True),
    )
    return EmissionLog(run_meta=run_meta, events=events)


def test_schema_version_is_frozen() -> None:
    assert SCHEMA_VERSION == "1.0"
    assert make_log().run_meta.schema_version == "1.0"


def test_load_raises_with_structured_report(tmp_path: Path) -> None:
    fixture = load_fixture(FIXTURE_DIR / "invalid_t_frame_decreasing.yaml")
    run_dir = write_run_dir(tmp_path, fixture)

    with pytest.raises(SchemaValidationError) as exc_info:
        EmissionLog.load(run_dir)

    report = exc_info.value.report
    assert not report.ok
    [issue] = report.errors
    assert issue.code == "t-frame-decreasing"
    assert issue.file == "emissions.jsonl"
    assert issue.line == 3
    assert issue.utt_id == "u1"
    # The exception message is the rendered report: actionable as-is.
    message = str(exc_info.value)
    assert "t-frame-decreasing" in message
    assert "emissions.jsonl:3" in message


def test_missing_files_are_reported_not_raised(tmp_path: Path) -> None:
    report = validate_run_dir(tmp_path)
    assert not report.ok
    assert {issue.code for issue in report.errors} == {"missing-file"}
    assert {issue.file for issue in report.errors} == {"run_meta.json", "emissions.jsonl"}

    with pytest.raises(SchemaValidationError):
        EmissionLog.load(tmp_path)


def test_programmatic_log_saves_and_validates(tmp_path: Path) -> None:
    log = make_log()
    assert log.validate().ok

    run_dir = tmp_path / "run"
    log.save(run_dir)
    assert (run_dir / "run_meta.json").is_file()
    assert (run_dir / "emissions.jsonl").is_file()
    assert EmissionLog.load(run_dir) == log


def test_in_memory_validation_reports_event_positions() -> None:
    log = make_log()
    # Break the ordering rules in memory: an emission after u1's final.
    broken = EmissionLog(
        run_meta=log.run_meta,
        events=(
            *log.events,
            Emission(utt_id="u1", t_frame=29, t_wall_ms=1.0, hyp=("X",), is_final=False),
        ),
    )
    report = broken.validate()
    [issue] = report.errors
    assert issue.code == "event-after-final"
    assert issue.file == "events"
    assert issue.line == 6  # 1-based position in EmissionLog.events
    assert issue.utt_id == "u1"


def test_unsupported_version_in_memory() -> None:
    log = make_log()
    meta = RunMeta(
        schema_version="1.1",
        run_id=log.run_meta.run_id,
        created_utc=log.run_meta.created_utc,
        git_sha=log.run_meta.git_sha,
        config_hash=log.run_meta.config_hash,
        experiment=log.run_meta.experiment,
        dataset=log.run_meta.dataset,
        model=log.run_meta.model,
        seed=log.run_meta.seed,
        hardware=log.run_meta.hardware,
        wallclock_protocol=log.run_meta.wallclock_protocol,
    )
    report = EmissionLog(run_meta=meta, events=log.events).validate()
    assert {issue.code for issue in report.errors} == {"unsupported-schema-version"}


def test_reference_coverage_semantics() -> None:
    log = make_log()  # logs u1 and u2

    exact = Reference(unit="gloss", utterances={"u1": ("MORGEN", "REGEN"), "u2": ()})
    report = log.validate(exact)
    assert report.ok
    assert not report.warnings

    missing = Reference(
        unit="gloss",
        utterances={"u1": ("MORGEN", "REGEN"), "u2": (), "u3": ("WIND",)},
    )
    report = log.validate(missing)
    assert not report.ok
    [issue] = report.errors
    assert issue.code == "missing-reference-utterance"
    assert issue.utt_id == "u3"

    # A logged utterance missing from the reference is only a warning:
    # suspicious (wrong split?) but not forbidden by the contract.
    subset = Reference(unit="gloss", utterances={"u1": ("MORGEN", "REGEN")})
    report = log.validate(subset)
    assert report.ok
    [warning] = report.warnings
    assert warning.code == "unexpected-utterance"
    assert warning.utt_id == "u2"


def test_report_summary_is_human_readable() -> None:
    log = make_log()
    assert log.validate().summary() == "emission log is valid"

    broken = EmissionLog(run_meta=log.run_meta, events=log.events[:2])  # u1 never finalized
    summary = broken.validate().summary()
    assert summary.startswith("emission log is INVALID: 1 error(s)")
    assert "missing-final-emission" in summary

    # Warnings alone leave the log valid, and the summary says so.
    with_warning = log.validate(Reference(unit="gloss", utterances={"u1": ("MORGEN", "REGEN")}))
    assert with_warning.summary().startswith("emission log is valid (1 warning(s))")


def test_broken_run_meta_json_is_reported(tmp_path: Path) -> None:
    log = make_log()
    run_dir = tmp_path / "run"
    log.save(run_dir)
    (run_dir / "run_meta.json").write_text('{"schema_version": ', encoding="utf-8")

    report = validate_run_dir(run_dir)
    [issue] = report.errors
    assert issue.code == "invalid-json"
    assert issue.file == "run_meta.json"


def test_blank_lines_in_emissions_are_tolerated(tmp_path: Path) -> None:
    log = make_log()
    run_dir = tmp_path / "run"
    log.save(run_dir)
    emissions = run_dir / "emissions.jsonl"
    lines = emissions.read_text(encoding="utf-8").splitlines()
    lines.insert(2, "   ")
    lines.insert(0, "")
    emissions.write_text("\n".join(lines) + "\n\n", encoding="utf-8")

    assert validate_run_dir(run_dir).ok
    assert EmissionLog.load(run_dir) == log


def test_event_from_json_obj_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown emission-log line type"):
        event_from_json_obj({"type": "revision", "utt_id": "u1"})


def test_reference_json_roundtrip() -> None:
    reference = Reference(unit="gloss", utterances={"u1": ("MORGEN", "REGEN"), "u2": ()})
    assert Reference.from_json_obj(reference.to_json_obj()) == reference
    assert reference.utt_ids == frozenset({"u1", "u2"})


def _with_nan_emission(log: EmissionLog) -> EmissionLog:
    return EmissionLog(
        run_meta=log.run_meta,
        events=(
            UttStart(utt_id="u9", n_frames=10),
            Emission(utt_id="u9", t_frame=9, t_wall_ms=float("nan"), hyp=(), is_final=True),
        ),
    )


def test_nonfinite_values_rejected_in_memory() -> None:
    log = make_log()

    report = _with_nan_emission(log).validate()
    assert {issue.code for issue in report.errors} == {"invalid-value"}
    [issue] = report.errors
    assert "t_wall_ms" in issue.message

    meta = replace(
        log.run_meta,
        dataset=DatasetInfo(name="phoenix14t", split="test", fps=float("inf"), unit="gloss"),
    )
    report = EmissionLog(run_meta=meta, events=log.events).validate()
    assert {issue.code for issue in report.errors} == {"invalid-value"}
    [issue] = report.errors
    assert issue.path == "$.dataset.fps"


def test_save_refuses_nonfinite_values(tmp_path: Path) -> None:
    # A NaN could otherwise be serialized as a literal no conformant JSON
    # parser accepts; save() must refuse rather than write such a file.
    with pytest.raises(ValueError):
        _with_nan_emission(make_log()).save(tmp_path / "run")
