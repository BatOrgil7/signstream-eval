"""Golden-driven tests of the emission-log validator.

Each YAML fixture under ``tests/fixtures/golden_logs/schema/`` is one
complete log plus its hand-derived expectation (valid or not, and exactly
which error/warning codes fire). The test materializes the fixture as a
real run directory and validates it through the same entry point third
parties use (``validate_run_dir``), so file parsing is covered too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from signstream.schema import (
    EMISSIONS_FILENAME,
    RUN_META_FILENAME,
    Reference,
    validate_run_dir,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden_logs" / "schema"
FIXTURE_PATHS = sorted(FIXTURE_DIR.glob("*.yaml"))


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        fixture = yaml.safe_load(handle)
    assert isinstance(fixture, dict), f"fixture {path.name} must be a YAML mapping"
    return fixture


def write_run_dir(base: Path, fixture: dict[str, Any]) -> Path:
    """Materialize a fixture as an on-disk run directory."""
    run_dir = base / "run"
    run_dir.mkdir()
    (run_dir / RUN_META_FILENAME).write_text(
        json.dumps(fixture["run_meta"], ensure_ascii=False), encoding="utf-8"
    )
    if "raw_emissions" in fixture:
        lines = "\n".join(fixture["raw_emissions"]) + "\n"
    else:
        lines = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in fixture["events"])
    (run_dir / EMISSIONS_FILENAME).write_text(lines, encoding="utf-8")
    return run_dir


def reference_of(fixture: dict[str, Any]) -> Reference | None:
    raw = fixture.get("reference")
    return None if raw is None else Reference.from_json_obj(raw)


def test_fixture_directory_is_populated() -> None:
    """Guard against silently running zero golden cases."""
    names = {path.stem for path in FIXTURE_PATHS}
    assert len(names) >= 15
    assert any(name.startswith("valid_") for name in names)
    assert any(name.startswith("invalid_") for name in names)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda p: p.stem)
def test_golden_fixture(path: Path, tmp_path: Path) -> None:
    fixture = load_fixture(path)
    expect = fixture["expect"]
    run_dir = write_run_dir(tmp_path, fixture)

    report = validate_run_dir(run_dir, reference=reference_of(fixture))
    detail = report.summary()

    assert report.ok == expect["valid"], detail
    assert {issue.code for issue in report.errors} == set(expect["error_codes"]), detail
    assert {issue.code for issue in report.warnings} == set(expect["warning_codes"]), detail


@pytest.mark.parametrize(
    "path",
    [p for p in FIXTURE_PATHS if p.stem.startswith("invalid_")],
    ids=lambda p: p.stem,
)
def test_invalid_fixture_errors_are_actionable(path: Path, tmp_path: Path) -> None:
    """Every reported error must say where to look: a file, and — for
    per-line problems — the line number."""
    fixture = load_fixture(path)
    run_dir = write_run_dir(tmp_path, fixture)

    report = validate_run_dir(run_dir, reference=reference_of(fixture))

    assert not report.ok
    for issue in report.errors:
        assert issue.file, issue
        assert issue.message, issue
        rendered = issue.render()
        assert issue.code in rendered
        assert issue.file in rendered
