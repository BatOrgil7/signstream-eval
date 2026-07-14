"""Round-trip tests: load -> save -> reload must be lossless.

Losslessness includes unknown extra fields (forward compatibility): a
third-party log with fields this schema version does not model must survive
a load/save cycle byte-for-byte at the JSON level.
"""

from __future__ import annotations

import json
from pathlib import Path

from test_schema_golden import FIXTURE_DIR, load_fixture, write_run_dir

from signstream.schema import EMISSIONS_FILENAME, RUN_META_FILENAME, EmissionLog


def _roundtrip(fixture_name: str, tmp_path: Path) -> tuple[EmissionLog, EmissionLog, Path]:
    fixture = load_fixture(FIXTURE_DIR / f"{fixture_name}.yaml")
    source_dir = write_run_dir(tmp_path, fixture)
    loaded = EmissionLog.load(source_dir)
    saved_dir = tmp_path / "resaved"
    loaded.save(saved_dir)
    reloaded = EmissionLog.load(saved_dir)
    return loaded, reloaded, saved_dir


def test_typical_log_roundtrips(tmp_path: Path) -> None:
    loaded, reloaded, saved_dir = _roundtrip("valid_typical", tmp_path)
    assert reloaded == loaded
    assert loaded.validate().ok
    assert loaded.utt_ids() == (
        "01April_2010_Thursday_heute-6694",
        "27October_2009_Tuesday_tagesschau-3479",
    )

    fixture = load_fixture(FIXTURE_DIR / "valid_typical.yaml")
    saved_meta = json.loads((saved_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert saved_meta == fixture["run_meta"]
    saved_events = [
        json.loads(line)
        for line in (saved_dir / EMISSIONS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert saved_events == fixture["events"]


def test_unknown_extra_fields_are_preserved(tmp_path: Path) -> None:
    loaded, reloaded, saved_dir = _roundtrip("valid_extra_fields", tmp_path)
    assert reloaded == loaded

    assert loaded.run_meta.extra["protocol_notes"] == "written by an external system"
    assert loaded.run_meta.dataset.extra["language"] == "DGS"
    assert loaded.run_meta.model.mask.extra["implementation"] == "flash"
    assert loaded.run_meta.wallclock_protocol.extra["clock_source"] == "monotonic"
    assert loaded.events[0].extra["source"] == "camera-2"
    assert loaded.events[1].extra["confidence"] == 0.71

    fixture = load_fixture(FIXTURE_DIR / "valid_extra_fields.yaml")
    saved_meta = json.loads((saved_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert saved_meta == fixture["run_meta"]
    saved_events = [
        json.loads(line)
        for line in (saved_dir / EMISSIONS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert saved_events == fixture["events"]


def test_null_mask_geometry_roundtrips(tmp_path: Path) -> None:
    loaded, reloaded, _ = _roundtrip("valid_offline_topline", tmp_path)
    assert reloaded == loaded
    assert loaded.run_meta.model.mask.mode == "bidirectional"
    assert loaded.run_meta.model.mask.chunk_frames is None
    assert loaded.run_meta.model.mask.lookahead_frames is None
