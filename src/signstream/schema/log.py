"""The ``EmissionLog`` object: one run's metadata plus its ordered events.

``EmissionLog`` is the in-memory form of a run's two log files
(``run_meta.json`` + ``emissions.jsonl``). ``load`` refuses to construct an
object from invalid files — the raised :class:`SchemaValidationError`
carries the full structured report — so any ``EmissionLog`` in hand is
structurally and semantically valid at load time. ``validate`` re-checks an
in-memory object (e.g. after programmatic construction) and is also where
the reference-coverage rule can be applied.

Dependency policy: standard library only in this module (leaf package).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from signstream.schema.records import EmissionEvent, Reference, RunMeta, event_from_json_obj
from signstream.schema.validation import (
    EMISSIONS_FILENAME,
    RUN_META_FILENAME,
    SchemaValidationError,
    ValidationReport,
    validate_event_rows,
    validate_run_dir,
    validate_run_meta_obj,
)


@dataclass(frozen=True)
class EmissionLog:
    """One run's emission log: the ``run_meta`` cover sheet plus all events.

    Attributes:
        run_meta: Provenance and configuration of the producing run.
        events: All ``emissions.jsonl`` lines, in file order.
    """

    run_meta: RunMeta
    events: tuple[EmissionEvent, ...]

    @classmethod
    def load(cls, run_dir: Path) -> EmissionLog:
        """Load and validate a run directory.

        Args:
            run_dir: Directory containing ``run_meta.json`` and
                ``emissions.jsonl``.

        Returns:
            The validated log.

        Raises:
            SchemaValidationError: If validation finds any error; the
                exception's ``report`` attribute holds every issue.
        """
        report = validate_run_dir(run_dir)
        if not report.ok:
            raise SchemaValidationError(report)
        meta_obj = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
        events = tuple(
            event_from_json_obj(json.loads(line))
            for line in (run_dir / EMISSIONS_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return cls(run_meta=RunMeta.from_json_obj(meta_obj), events=events)

    def save(self, run_dir: Path) -> None:
        """Write ``run_meta.json`` and ``emissions.jsonl`` into ``run_dir``.

        ``run_meta.json`` is indented for humans; emission lines are compact
        (one JSON object per line, no padding), matching the contract's
        size expectations.

        Args:
            run_dir: Target directory; created if missing.

        Raises:
            ValueError: If any value is a non-finite float (NaN/Infinity) —
                such a file would not be valid JSON, so it is never written.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_text = json.dumps(
            self.run_meta.to_json_obj(), indent=2, ensure_ascii=False, allow_nan=False
        )
        (run_dir / RUN_META_FILENAME).write_text(meta_text + "\n", encoding="utf-8")
        lines = "".join(
            json.dumps(
                event.to_json_obj(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            + "\n"
            for event in self.events
        )
        (run_dir / EMISSIONS_FILENAME).write_text(lines, encoding="utf-8")

    def validate(self, reference: Reference | None = None) -> ValidationReport:
        """Validate this in-memory log against the contract.

        Args:
            reference: When given, additionally enforce that every reference
                utterance appears in the log (missing ones are errors;
                logged utterances absent from the reference are warnings).

        Returns:
            The full validation report. Issue ``line`` numbers refer to
            1-based positions in :attr:`events`, and ``file`` is labeled
            ``"events"`` (there is no file on disk to point into).
        """
        issues = validate_run_meta_obj(self.run_meta.to_json_obj(), file="run_meta")
        rows: list[tuple[int, object]] = [
            (index, event.to_json_obj()) for index, event in enumerate(self.events, start=1)
        ]
        issues.extend(validate_event_rows(rows, file="events", reference=reference))
        return ValidationReport(tuple(issues))

    def utt_ids(self) -> tuple[str, ...]:
        """All utterance ids appearing in the log, in first-seen order."""
        seen: dict[str, None] = {}
        for event in self.events:
            seen.setdefault(event.utt_id, None)
        return tuple(seen)
