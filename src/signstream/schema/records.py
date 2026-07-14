"""Typed records of the emission-log contract (schema v1.0, frozen).

Every class here mirrors one JSON object of the protocol's on-disk format:
``RunMeta`` (and its nested parts) is the single object in ``run_meta.json``;
``UttStart`` and ``Emission`` are the two ``emissions.jsonl`` line types,
united under the ``EmissionEvent`` alias. ``Reference`` is the ground-truth
object metrics compare logs against.

Forward compatibility: the contract allows unknown extra fields on every
object (a schema-version *patch* adds optional fields). Each record
therefore carries an ``extra`` mapping holding any fields this version does
not model, so loading and re-saving a third-party log is lossless.

``from_json_obj`` constructors assume the raw object has already passed
structural validation (see :mod:`signstream.schema.validation`); they are
not a validation layer themselves.

Dependency policy: standard library only in this module. No torch, no
numpy, no project-internal imports (leaf package).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Literal, TypeAlias, cast

#: Version of the emission-log contract (semver; the protocol's public
#: interface). Bumps: patch = additive optional fields; minor = new line
#: types; major = breaking (not before v2).
SCHEMA_VERSION: Final[str] = "1.0"

#: The protocol's token decision, declared per dataset and echoed into every
#: log (ADR-7): word-level glosses or characters (fingerspelling).
Unit: TypeAlias = Literal["gloss", "char"]

#: Attention-mask variant of the recognizer that produced a run (ADR-4).
MaskMode: TypeAlias = Literal["bidirectional", "causal", "chunked"]

JsonObj: TypeAlias = dict[str, Any]


def _extras(raw: Mapping[str, Any], known: frozenset[str]) -> dict[str, Any]:
    """Collect fields of ``raw`` that this schema version does not model."""
    return {k: v for k, v in raw.items() if k not in known}


@dataclass(frozen=True)
class DatasetInfo:
    """The ``dataset`` block of ``run_meta.json``.

    Attributes:
        name: Corpus identifier (e.g. ``"phoenix14t"``).
        split: Evaluated split (e.g. ``"test"``).
        fps: Frame rate of the corpus's logical frame clock (> 0).
        unit: Token unit for ``hyp`` sequences: ``"gloss"`` or ``"char"``.
        extra: Unknown extra fields, preserved for round-tripping.
    """

    name: str
    split: str
    fps: float
    unit: Unit
    extra: Mapping[str, Any] = field(default_factory=dict)

    _KNOWN: ClassVar[frozenset[str]] = frozenset({"name", "split", "fps", "unit"})

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> DatasetInfo:
        """Build from a structurally validated JSON object."""
        return cls(
            name=str(raw["name"]),
            split=str(raw["split"]),
            fps=float(raw["fps"]),
            unit=cast(Unit, raw["unit"]),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON shape (spec field order, extras last)."""
        obj: JsonObj = {"name": self.name, "split": self.split, "fps": self.fps, "unit": self.unit}
        obj.update(self.extra)
        return obj


@dataclass(frozen=True)
class MaskInfo:
    """The ``model.mask`` block: which matched variant produced the run.

    ``chunk_frames`` and ``lookahead_frames`` are ``None`` for the
    bidirectional offline topline (the k = infinity point), where neither is
    meaningful; streaming variants record non-null integers.

    Attributes:
        mode: ``"bidirectional"``, ``"causal"``, or ``"chunked"``.
        chunk_frames: Frames per replay chunk (>= 1), or ``None``.
        lookahead_frames: Look-ahead budget k in frames (>= 0), or ``None``.
        extra: Unknown extra fields, preserved for round-tripping.
    """

    mode: MaskMode
    chunk_frames: int | None
    lookahead_frames: int | None
    extra: Mapping[str, Any] = field(default_factory=dict)

    _KNOWN: ClassVar[frozenset[str]] = frozenset({"mode", "chunk_frames", "lookahead_frames"})

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> MaskInfo:
        """Build from a structurally validated JSON object."""
        chunk = raw["chunk_frames"]
        look = raw["lookahead_frames"]
        return cls(
            mode=cast(MaskMode, raw["mode"]),
            chunk_frames=None if chunk is None else int(chunk),
            lookahead_frames=None if look is None else int(look),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON shape (spec field order, extras last)."""
        obj: JsonObj = {
            "mode": self.mode,
            "chunk_frames": self.chunk_frames,
            "lookahead_frames": self.lookahead_frames,
        }
        obj.update(self.extra)
        return obj


@dataclass(frozen=True)
class ModelInfo:
    """The ``model`` block of ``run_meta.json``.

    Attributes:
        name: Recognizer identifier (e.g. ``"transformer_ctc"``).
        params: Parameter count of the model (>= 0).
        mask: The variant switch that distinguishes matched runs.
        extra: Unknown extra fields, preserved for round-tripping.
    """

    name: str
    params: int
    mask: MaskInfo
    extra: Mapping[str, Any] = field(default_factory=dict)

    _KNOWN: ClassVar[frozenset[str]] = frozenset({"name", "params", "mask"})

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> ModelInfo:
        """Build from a structurally validated JSON object."""
        return cls(
            name=str(raw["name"]),
            params=int(raw["params"]),
            mask=MaskInfo.from_json_obj(raw["mask"]),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON shape (spec field order, extras last)."""
        obj: JsonObj = {"name": self.name, "params": self.params, "mask": self.mask.to_json_obj()}
        obj.update(self.extra)
        return obj


@dataclass(frozen=True)
class WallclockProtocol:
    """The ``wallclock_protocol`` block: how computational latency was measured.

    Attributes:
        batch_size: Inference batch size during simulation (protocol: 1).
        warmup_utts: Utterances excluded from wall-clock aggregates
            (protocol: 10).
        timer: Timer used around each agent step (protocol:
            ``"perf_counter"``).
        extra: Unknown extra fields, preserved for round-tripping.
    """

    batch_size: int
    warmup_utts: int
    timer: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    _KNOWN: ClassVar[frozenset[str]] = frozenset({"batch_size", "warmup_utts", "timer"})

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> WallclockProtocol:
        """Build from a structurally validated JSON object."""
        return cls(
            batch_size=int(raw["batch_size"]),
            warmup_utts=int(raw["warmup_utts"]),
            timer=str(raw["timer"]),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON shape (spec field order, extras last)."""
        obj: JsonObj = {
            "batch_size": self.batch_size,
            "warmup_utts": self.warmup_utts,
            "timer": self.timer,
        }
        obj.update(self.extra)
        return obj


@dataclass(frozen=True)
class RunMeta:
    """The single JSON object in ``run_meta.json`` — a run's cover sheet.

    Records full provenance so any number derived from the run's emissions
    can be traced to code, config, data, and hardware.

    Attributes:
        schema_version: Contract version the log was written against.
        run_id: Run identifier (e.g. ``"E1/chunked-k4-s42"``).
        created_utc: Creation time, ISO-8601 UTC with a ``Z`` suffix.
        git_sha: Commit of the producing code; ``-dirty`` suffix allowed.
        config_hash: 8-hex-digit hash of the composed run config.
        experiment: Experiment name (e.g. ``"e1_pareto"``).
        dataset: Corpus, split, fps, and unit declaration.
        model: Recognizer identity and the matched-variant mask.
        seed: Seed the run was trained/simulated with.
        hardware: Free-form hardware block (GPU, driver, torch,
            determinism flags, ...). Kept free-form: CPU-only and
            third-party runs record different keys.
        wallclock_protocol: How computational latency was measured.
        extra: Unknown extra fields, preserved for round-tripping.
    """

    schema_version: str
    run_id: str
    created_utc: str
    git_sha: str
    config_hash: str
    experiment: str
    dataset: DatasetInfo
    model: ModelInfo
    seed: int
    hardware: Mapping[str, Any]
    wallclock_protocol: WallclockProtocol
    extra: Mapping[str, Any] = field(default_factory=dict)

    _KNOWN: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "run_id",
            "created_utc",
            "git_sha",
            "config_hash",
            "experiment",
            "dataset",
            "model",
            "seed",
            "hardware",
            "wallclock_protocol",
        }
    )

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> RunMeta:
        """Build from a structurally validated JSON object."""
        return cls(
            schema_version=str(raw["schema_version"]),
            run_id=str(raw["run_id"]),
            created_utc=str(raw["created_utc"]),
            git_sha=str(raw["git_sha"]),
            config_hash=str(raw["config_hash"]),
            experiment=str(raw["experiment"]),
            dataset=DatasetInfo.from_json_obj(raw["dataset"]),
            model=ModelInfo.from_json_obj(raw["model"]),
            seed=int(raw["seed"]),
            hardware=dict(raw["hardware"]),
            wallclock_protocol=WallclockProtocol.from_json_obj(raw["wallclock_protocol"]),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON shape (spec field order, extras last)."""
        obj: JsonObj = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "experiment": self.experiment,
            "dataset": self.dataset.to_json_obj(),
            "model": self.model.to_json_obj(),
            "seed": self.seed,
            "hardware": dict(self.hardware),
            "wallclock_protocol": self.wallclock_protocol.to_json_obj(),
        }
        obj.update(self.extra)
        return obj


@dataclass(frozen=True)
class UttStart:
    """An ``emissions.jsonl`` line of type ``"utt_start"``.

    Announces one utterance before any of its emissions and fixes its
    length on the logical frame clock.

    Attributes:
        utt_id: Utterance identifier, unique within the run.
        n_frames: Total frames T of the utterance (>= 1); valid emission
            times are ``0 .. n_frames - 1``.
        extra: Unknown extra fields, preserved for round-tripping.
    """

    utt_id: str
    n_frames: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    TYPE: ClassVar[Literal["utt_start"]] = "utt_start"
    _KNOWN: ClassVar[frozenset[str]] = frozenset({"type", "utt_id", "n_frames"})

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> UttStart:
        """Build from a structurally validated JSON object."""
        return cls(
            utt_id=str(raw["utt_id"]),
            n_frames=int(raw["n_frames"]),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON line shape (extras last)."""
        obj: JsonObj = {"type": self.TYPE, "utt_id": self.utt_id, "n_frames": self.n_frames}
        obj.update(self.extra)
        return obj


@dataclass(frozen=True)
class Emission:
    """An ``emissions.jsonl`` line of type ``"emission"``.

    One full hypothesis snapshot (never a diff — ADR-5; revisions are
    derived downstream by the metric engine). An emission at ``t_frame``
    means "hypothesis available after observing frame ``t_frame``".

    Attributes:
        utt_id: Utterance this emission belongs to.
        t_frame: Newest observed frame on the logical clock (0-based).
        t_wall_ms: Wall-clock milliseconds of the producing step (>= 0).
        hyp: Full unit sequence (units per ``dataset.unit``); may be empty.
        is_final: True for the utterance's single final hypothesis, which
            must be its last logged event.
        extra: Unknown extra fields, preserved for round-tripping.
    """

    utt_id: str
    t_frame: int
    t_wall_ms: float
    hyp: tuple[str, ...]
    is_final: bool
    extra: Mapping[str, Any] = field(default_factory=dict)

    TYPE: ClassVar[Literal["emission"]] = "emission"
    _KNOWN: ClassVar[frozenset[str]] = frozenset(
        {"type", "utt_id", "t_frame", "t_wall_ms", "hyp", "is_final"}
    )

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> Emission:
        """Build from a structurally validated JSON object."""
        return cls(
            utt_id=str(raw["utt_id"]),
            t_frame=int(raw["t_frame"]),
            t_wall_ms=float(raw["t_wall_ms"]),
            hyp=tuple(str(u) for u in raw["hyp"]),
            is_final=bool(raw["is_final"]),
            extra=_extras(raw, cls._KNOWN),
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to the schema's JSON line shape (extras last)."""
        obj: JsonObj = {
            "type": self.TYPE,
            "utt_id": self.utt_id,
            "t_frame": self.t_frame,
            "t_wall_ms": self.t_wall_ms,
            "hyp": list(self.hyp),
            "is_final": self.is_final,
        }
        obj.update(self.extra)
        return obj


#: One ``emissions.jsonl`` line: either of the two v1.0 line types. A schema
#: *minor* bump is what introduces new line types.
EmissionEvent: TypeAlias = UttStart | Emission


def event_from_json_obj(raw: Mapping[str, Any]) -> EmissionEvent:
    """Build the right event class from a structurally validated JSON line.

    Args:
        raw: A parsed ``emissions.jsonl`` line object.

    Returns:
        The corresponding :class:`UttStart` or :class:`Emission`.

    Raises:
        ValueError: If ``raw["type"]`` is not a v1.0 line type (callers are
            expected to have validated the line first).
    """
    event_type = raw.get("type")
    if event_type == UttStart.TYPE:
        return UttStart.from_json_obj(raw)
    if event_type == Emission.TYPE:
        return Emission.from_json_obj(raw)
    raise ValueError(f"unknown emission-log line type: {event_type!r}")


@dataclass(frozen=True)
class Reference:
    """Ground truth the metric engine compares an emission log against.

    Carries the per-utterance reference unit sequences and the unit
    declaration for one dataset split. Alignment spans (per-unit frame
    times) are intentionally *not* part of this object — they are produced
    by the alignment provider and consumed only by metrics that declare
    ``requires={"alignment"}``.

    Attributes:
        unit: Token unit of the reference sequences: ``"gloss"`` or
            ``"char"``.
        utterances: Mapping from ``utt_id`` to its reference unit sequence.
    """

    unit: Unit
    utterances: Mapping[str, tuple[str, ...]]

    @property
    def utt_ids(self) -> frozenset[str]:
        """All utterance ids in the reference split."""
        return frozenset(self.utterances)

    @classmethod
    def from_json_obj(cls, raw: Mapping[str, Any]) -> Reference:
        """Build from a JSON object of shape ``{"unit": ..., "utterances": {id: [units]}}``."""
        return cls(
            unit=cast(Unit, raw["unit"]),
            utterances={
                str(utt_id): tuple(str(u) for u in units)
                for utt_id, units in raw["utterances"].items()
            },
        )

    def to_json_obj(self) -> JsonObj:
        """Serialize to ``{"unit": ..., "utterances": {id: [units]}}``."""
        return {
            "unit": self.unit,
            "utterances": {utt_id: list(units) for utt_id, units in self.utterances.items()},
        }
