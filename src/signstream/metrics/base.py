"""Metric abstraction layer: the ``Metric`` protocol, ``MetricResult``, and the registry.

Every protocol metric is a pure function ``(Reference, EmissionLog) ->
MetricResult``: no I/O, no randomness, no global mutable state inside
``compute``. The scorer — not the metric — loads and validates logs, checks
each metric's ``requires`` against what a run provides, and skips with a
warning otherwise. ``None`` in ``per_utt`` is the one documented sentinel:
"this metric is undefined for this utterance"; every aggregate is reported
next to a ``coverage`` value so undefined utterances are never silently
dropped.

Registration is deliberately boring: a plain dict plus a decorator, no
plugin framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from signstream.schema import EmissionLog, Reference


@dataclass(frozen=True)
class MetricResult:
    """Outcome of one metric over one run.

    Attributes:
        name: Metric key, ``family/name`` (e.g. ``"quality/wer"``).
        per_utt: Value per reference utterance; ``None`` means the metric is
            undefined for that utterance (the documented sentinel).
        aggregate: Test-set aggregates (e.g. the corpus-level value and
            ``coverage``, the fraction of utterances where the metric is
            defined).
    """

    name: str
    per_utt: Mapping[str, float | None]
    aggregate: Mapping[str, float]


class Metric(Protocol):
    """The protocol every metric implements.

    Attributes:
        name: Metric key, ``family/name``.
        requires: Extra inputs the metric needs beyond ``(reference, log)``;
            a subset of ``{"alignment", "tiers", "wallclock"}``. The scorer
            skips (with a warning) metrics whose requirements a run does not
            provide.
    """

    name: str
    requires: frozenset[str]

    def compute(self, ref: Reference, log: EmissionLog) -> MetricResult:
        """Compute the metric. Pure: no I/O, no randomness, no global state."""
        ...


#: All registered metrics, keyed by metric name. Populated at import time by
#: the :func:`register` decorator; never mutated by ``compute``.
REGISTRY: dict[str, Metric] = {}

_M = TypeVar("_M", bound=type[Metric])


def register(cls: _M) -> _M:
    """Class decorator adding one (stateless) metric instance to ``REGISTRY``.

    Args:
        cls: A zero-argument-constructible class implementing
            :class:`Metric`.

    Returns:
        ``cls`` unchanged, so the decorator stacks cleanly.

    Raises:
        ValueError: If a metric with the same name is already registered.
    """
    instance = cls()
    if instance.name in REGISTRY:
        raise ValueError(f"metric {instance.name!r} is already registered")
    REGISTRY[instance.name] = instance
    return cls
