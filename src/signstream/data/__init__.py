"""Datasets and the landmark cache: a uniform ``Sample`` view over heterogeneous corpora.

Planned public interface: ``Span`` and ``Sample`` (frozen dataclasses),
``BaseDataset`` (Protocol), per-corpus adapters (``Phoenix14TDataset``,
``FSboardDataset``, ``TinySetDataset``), ``build_cache``, and the
PyTorch ``Dataset``/``DataLoader`` wrappers with padding and collate.

A ``Sample`` carries ``frames`` of shape ``[T, 543, 3]`` float16 (MediaPipe
Holistic landmarks), the corpus fps, the reference unit sequence, and the
per-dataset unit decision (gloss or character).

The MediaPipe Holistic cache-building pipeline lives in
:mod:`signstream.data.landmarks`. It is deliberately not imported here:
importing ``signstream.data`` must never require mediapipe or opencv, which
are ``full``-extra dependencies.
"""
