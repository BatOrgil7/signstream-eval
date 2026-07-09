"""
Landmark cache I/O: content-addressed by (video_id, extractor_version),
atomic writes, and the resumability check the whole pipeline depends on.

Two correctness properties matter more than anything else in this file:

1. Atomicity. If the extraction script is killed mid-write (OOM, SLURM
   preemption, laptop sleeps, you Ctrl-C it) while writing a 200MB .npz, a
   naive `np.savez(final_path, ...)` can leave a truncated, corrupt file at
   `final_path`. The resumability check would then see "file exists" and
   skip it forever, silently poisoning your dataset with one unreadable
   utterance. Fix: write to a temp file, then atomically rename — `os.rename`
   on the same filesystem is atomic on POSIX, so the final path only ever
   contains either nothing or a complete, valid file.

2. Content-addressing. The cache key is (video_id, extractor_version), not
   just video_id. This is what makes EXTRACTOR_VERSION in schema.py load-
   bearing: bump that string when you change how extraction works (new
   mediapipe version, different confidence thresholds, fixed a bug in
   _fill_group), and every video is correctly treated as "not yet extracted
   under the new version" rather than silently reusing stale landmarks.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from signstream.landmarks.schema import EXTRACTOR_VERSION
from signstream.landmarks.video_processor import VideoLandmarks


def cache_path_for(cache_dir: Path, corpus: str, video_id: str) -> Path:
    """
    The one place that decides where a cache entry lives on disk.

    Matches the spec's `data/cache/{corpus}/{utt}.npz` layout. Extractor
    version is stored *inside* the file (see write_cache below) rather than
    baked into the filename, so re-running with a new EXTRACTOR_VERSION
    doesn't require renaming a whole directory tree — the version check
    happens at read time in `is_cached`.
    """
    return cache_dir / corpus / f"{video_id}.npz"


def is_cached(cache_dir: Path, corpus: str, video_id: str) -> bool:
    """
    Resumability check: has this exact video already been extracted under
    the *current* EXTRACTOR_VERSION?

    This is called once per video, before doing any MediaPipe work, so a
    re-run of the extraction script after a crash is cheap: already-done
    videos are skipped in O(1) file-stat + a tiny header read, not
    re-decoded and re-run through Holistic.
    """
    path = cache_path_for(cache_dir, corpus, video_id)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["extractor_version"]) == EXTRACTOR_VERSION
    except Exception:
        # Corrupt or partially-written file that somehow survived (e.g. from
        # a pre-atomic-write bug, or disk corruption). Treat as "not cached"
        # so it gets correctly re-extracted rather than silently skipped.
        return False


def write_cache(
    cache_dir: Path, corpus: str, video_id: str, landmarks: VideoLandmarks
) -> Path:
    """
    Atomically write one video's landmarks to the content-addressed cache.

    Storage dtype is float16, per the frozen spec (§8.3) — normalized
    coordinates in [0, 1]-ish ranges don't need float32 precision, and this
    roughly halves the ~4GB PHOENIX-2014T cache size the spec estimates.
    """
    out_path = cache_path_for(cache_dir, corpus, video_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # video_id may contain path separators (e.g. "subdir/vid_b") to avoid
    # filename collisions across corpus subdirectories -- see
    # extract_landmarks.py:_video_id_from_path. But tempfile.mkstemp's
    # `prefix` must be a flat filename component: a prefix containing "/" is
    # interpreted as a path fragment and mkstemp will try to create the file
    # inside a nested directory that doesn't exist, raising FileNotFoundError.
    # Verified empirically -- this broke on the very first nested-directory
    # test case. Use only the final path component for the prefix; the full
    # video_id (including subdirectories) is still what determines out_path,
    # so no collision risk is reintroduced.
    safe_prefix = Path(video_id).name
    fd, tmp_path_str = tempfile.mkstemp(
        dir=out_path.parent, suffix=".npz.tmp", prefix=f"{safe_prefix}_"
    )
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    try:
        # IMPORTANT: np.savez auto-appends ".npz" to the filename if given a
        # path string that doesn't already end in ".npz" -- so passing
        # tmp_path (which ends in ".npz.tmp") as a string would silently
        # write to "{tmp_path}.npz" instead, breaking the atomic rename
        # below. Passing an open file HANDLE bypasses that auto-suffix
        # behavior entirely. Verified empirically; do not "fix" this by
        # removing the `open(...)` and passing tmp_path directly.
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
                frames=landmarks.frames.astype(np.float16),
                detected=landmarks.detected,
                fps=np.float32(landmarks.fps),
                extractor_version=EXTRACTOR_VERSION,
            )
        os.replace(tmp_path, out_path)  # atomic on POSIX, same filesystem
    finally:
        tmp_path.unlink(missing_ok=True)  # no-op if replace already moved it

    return out_path
