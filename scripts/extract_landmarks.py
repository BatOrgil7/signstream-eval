#!/usr/bin/env python3
"""
One-time MediaPipe Holistic extraction over an entire corpus, per the frozen
spec (§3.12, §8.3): CPU-bound, multiprocessing, resumable, per-video .npz,
progress bar, and a failure manifest for the <1% of videos MediaPipe can't
process.

Usage:
    python scripts/extract_landmarks.py \\
        --video-dir /data/phoenix14t/videos \\
        --cache-dir /data/cache \\
        --corpus phoenix14t \\
        --workers 8

Why multiprocessing and not multithreading:
This work is CPU-bound (MediaPipe's model inference + video decoding both
saturate a CPU core), and Python's GIL means threads don't give you real
parallelism for CPU-bound work -- only for I/O-bound waiting. Processes each
get their own interpreter and their own GIL, so `workers` processes actually
use `workers` cores. This is the textbook case multiprocessing exists for.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make `signstream` importable when this script is run directly from a repo
# checkout, without requiring an editable pip install first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signstream.data.landmarks.cache_writer import is_cached, write_cache
from signstream.data.landmarks.video_processor import (
    VideoReadError,
    extract_frame_folder_landmarks,
    extract_video_landmarks,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class ExtractionTask:
    video_path: Path
    video_id: str
    corpus: str
    cache_dir: Path
    input_mode: str  # "video" or "frame-folder"
    fps: float | None = None  # required when input_mode == "frame-folder"


@dataclass(frozen=True)
class ExtractionResult:
    video_id: str
    ok: bool
    error_message: str | None = None
    n_frames: int | None = None
    elapsed_seconds: float | None = None


def discover_videos(video_dir: Path) -> list[Path]:
    """Find every video file under `video_dir`, sorted for a deterministic
    processing order (deterministic order matters for reproducing progress
    logs and for making partial-run debugging sane -- "which video was #4127"
    should mean the same thing on every run)."""
    return sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def discover_frame_folders(root_dir: Path) -> list[Path]:
    """
    Find every "utterance folder" under `root_dir`: a directory whose direct
    children include at least one image file. This matches PHOENIX-2014T's
    layout (`features/fullFrame-210x260px/{split}/{utt}/imagesNNNN.png`) --
    one folder per utterance, images as direct children, no nested video
    container. Sorted for the same determinism reason as discover_videos.
    """
    folders = []
    for d in root_dir.rglob("*"):
        if not d.is_dir():
            continue
        if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file()):
            folders.append(d)
    return sorted(folders)


def _video_id_from_path(video_path: Path, video_dir: Path) -> str:
    """utt_id is the video's path relative to the video root, extension
    stripped -- e.g. videos/train/01April_2010.mp4 -> train/01April_2010.
    Using the relative path (not just the filename) avoids id collisions
    across corpus subfolders that happen to share a filename."""
    return str(video_path.relative_to(video_dir).with_suffix(""))


def _process_one(task: ExtractionTask) -> ExtractionResult:
    """
    Runs in a worker process. Must be a module-level function (not a lambda
    or a bound method) so it can be pickled and sent to worker processes --
    this is a common multiprocessing gotcha: Pool.map silently fails with a
    confusing PicklingError if you pass it a closure or an instance method
    that captures unpicklable state.
    """
    start = time.perf_counter()
    if is_cached(task.cache_dir, task.corpus, task.video_id):
        return ExtractionResult(video_id=task.video_id, ok=True, n_frames=None, elapsed_seconds=0.0)

    try:
        if task.input_mode == "frame-folder":
            landmarks = extract_frame_folder_landmarks(task.video_path, fps=task.fps)
        else:
            landmarks = extract_video_landmarks(task.video_path)
        write_cache(task.cache_dir, task.corpus, task.video_id, landmarks)
        return ExtractionResult(
            video_id=task.video_id,
            ok=True,
            n_frames=landmarks.n_frames,
            elapsed_seconds=time.perf_counter() - start,
        )
    except VideoReadError as e:
        # Expected failure mode: corrupt file, bad codec, zero frames. Record
        # and move on -- one bad video must never crash a multi-hour run.
        return ExtractionResult(video_id=task.video_id, ok=False, error_message=str(e))
    except Exception as e:
        # Unexpected failure. Still don't crash the whole pool -- but this
        # gets logged distinctly (see main()) because "unexpected exception"
        # deserves more attention than "known-bad video file."
        return ExtractionResult(
            video_id=task.video_id, ok=False, error_message=f"UNEXPECTED: {e!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["video", "frame-folder"],
        default="video",
        help=(
            "'video': --video-dir contains video files (mp4/avi/...). "
            "'frame-folder': --video-dir contains one subfolder per "
            "utterance, each full of individual frame images -- this is "
            "PHOENIX-2014T's actual on-disk format under "
            "features/fullFrame-210x260px/{split}/."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Required when --input-mode=frame-folder, since a loose image "
            "sequence has no embedded frame-rate metadata. PHOENIX-2014T is "
            "documented as a fixed 25 fps for every clip -- pass --fps 25.0 "
            "explicitly rather than relying on this default, so the value "
            "in your cache is traceable to a decision, not a guess."
        ),
    )
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = parser.parse_args()

    if args.input_mode == "frame-folder" and args.fps is None:
        print(
            "ERROR: --fps is required when --input-mode=frame-folder "
            "(e.g. --fps 25.0 for PHOENIX-2014T).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.input_mode == "frame-folder":
        videos = discover_frame_folders(args.video_dir)
    else:
        videos = discover_videos(args.video_dir)

    if not videos:
        print(f"No videos found under {args.video_dir}", file=sys.stderr)
        sys.exit(1)

    tasks = [
        ExtractionTask(
            video_path=v,
            video_id=_video_id_from_path(v, args.video_dir),
            corpus=args.corpus,
            cache_dir=args.cache_dir,
            input_mode=args.input_mode,
            fps=args.fps,
        )
        for v in videos
    ]

    failure_manifest_path = args.cache_dir / args.corpus / "extraction_failures.jsonl"
    failure_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_failed = 0
    n_skipped_cached = 0
    t_start = time.perf_counter()

    # 'spawn' is safer than the default 'fork' on Linux for processes that
    # load native libraries (mediapipe wraps C++ via native bindings) --
    # fork can inherit a parent's already-initialized native state in a way
    # that causes hard-to-debug crashes in child processes.
    # Write mode ("w"), not append: only successfully-extracted videos are
    # ever cached (write_cache is called only on success), so a failing video
    # is retried on *every* re-run of this script until it succeeds or the
    # underlying file is fixed. If we appended, a video stuck at the same
    # error would get logged again on every run, and this file would grow
    # unboundedly across a 12-week project's worth of re-runs -- verified by
    # actually reproducing the duplicate-entry behavior before fixing it.
    # Each run's manifest should be read as "what's currently broken," a
    # snapshot, not a historical log.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool, open(failure_manifest_path, "w") as fail_log:
        for i, result in enumerate(pool.imap_unordered(_process_one, tasks), start=1):
            if result.ok:
                n_ok += 1
                if result.elapsed_seconds == 0.0 and result.n_frames is None:
                    n_skipped_cached += 1
            else:
                n_failed += 1
                fail_log.write(
                    json.dumps({"video_id": result.video_id, "error": result.error_message}) + "\n"
                )
                fail_log.flush()

            if i % 25 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t_start
                print(
                    f"[{i}/{len(tasks)}] ok={n_ok} (cached={n_skipped_cached}) "
                    f"failed={n_failed} elapsed={elapsed:.0f}s",
                    flush=True,
                )

    failure_rate = n_failed / len(tasks)
    print(
        f"\nDone. {n_ok}/{len(tasks)} succeeded "
        f"({n_skipped_cached} already cached), {n_failed} failed "
        f"({failure_rate:.2%}). Failures logged to {failure_manifest_path}"
    )
    # Spec's own expectation is <1% failures (§8.3). A higher rate almost
    # always means something systemic -- wrong codec assumption, wrong path,
    # a corrupted download batch -- not "many individually bad videos."
    if failure_rate > 0.01:
        print(
            "WARNING: failure rate exceeds the <1% expected in the spec. "
            "Inspect the manifest before trusting this cache.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
