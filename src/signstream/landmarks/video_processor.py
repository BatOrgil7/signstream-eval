"""
Video -> [T, 543, 3] landmark array.

This module owns OpenCV. Nothing outside this file should call cv2 directly
for landmark extraction — same insulation principle as extractor.py owning
mediapipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from signstream.landmarks.extractor import HolisticFrameExtractor
from signstream.landmarks.schema import N_COORDS, N_TOTAL_LANDMARKS


class VideoReadError(RuntimeError):
    """Raised when a video file cannot be opened or yields zero readable frames.

    A dedicated exception type (rather than letting a raw cv2/OS error
    propagate) matters for the orchestration script: it needs to distinguish
    "this specific video is corrupt, log it and move on" from "something is
    structurally wrong with the whole run, stop everything." A bare Exception
    can't be told apart from a bug in our own code.
    """


@dataclass(frozen=True)
class VideoLandmarks:
    """Full extraction result for one video, ready to be cached."""

    frames: np.ndarray  # [T, 543, 3] float32
    detected: np.ndarray  # [T, 543] bool
    fps: float
    n_frames: int


def extract_video_landmarks(video_path: Path) -> VideoLandmarks:
    """
    Run Holistic extraction over every frame of one video FILE (mp4/avi/etc).

    Use this for corpora that ship as actual video containers. For corpora
    that ship as a folder of individual frame images (e.g. PHOENIX-2014T's
    `features/fullFrame-210x260px/{split}/{utt}/` -- confirmed by inspecting
    the dataset's own HuggingFace loading script, which does
    `os.listdir(frames_path)` over individual PNGs, NOT `cv2.VideoCapture`),
    use `extract_frame_folder_landmarks` instead.

    Args:
        video_path: path to a video file OpenCV can decode (mp4, avi, etc.)

    Returns:
        VideoLandmarks for the whole clip.

    Raises:
        VideoReadError: if the file can't be opened, or opens but has 0 frames
            (both happen in practice with corrupted downloads or codec
            mismatches — your own audit doc flagged "<1% failures" as the
            expected rate for this exact kind of thing).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoReadError(f"OpenCV could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # Some containers don't report fps reliably; PHOENIX-2014T is 25fps
        # per your own spec (§5.1) — but don't silently assume that for every
        # corpus. Fail loudly instead of writing a cache entry with a wrong
        # fps that only shows up as a bug three steps later, in a latency
        # number that's subtly off.
        cap.release()
        raise VideoReadError(f"Could not read a valid fps for: {video_path}")

    per_frame_coords: list[np.ndarray] = []
    per_frame_detected: list[np.ndarray] = []

    with HolisticFrameExtractor() as extractor:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = extractor.extract(frame_rgb)
            per_frame_coords.append(result.coords)
            per_frame_detected.append(result.detected)

    cap.release()

    if len(per_frame_coords) == 0:
        raise VideoReadError(f"Zero decodable frames: {video_path}")

    frames = np.stack(per_frame_coords, axis=0)  # [T, 543, 3]
    detected = np.stack(per_frame_detected, axis=0)  # [T, 543]

    assert frames.shape[1:] == (N_TOTAL_LANDMARKS, N_COORDS)

    return VideoLandmarks(
        frames=frames,
        detected=detected,
        fps=float(fps),
        n_frames=frames.shape[0],
    )


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def extract_frame_folder_landmarks(frame_dir: Path, fps: float) -> VideoLandmarks:
    """
    Run Holistic extraction over a folder of individual frame images.

    This is the path PHOENIX-2014T actually needs: its `features/
    fullFrame-210x260px/{split}/{utt}/` directories each contain the frames
    of one utterance as separate image files (e.g. `images0001.png`,
    `images0002.png`, ...), not a video container. There is no embedded fps
    metadata in a folder of images, so `fps` must be supplied by the caller
    -- PHOENIX-2014T's own documentation states a fixed 25 fps for every
    clip, confirmed on the dataset's official page, so callers extracting
    this corpus should pass `fps=25.0` explicitly rather than guessing.

    Args:
        frame_dir: directory containing one image file per frame.
        fps: frame rate for this clip. Must come from the corpus's
            documented value, not inferred, since no embedded metadata
            exists for a loose image sequence.

    Returns:
        VideoLandmarks for the whole clip.

    Raises:
        VideoReadError: if the directory has zero readable image frames.
    """
    frame_paths = sorted(
        p for p in frame_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not frame_paths:
        raise VideoReadError(f"Zero image frames found in: {frame_dir}")

    per_frame_coords: list[np.ndarray] = []
    per_frame_detected: list[np.ndarray] = []

    with HolisticFrameExtractor() as extractor:
        for frame_path in frame_paths:
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                # A single unreadable frame inside an otherwise-good clip.
                # Treated the same as "detector found nothing" for that
                # frame -- zero-filled + marked not-detected -- rather than
                # failing the whole utterance over one bad image, since the
                # rest of the clip's landmarks are still usable evidence.
                per_frame_coords.append(np.zeros((N_TOTAL_LANDMARKS, N_COORDS), dtype=np.float32))
                per_frame_detected.append(np.zeros(N_TOTAL_LANDMARKS, dtype=bool))
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = extractor.extract(frame_rgb)
            per_frame_coords.append(result.coords)
            per_frame_detected.append(result.detected)

    frames = np.stack(per_frame_coords, axis=0)
    detected = np.stack(per_frame_detected, axis=0)
    assert frames.shape[1:] == (N_TOTAL_LANDMARKS, N_COORDS)

    return VideoLandmarks(
        frames=frames,
        detected=detected,
        fps=float(fps),
        n_frames=frames.shape[0],
    )
