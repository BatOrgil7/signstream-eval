"""
Landmark schema constants.

Why this file exists on its own:
MediaPipe Holistic's landmark counts (33 pose, 468 face, 21 per hand) are a
*contract* that every downstream consumer of the cache depends on: the model's
input layer, the face-ablation experiment, the tinyset CI fixture generator.
If this number is scattered as a magic literal across multiple files and
MediaPipe ever changes a model version, you get silent shape-mismatch bugs
three modules away from the actual cause. One source of truth fixes that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Landmark group sizes, in the fixed order they are concatenated.
N_POSE_LANDMARKS = 33
N_FACE_LANDMARKS = 468
N_HAND_LANDMARKS = 21  # per hand

N_TOTAL_LANDMARKS = N_POSE_LANDMARKS + N_FACE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS
assert N_TOTAL_LANDMARKS == 543, "Landmark schema drifted from the frozen spec (543)."

N_COORDS = 3  # (x, y, z) per landmark

# Fixed concatenation order along the landmark axis. Downstream code (e.g. a
# face-masking ablation) slices this array by these offsets rather than
# re-deriving them, so the offsets are named constants, not re-computed math.
POSE_SLICE = slice(0, N_POSE_LANDMARKS)
FACE_SLICE = slice(N_POSE_LANDMARKS, N_POSE_LANDMARKS + N_FACE_LANDMARKS)
LEFT_HAND_SLICE = slice(
    N_POSE_LANDMARKS + N_FACE_LANDMARKS,
    N_POSE_LANDMARKS + N_FACE_LANDMARKS + N_HAND_LANDMARKS,
)
RIGHT_HAND_SLICE = slice(
    N_POSE_LANDMARKS + N_FACE_LANDMARKS + N_HAND_LANDMARKS,
    N_TOTAL_LANDMARKS,
)

# Bump this any time the extraction logic OR the pinned mediapipe version
# changes in a way that alters output values. The cache is content-addressed
# by (video_id, extractor_version) — see cache_writer.py — so bumping this
# forces re-extraction instead of silently mixing old/new landmark values.
EXTRACTOR_VERSION = "mediapipe-0.10.14-holistic-v1"


@dataclass(frozen=True)
class FrameLandmarks:
    """
    Output of extracting landmarks from a single video frame.

    `coords` has shape [543, 3]. `detected` has shape [543] and is True for
    landmarks MediaPipe actually found this frame, False for landmarks that
    were zero-filled because the corresponding detector (pose/face/hand)
    failed on this frame (occlusion, out-of-frame, motion blur, etc.).

    Storing `detected` separately from `coords` matters: (0.0, 0.0, 0.0) is a
    *valid* normalized image coordinate (top-left corner), so without this
    mask, a genuinely missing hand is indistinguishable from a hand that is
    truly detected at the top-left of the frame.
    """

    coords: np.ndarray
    detected: np.ndarray
