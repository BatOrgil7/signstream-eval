"""
Single-frame landmark extraction via MediaPipe Holistic.

Design note (why this is its own class, not a function):
Every other module that touches "landmarks" (the cache writer, the video
processor, tests) should depend on the *shape* of what this class produces
(FrameLandmarks: [543, 3] + a detected mask), never on `mediapipe` directly.
That is the Dependency Inversion half of SOLID: if MediaPipe ever changes its
API (as it already has once — see the version note below), exactly one file
needs to change. Everything downstream is insulated.

Pinned dependency: mediapipe==0.10.14. Versions 0.10.18+ removed the legacy
`solutions.holistic.Holistic` API in favor of a Tasks API that does not
expose a single combined Holistic model — verified empirically 2026-07-06,
not from memory, precisely because "I'm pretty sure MediaPipe has Holistic"
is exactly the kind of assumption that wastes a week when it's wrong.
"""

from __future__ import annotations

import numpy as np

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover - environment guard, not logic
    raise ImportError(
        "mediapipe is required. Install the pinned version with: "
        "pip install mediapipe==0.10.14 --break-system-packages"
    ) from e

from signstream.data.landmarks.schema import (
    N_COORDS,
    N_FACE_LANDMARKS,
    N_HAND_LANDMARKS,
    N_POSE_LANDMARKS,
    N_TOTAL_LANDMARKS,
    FrameLandmarks,
)


class HolisticFrameExtractor:
    """
    Stateful wrapper around one `mp.solutions.holistic.Holistic` instance.

    Stateful on purpose: MediaPipe's model expects to be reused across
    consecutive frames of the *same* video for its internal tracking to work
    well (it's not a stateless per-frame classifier — it tracks landmarks
    frame-to-frame and only re-detects from scratch when tracking is lost).
    Constructing a fresh Holistic() per frame would both be ~10x slower and
    quietly degrade tracking quality. One instance per video is the contract;
    see VideoLandmarkProcessor for the corresponding lifecycle management.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_rgb: np.ndarray) -> FrameLandmarks:
        """
        Extract landmarks from one RGB frame.

        Args:
            frame_rgb: [H, W, 3] uint8 array, RGB channel order (NOT BGR —
                OpenCV reads frames as BGR by default; the caller is
                responsible for the conversion, so this class never silently
                gets it wrong for you).

        Returns:
            FrameLandmarks with coords[543, 3] float32 and detected[543] bool.
        """
        results = self._holistic.process(frame_rgb)

        coords = np.zeros((N_TOTAL_LANDMARKS, N_COORDS), dtype=np.float32)
        detected = np.zeros(N_TOTAL_LANDMARKS, dtype=bool)

        offset = 0
        offset = self._fill_group(
            coords, detected, offset, results.pose_landmarks, N_POSE_LANDMARKS
        )
        offset = self._fill_group(
            coords, detected, offset, results.face_landmarks, N_FACE_LANDMARKS
        )
        offset = self._fill_group(
            coords, detected, offset, results.left_hand_landmarks, N_HAND_LANDMARKS
        )
        offset = self._fill_group(
            coords, detected, offset, results.right_hand_landmarks, N_HAND_LANDMARKS
        )

        return FrameLandmarks(coords=coords, detected=detected)

    @staticmethod
    def _fill_group(
        coords: np.ndarray,
        detected: np.ndarray,
        offset: int,
        landmark_group,  # mediapipe NormalizedLandmarkList | None
        group_size: int,
    ) -> int:
        """
        Write one detector's output (pose/face/left-hand/right-hand) into the
        shared [543, 3] array at `offset`, or leave it zero-filled with
        detected=False if that detector found nothing this frame.

        Returns the next offset, so the caller can chain groups without
        hand-tracking arithmetic at every call site (a small thing, but it's
        the difference between one place that can have an off-by-one and
        four).
        """
        if landmark_group is not None:
            for i, lm in enumerate(landmark_group.landmark):
                coords[offset + i] = (lm.x, lm.y, lm.z)
            detected[offset : offset + group_size] = True
        return offset + group_size

    def close(self) -> None:
        """Release the underlying MediaPipe graph. Call this exactly once
        per video, after the last frame — not per frame (see class docstring)."""
        self._holistic.close()

    def __enter__(self) -> HolisticFrameExtractor:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
