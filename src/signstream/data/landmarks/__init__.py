"""MediaPipe Holistic landmark extraction and the content-addressed ``.npz`` cache.

Submodules:
    ``schema``: frozen landmark-layout constants — ``[543, 3]`` per frame
        (33 pose + 468 face + 21 per hand), slice offsets, and the
        ``EXTRACTOR_VERSION`` cache key component.
    ``extractor``: single-frame extraction; the only module allowed to
        import mediapipe.
    ``video_processor``: video-file and frame-folder input ->
        ``[T, 543, 3]`` arrays; the only module allowed to import cv2.
    ``cache_writer``: atomic, resumable, content-addressed cache I/O.

Nothing is re-exported here on purpose: importing this package must stay
cheap and dependency-free. ``extractor``, ``video_processor``, and
``cache_writer`` require the ``full`` extra (mediapipe/opencv); ``schema``
needs only numpy. The driver script is ``scripts/extract_landmarks.py``.
"""
