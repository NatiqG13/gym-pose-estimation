"""
video_loader.py
By: Natiq Ghafoor

Video frame loader for webcam and video files (OpenCV).

Key detail:
- For video files, timestamps come from the video timecode (CAP_PROP_POS_MSEC),
  not wall-clock time. This keeps durations stable even if inference is slow.
- For webcam/live capture, timestamps fall back to wall-clock time.
"""

from __future__ import annotations

import time

import cv2


def frame_generator(cap):
    """
    Yields (frame, timestamp_seconds) from an OpenCV capture object.

    timestamp_seconds:
      - video file: derived from CAP_PROP_POS_MSEC
      - webcam/live: time.time()
    """
    # Video files generally have a known frame count; webcams typically do not.
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    use_video_time = bool(frame_count and frame_count > 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_video_time:
            ts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = float(ts_msec) / 1000.0 if ts_msec is not None else 0.0
        else:
            timestamp = time.time()

        yield frame, timestamp

    cap.release()


def VideoLoader(mode, source):
    """
    Creates a cv2.VideoCapture for either webcam or a file path.

    mode:
      - "webcam"
      - "video_file"
    """
    if mode == "webcam":
        return cv2.VideoCapture(0)
    if mode == "video_file":
        return cv2.VideoCapture(source)

    raise ValueError(f"Unsupported mode: {mode}. Must be 'webcam' or 'video_file'.")
