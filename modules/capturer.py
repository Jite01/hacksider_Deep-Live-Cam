from typing import Any
import cv2
import numpy as np
import modules.globals  # for color_correction toggle

def get_video_frame(video_path: str, frame_number: int = 0) -> Any:
    """
    Grab a single frame from any video file or RTSP URL.
    If modules.globals.color_correction is enabled, convert to RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    # Force MJPEG for correct colors, if needed
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if modules.globals.color_correction:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # clamp to valid range
    idx = max(0, min(total_frames - 1, frame_number))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    if modules.globals.color_correction:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_video_frame_total(video_path: str) -> int:
    """Return total frame count, or 0 if unavailable."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total
