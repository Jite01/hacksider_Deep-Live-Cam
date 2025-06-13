import sys
import importlib
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules.globals

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]

# RTSP-specific configurations
RTSP_RECONNECT_DELAY = 2  # seconds
RTSP_MAX_RETRIES = 5
RTSP_TIMEOUT = 5000  # milliseconds


def is_rtsp(path: str) -> bool:
    """Check if path is an RTSP stream URL"""
    return isinstance(path, str) and path.lower().startswith('rtsp://')


def open_rtsp_stream(rtsp_url: str) -> cv2.VideoCapture:
    """Open RTSP stream with optimized parameters"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def extract_frames(video_path: str, temp_frame_paths: List[str], max_frames: int = None) -> float:
    """Enhanced frame extraction supporting both files and RTSP streams"""
    if is_rtsp(video_path):
        # RTSP streams are handled in real-time, no frame extraction needed
        return 30.0  # Default FPS for live streams

    # Standard video file processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {video_path}")
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = 0
    os.makedirs(modules.globals.temp_path, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(modules.globals.temp_path, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        temp_frame_paths.append(frame_path)

        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return fps


def process_rtsp_stream(rtsp_url: str, process_frame_fn: Callable) -> None:
    """Real-time RTSP stream processing"""
    retries = 0
    while retries < RTSP_MAX_RETRIES:
        cap = open_rtsp_stream(rtsp_url)
        if not cap.isOpened():
            print(f"[RTSP] Connection failed (attempt {retries + 1}/{RTSP_MAX_RETRIES})")
            retries += 1
            time.sleep(RTSP_RECONNECT_DELAY)
            continue

        print(f"[RTSP] Stream connected: {rtsp_url}")
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[RTSP] Frame read error")
                    break

                # Process frame directly without saving
                processed_frame = process_frame_fn(rtsp_url, frame)

                # Output handling would go here
                frame_count += 1

                # Calculate FPS every second
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    print(f"[RTSP] Processing FPS: {fps:.1f}", end='\r')
                    frame_count = 0
                    start_time = time.time()

        except Exception as e:
            print(f"[RTSP] Error: {str(e)}")

        finally:
            cap.release()
            print("\n[RTSP] Stream disconnected")
            retries += 1
            time.sleep(RTSP_RECONNECT_DELAY)

    print("[RTSP] Max retries reached, giving up")


# ... [rest of the existing functions remain unchanged] ...

def process_video(source_path: str, frame_paths: list[str], process_frames: Callable) -> None:
    """Modified to handle both file and RTSP processing"""
    if is_rtsp(source_path):
        process_rtsp_stream(source_path, lambda path, frame: process_frames(path, [frame], None))
        return

    # Original file-based processing
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True,
              bar_format=progress_bar_format) as progress:
        progress.set_postfix({
            'execution_providers': modules.globals.execution_providers,
            'execution_threads': modules.globals.execution_threads,
            'max_memory': modules.globals.max_memory
        })
        multi_process_frame(source_path, frame_paths, process_frames, progress)