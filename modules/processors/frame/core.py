import sys
import importlib
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.globals

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]

# RTSP-specific settings
RTSP_RECONNECT_DELAY = 2  # seconds to wait before reconnecting
RTSP_TIMEOUT = 5000  # milliseconds


def is_rtsp(path: str) -> bool:
    """Check if path is an RTSP stream URL"""
    return isinstance(path, str) and path.lower().startswith('rtsp://')


def open_video_capture(video_path: str):
    """Open video capture with proper settings for RTSP streams"""
    if is_rtsp(video_path):
        # For RTSP, use FFMPEG backend with buffer optimizations
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT)
    else:
        cap = cv2.VideoCapture(video_path)
    return cap


def extract_frames(video_path: str, temp_frame_paths: List[str], max_frames: int = None) -> float:
    """Extract frames from video source (file or RTSP)"""
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_path}")
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5  # Max failures before giving up

    # Create temp directory if it doesn't exist
    os.makedirs(modules.globals.temp_path, exist_ok=True)

    while True:
        ret, frame = cap.read()

        # Handle RTSP disconnections
        if not ret and is_rtsp(video_path):
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"RTSP stream disconnected. Attempting to reconnect...")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = open_video_capture(video_path)
                consecutive_failures = 0
                continue
            else:
                time.sleep(0.1)  # Short delay before retrying
                continue

        if not ret:
            break  # End of video or unrecoverable error

        consecutive_failures = 0  # Reset on successful frame

        frame_path = os.path.join(modules.globals.temp_path, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        temp_frame_paths.append(frame_path)

        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return fps


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                sys.exit()
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES


def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.globals.fp_ui.items():
        if state == True and frame_processor not in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
            modules.globals.frame_processors.append(frame_processor)
        if state == False:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                modules.globals.frame_processors.remove(frame_processor)
            except:
                pass


def multi_process_frame(source_path: str, temp_frame_paths: List[str],
                        process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures.append(future)
        for future in futures:
            future.result()


def process_video(source_path: str, frame_paths: list[str],
                  process_frames: Callable[[str, List[str], Any], None]) -> None:
    # If source is RTSP, use real-time processing
    if is_rtsp(source_path):
        print("Processing RTSP stream in real-time mode...")
        process_rtsp_stream(source_path, frame_paths, process_frames)
        return

    # For file-based videos
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True,
              bar_format=progress_bar_format) as progress:
        progress.set_postfix({'execution_providers': modules.globals.execution_providers,
                              'execution_threads': modules.globals.execution_threads,
                              'max_memory': modules.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)


def process_rtsp_stream(rtsp_url: str, frame_paths: list[str],
                        process_frames: Callable[[str, List[str], Any], None]) -> None:
    """Real-time processing for RTSP streams"""
    cap = open_video_capture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {rtsp_url}")
        return

    print("RTSP stream opened successfully. Starting real-time processing...")

    # Create temp directory if it doesn't exist
    os.makedirs(modules.globals.temp_path, exist_ok=True)

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed. Attempting to reconnect...")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = open_video_capture(rtsp_url)
                if not cap.isOpened():
                    print("Reconnection failed. Exiting.")
                    break
                continue

            # Save frame to temp path
            frame_path = os.path.join(modules.globals.temp_path, f"rtsp_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            # Process the frame
            process_frames(rtsp_url, [frame_path], None)

            # Clean up: remove processed frame
            try:
                os.remove(frame_path)
            except:
                pass

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:  # Update every second
                fps = frame_count / (current_time - last_time)
                print(f"Processing FPS: {fps:.2f}", end='\r')
                frame_count = 0
                last_time = current_time

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

    finally:
        cap.release()
        print("RTSP processing stopped")