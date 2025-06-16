import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm
import cv2
import os

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

def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                print(f"Frame processor {frame_processor} missing required method: {method_name}")
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
        if state is True and frame_processor not in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
            modules.globals.frame_processors.append(frame_processor)
        elif state is False:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                modules.globals.frame_processors.remove(frame_processor)
            except:
                pass

def extract_frames_from_stream(video_path: str, temp_frame_paths: List[str], max_frames: int = None) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open stream: {video_path}")
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
    modules.globals.fps = fps
    return fps

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures.append(future)
        for future in futures:
            future.result()

def process_video(source_path: str, frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix({
            'execution_providers': modules.globals.execution_providers,
            'execution_threads': modules.globals.execution_threads,
            'max_memory': modules.globals.max_memory
        })
        multi_process_frame(source_path, frame_paths, process_frames, progress)
