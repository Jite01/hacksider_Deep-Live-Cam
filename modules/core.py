import os
import sys
import subprocess  # Added missing import

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
import cv2
import time

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension, is_image, is_video,
    detect_fps, create_video, extract_frames,
    get_temp_frame_paths, restore_audio, create_temp,
    move_temp, clean_temp, normalize_output_path
)

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select a source image', dest='source_path')
    program.add_argument('-t', '--target', help='select a target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor',
                         default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true',
                         default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true',
                         default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true',
                         default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264',
                         choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int,
                         default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int,
                         default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'],
                         choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int,
                         default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version',
                         version=f'{modules.metadata.name} {modules.metadata.version}')

    # Deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path,
                                                        args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads

    # Face enhancer toggle
    modules.globals.fp_ui['face_enhancer'] = 'face_enhancer' in args.frame_processor
    modules.globals.nsfw = False

    # Handle deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path,
                                                            args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider rocm instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [ep.replace('ExecutionProvider', '').lower() for ep in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, short in zip(onnxruntime.get_available_providers(),
                                                encode_execution_providers(onnxruntime.get_available_providers()))
            if any(ep in short for ep in execution_providers)]


def suggest_max_memory() -> int:
    return 4 if platform.system().lower() == 'darwin' else 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers or 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)


def is_stream(path: str) -> bool:
    return path.startswith('rtsp://') or path.startswith('rtmp://')


def process_stream(source_path: str, stream_url: str, output_url: str) -> None:
    """Process RTSP stream with advanced diagnostics and fallbacks"""
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture()

    # Configure with minimal buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try different transport protocols
    transport_options = [
        ('udp', stream_url),  # Default UDP
        ('tcp', stream_url + '?tcp'),
        ('udp_mcast', stream_url + '?udp_multicast=1'),
        ('http', stream_url.replace('rtsp://', 'http://'))
    ]

    connected = False
    for transport_name, transport_url in transport_options:
        update_status(f'Trying {transport_name} transport: {transport_url}')
        if cap.open(transport_url, cv2.CAP_FFMPEG):
            update_status(f'Connected via {transport_name} transport')
            connected = True
            break
        else:
            update_status(f'Failed with {transport_name} transport')

    if not connected:
        update_status('All transport protocols failed - check stream URL and network')
        # Try direct URL without FFMPEG backend
        update_status('Trying direct open without FFMPEG backend')
        if cap.open(stream_url):
            update_status('Connected without FFMPEG backend')
            connected = True

    if not connected:
        update_status('Could not connect to stream - exiting')
        return

    # Get stream properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    update_status(f'Stream properties: {width}x{height} @ {fps:.2f} FPS')

    # Initialize processors
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    for processor in frame_processors:
        if not processor.pre_start():
            cap.release()
            return

    # Configure RTMP output
    writer = None
    if output_url.startswith('rtmp://'):
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'flv',
            output_url
        ]
        update_status(f'Starting FFmpeg output: {" ".join(command)}')
        writer = subprocess.Popen(command, stdin=subprocess.PIPE)

    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    max_retries = 10
    retry_count = 0

    try:
        while retry_count < max_retries:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                update_status(f'Frame read failed (attempt {retry_count}/{max_retries})')
                time.sleep(0.1)
                continue

            retry_count = 0  # Reset on success

            # Process frame
            processed_frame = frame.copy()
            for processor in frame_processors:
                try:
                    processed_frame = processor.process_frame(source_path, processed_frame)
                except Exception as e:
                    update_status(f'Processor error: {str(e)}')
                    continue

            # Output
            if writer:
                try:
                    writer.stdin.write(processed_frame.tobytes())
                except Exception as e:
                    update_status(f'RTMP write error: {str(e)}')
                    break

            # Log progress
            frame_count += 1
            if time.time() - last_log_time >= 2:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                update_status(f'Frames: {frame_count} | FPS: {current_fps:.1f} | Retries: {retry_count}')
                last_log_time = time.time()

    except KeyboardInterrupt:
        update_status('Stopped by user')
    except Exception as e:
        update_status(f'Error: {str(e)}')
    finally:
        cap.release()
        if writer:
            writer.stdin.close()
            writer.wait()
        update_status(f'Stream processing ended | Total frames: {frame_count}')

def start() -> None:
    for processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not processor.pre_start():
            return

    # Process image to image
    if has_image_extension(modules.globals.target_path):
        if not modules.globals.nsfw:
            from modules.predicter import predict_image
            if predict_image(modules.globals.target_path):
                destroy()
        shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        for processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', processor.NAME)
            processor.process_image(modules.globals.source_path, modules.globals.output_path,
                                    modules.globals.output_path)
            release_resources()
        update_status(
            'Processing to image succeed!' if is_image(modules.globals.target_path) else 'Processing to image failed!')
        return

    # Process streams differently
    if is_stream(modules.globals.target_path):
        if not modules.globals.nsfw:
            # Skip NSFW check for streams since it's real-time
            pass

        update_status('Processing live stream...')
        process_stream(modules.globals.source_path, modules.globals.target_path, modules.globals.output_path)
        update_status('Stream processing completed')
        return

    # Process regular videos
    if not modules.globals.nsfw:
        from modules.predicter import predict_video
        if predict_video(modules.globals.target_path):
            destroy()

    update_status('Creating temp resources...')
    create_temp(modules.globals.target_path)

    update_status('Extracting frames...')
    extract_frames(modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    for processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', processor.NAME)
        processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()

    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path)

    if modules.globals.keep_audio:
        update_status(
            'Restoring audio...' if modules.globals.keep_fps else 'Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)

    clean_temp(modules.globals.target_path)
    update_status(
        'Processing to video succeed!' if is_video(modules.globals.target_path) else 'Processing to video failed!')


def destroy() -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()