import cv2
import numpy as np
from typing import Optional, Tuple, Callable, Union
import platform
import threading
import logging

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# Set up logging
logger = logging.getLogger(__name__)

def is_rtsp(path: Union[int, str]) -> bool:
    """Check if path is an RTSP stream URL"""
    return isinstance(path, str) and path.lower().startswith("rtsp://")

class VideoCapturer:
    def __init__(self, device_index: Union[int, str]):
        """
        device_index: either an int (local camera) or str (rtsp:// URL)
        """
        self.device_index = device_index
        self.is_rtsp = is_rtsp(device_index)
        self.frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._current_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.cap: Optional[cv2.VideoCapture] = None

        # On Windows and not RTSP, verify device index
        if platform.system() == "Windows" and not self.is_rtsp:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if isinstance(device_index, int) and device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture"""
        try:
            # ---- RTSP stream ----
            if self.is_rtsp:
                logger.info(f"Opening RTSP stream: {self.device_index}")
                # Use FFMPEG backend for RTSP
                self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open RTSP stream: {self.device_index}")
                # Minimal buffer for low latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Frame rate hint
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                # Try to read native resolution
                w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if w <= 0 or h <= 0:
                    # fallback to user‑specified
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                logger.info("RTSP stream opened successfully")

            # ---- Windows camera ----
            elif platform.system() == "Windows":
                # Try various backends
                for dev, backend in [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_ANY),
                    (-1,                cv2.CAP_ANY),
                    (0,                 cv2.CAP_ANY),
                ]:
                    cap = cv2.VideoCapture(dev, backend)
                    if cap.isOpened():
                        self.cap = cap
                        break
                    cap.release()
                if not self.cap:
                    raise RuntimeError("Failed to open camera on Windows")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS,           fps)
                logger.info("Windows camera opened successfully")

            # ---- Unix / macOS camera ----
            else:
                self.cap = cv2.VideoCapture(self.device_index)
                if not self.cap.isOpened():
                    raise RuntimeError("Failed to open camera")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS,           fps)
                logger.info("Camera opened successfully (Unix/Mac)")

            self.is_running = True
            return True

        except Exception as e:
            logger.error(f"[VideoCapturer] start() error: {e}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source, auto‑reconnect RTSP on failure"""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret and self.is_rtsp:
            logger.warning("RTSP disconnected, attempting reconnect...")
            self.release()
            if self.start():
                ret, frame = self.cap.read()

        if ret:
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        return False, None

    def release(self) -> None:
        """Stop capture and free resources"""
        if self.cap:
            self.cap.release()
        self.is_running = False
        self.cap = None
        logger.info("VideoCapturer released")

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a frame‑ready callback"""
        self.frame_callback = callback
