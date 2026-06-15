import time

import cv2 as cv
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision

from logger import logger


STREAM = vision.RunningMode.LIVE_STREAM
VIDEO = vision.RunningMode.VIDEO
IMAGE = vision.RunningMode.IMAGE


class PoliInputFaceDetector:
    def __init__(self, model: str, running_mode: vision.RunningMode) -> None:
        self.running_mode = running_mode
        self.model = str(model)
        self.latest_result = None
        self.detection_confidence = 0.7

        base_options = py.BaseOptions(model_asset_path=self.model)

        if self.running_mode == STREAM:
            detector_options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=self.running_mode,
                min_detection_confidence=self.detection_confidence,
                result_callback=self._stream_callback,
            )
        else:
            detector_options = vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=self.running_mode,
                min_detection_confidence=self.detection_confidence,
            )

        self.detector = vision.FaceDetector.create_from_options(detector_options)
        logger.debug("FaceDetector initialized (mode=%s)", self.running_mode)

    def _stream_callback(
        self,
        result: vision.FaceDetectorResult,
        output_image: Image,
        timestamp_ms: int,
    ) -> None:
        self.latest_result = result
        if not result.detections:
            logger.debug("Stream callback: no faces detected at %d ms", timestamp_ms)

    def detection(
        self, frame: np.ndarray, timestamp_ms: int = None
    ) -> vision.FaceDetectorResult:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_frame = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        if self.running_mode == IMAGE:
            result = self.detector.detect(mp_frame)
            logger.debug("Image detection: %d face(s) found", len(result.detections) if result.detections else 0)
            return result

        if self.running_mode == VIDEO:
            result = self.detector.detect_for_video(mp_frame, timestamp_ms)
            logger.debug("Video detection at %d ms: %d face(s) found", timestamp_ms, len(result.detections) if result.detections else 0)
            return result

        if self.running_mode == STREAM:
            self.detector.detect_async(mp_frame, timestamp_ms)

    def close(self) -> None:
        self.detector.close()
        logger.debug("FaceDetector closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
