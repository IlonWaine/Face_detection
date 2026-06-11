import cv2 as cv
import time
import numpy as np
from typing import Optional, List, Tuple

from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

img = r'C:\Users\ilonw\Self_study_ML\Face_detection\data\20210227_163953.jpg'
video_input = r'C:\Users\ilonw\Self_study_ML\Face_detection\data\20210418 032145.mp4'


STREAM = vision.RunningMode.LIVE_STREAM
VIDEO = vision.RunningMode.VIDEO
IMAGE  = vision.RunningMode.IMAGE

face_mesh = vision.FaceLandmarker

class PoliInputFaceDetector:
    def __init__(self, model: str, running_mode: vision.RunningMode) -> None:
        self.running_mode = running_mode
        self.model = str(model)
        self.latest_result = None
        self.detection_confidense = 0.7
        base_options = py.BaseOptions(model_asset_path = self.model)

        if self.running_mode == STREAM:
            detector_options = vision.FaceDetectorOptions(
                base_options = base_options,
                running_mode = self.running_mode,
                min_detection_confidence = self.detection_confidense,
                result_callback  = self._stream_callback 
            )
        else:
            detector_options = vision.FaceDetectorOptions(
                base_options = base_options,
                running_mode = self.running_mode,
                min_detection_confidence = self.detection_confidense,
            )

        self.detector = vision.FaceDetector.create_from_options(detector_options)


    def _stream_callback(self,result: vision.FaceDetectorResult, output_image: Image, timestamp_ms: int) -> None:
            if result.detections:
                self.latest_result = result
            else:
                self.latest_result = result
                print('There is no detection ')


    def detection(self, frame: np.ndarray, timestamp_ms: int = None) -> vision.FaceDetectorResult:
        
        rbg_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        mp_frame = Image(image_format=ImageFormat.SRGB, data=rbg_frame)

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        if self.running_mode == vision.RunningMode.IMAGE:
            return self.detector.detect(mp_frame)
            
        elif self.running_mode == vision.RunningMode.VIDEO:
            return self.detector.detect_for_video(mp_frame, timestamp_ms)
            
        elif self.running_mode == vision.RunningMode.LIVE_STREAM:
            self.detector.detect_async(mp_frame, timestamp_ms)
            


    def close(self):
        self.detector.close()


    def __enter__(self): 
        return self


    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.close()



   
