import time

import cv2 as cv

import config
from detection_visualization import resize, visualize
from detector import IMAGE, STREAM, VIDEO, PoliInputFaceDetector
from logger import logger


MODEL_PATH = config.MODEL_PATH


def stream_detection() -> None:
    """Run real-time face detection from the default webcam."""
    capture = cv.VideoCapture(0)
    p_time = 0

    with PoliInputFaceDetector(MODEL_PATH, running_mode=STREAM) as detector:
        while True:
            is_true, frame = capture.read()
            if not is_true:
                logger.warning("Failed to read frame from webcam")
                break

            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            detector.detection(frame)
            new_frame = visualize(frame, detector.latest_result)
            cv.putText(new_frame, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
            cv.imshow("video", new_frame)

            if cv.waitKey(20) == ord("d"):
                logger.info("Stream stopped by user")
                break

    capture.release()


def video_detection(input_path: str) -> None:
    """Run face detection on a video file."""
    capture = cv.VideoCapture(input_path)
    p_time = 0

    with PoliInputFaceDetector(MODEL_PATH, running_mode=VIDEO) as detector:
        while True:
            is_true, raw_frame = capture.read()
            if not is_true:
                logger.info("End of video file reached")
                break

            frame = resize(raw_frame)
            c_time = time.time()
            frame_timestamp_ms = int(capture.get(cv.CAP_PROP_POS_MSEC))
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            detection_result = detector.detection(frame, frame_timestamp_ms)
            draw_face = visualize(frame, detection_result)
            cv.putText(draw_face, f"FPS: {int(fps)}", (20, 70), cv.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
            cv.imshow("video", draw_face)
            cv.waitKey(20)

            if cv.getWindowProperty("video", cv.WND_PROP_VISIBLE) < 1:
                logger.info("Video window closed by user")
                break

    capture.release()


def image_detection(img_path: str) -> None:
    """Run face detection on a single image file."""
    with PoliInputFaceDetector(MODEL_PATH, running_mode=IMAGE) as detector:
        capture = cv.imread(img_path)
        if capture is None:
            logger.error("Failed to load image: %s", img_path)
            return

        frame = resize(capture)
        detection_result = detector.detection(frame)
        annotated = visualize(frame, detection_result)
        cv.imshow("img", annotated)
        cv.waitKey(0)


cv.destroyAllWindows()
