from math import floor

import cv2 as cv
import numpy as np


MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)


def resize(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Proportionally resize image so the longest side does not exceed max_dim."""
    h, w = img.shape[:2]

    if max(h, w) <= max_dim:
        return img

    if w > h:
        r = max_dim / float(w)
        new_dim = (max_dim, int(h * r))
    else:
        r = max_dim / float(h)
        new_dim = (int(w * r), max_dim)

    return cv.resize(img, new_dim, interpolation=cv.INTER_AREA)


def normalize_to_pixel(
    x: float, y: float, width: int, height: int
) -> tuple[int, int]:
    """Convert normalized [0, 1] coordinates to pixel coordinates."""
    x_val = np.clip(x, 0.0, 1.0)
    y_val = np.clip(y, 0.0, 1.0)
    x_px = min(floor(x_val * width), width - 1)
    y_px = min(floor(y_val * height), height - 1)
    return x_px, y_px


def visualize(image: np.ndarray, detection_result) -> np.ndarray:
    """Draw bounding boxes, keypoints and labels for each detected face."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    if detection_result is None:
        return annotated_image

    for detection in detection_result.detections:
        # Bounding box
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Facial keypoints
        for keypoint in detection.keypoints:
            keypoint_px = normalize_to_pixel(keypoint.x, keypoint.y, width, height)
            cv.circle(annotated_image, keypoint_px, radius=2, color=(0, 255, 0), thickness=2)

        # Label and confidence score
        category = detection.categories[0]
        category_name = category.category_name or ""
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv.putText(
            annotated_image,
            result_text,
            text_location,
            cv.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return annotated_image
