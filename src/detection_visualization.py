import numpy as np
from math import floor
import cv2 as cv


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  


def resize(img, max_dim=800):
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


def normalize_to_pixel(x,y,width,height) -> None | tuple[int, int]:
    x_val = np.clip(x, 0.0, 1.0)
    y_val = np.clip(y, 0.0, 1.0)
    
    x_px = min(floor(x_val * width), width - 1)
    y_px = min(floor(y_val * height), height - 1)
    return x_px, y_px



def visualize(image, detection_result) -> np.ndarray:
  
  annotated_image = image.copy()
  height, width, _ = image.shape
  if detection_result is not None:

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = normalize_to_pixel(keypoint.x, keypoint.y, width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv.circle(annotated_image, keypoint_px, radius, color, thickness)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv.putText(annotated_image, result_text, text_location, cv.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return annotated_image
  else: 
    return image
  

