
import cv2 as cv
import time


from detector import PoliInputFaceDetector, IMAGE, VIDEO, STREAM
from detection_visualization import visualize, resize
import config

img = r'D:\ML_Projects\Face_feature_detection\data\20210823_122850.jpg'
MODEL_PATH = config.MODEL_PATH
video_input = r'D:\ML_Projects\Face_feature_detection\data\20210418 032145.mp4'


def stream_detection():

    capture = cv.VideoCapture(0)

    with PoliInputFaceDetector(MODEL_PATH,running_mode=STREAM ) as detector:

        p_time = 0

        while True:
            is_True, frame = capture.read()

            if not is_True:
                break

            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            detection_result = detector.detection(frame)
            new_frame = visualize(frame,detector.latest_result)

            cv.putText( new_frame,f'FPs: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN, 5, (0,255,0), 2 )
            cv.imshow('video',new_frame)
            
            if cv.waitKey(20) == ord('d'):
                print(f"Знайдено облич: {detection_result}")
                print('d worked') 
                break
    capture.release()

def video_detection(input):

    capture = cv.VideoCapture(input)

    with PoliInputFaceDetector(MODEL_PATH,running_mode=VIDEO ) as detector:
        p_time = 0
        while True:
            is_True, raw_frame = capture.read()
            frame = resize(raw_frame)
            if not is_True:
                break

            c_time = time.time()
            frame_timestamp_ms = int(capture.get(cv.CAP_PROP_POS_MSEC))
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            detection_result = detector.detection(frame,frame_timestamp_ms)
            draw_face = visualize(frame,detection_result)
            cv.putText( draw_face,f'FPs: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN, 5, (0,255,0), 2 )
            cv.imshow('video',draw_face)
            cv.waitKey(20)
            if cv.getWindowProperty('video', cv.WND_PROP_VISIBLE) < 1:
                print(f"Window '{'video'}' was closed manually. Breaking loop.")
                break

    capture.release()

def image_detection(img):
    with PoliInputFaceDetector(MODEL_PATH,running_mode=IMAGE ) as detector:
        capture = cv.imread(img)
        frame = resize(capture)
        detection_result = detector.detection(frame)
        draw_face = visualize(frame,detection_result)
        drawen_img = visualize(frame,draw_face)
        cv.imshow('img',drawen_img)
        cv.waitKey(0) 



# final_image = stream_detection()

cv.destroyAllWindows()            
