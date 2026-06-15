import base64

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from Face_detection.src.config import MODEL_PATH, ROOT
from Face_detection.src.detection_visualization import resize, visualize
from Face_detection.src.detector import IMAGE, PoliInputFaceDetector
from Face_detection.src.logger import logger


app = FastAPI(title="Vision Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_html_response(file_name: str) -> HTMLResponse:
    with open(ROOT / "templates" / file_name, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/", response_class=HTMLResponse)
async def home_page() -> HTMLResponse:
    return get_html_response("index.html")


@app.websocket("/api/v1/stream-detect")
async def websocket_stream_detect(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connection established")

    with PoliInputFaceDetector(MODEL_PATH, running_mode=IMAGE) as detector:
        try:
            while True:
                # Receive base64-encoded frame from the client
                base64_data = await websocket.receive_text()
                image_bytes = base64.b64decode(base64_data)

                # Decode bytes into an OpenCV frame
                np_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv.imdecode(np_array, cv.IMREAD_COLOR)

                if frame is None:
                    logger.warning("Received an invalid frame, skipping")
                    continue

                detection_result = detector.detection(frame)

                # Serialize detections to JSON-compatible format
                faces_list = []
                if detection_result and detection_result.detections:
                    for detection in detection_result.detections:
                        bbox = detection.bounding_box
                        faces_list.append({
                            "score": detection.categories[0].score,
                            "bounding_box": {
                                "xmin": int(bbox.origin_x),
                                "ymin": int(bbox.origin_y),
                                "width": int(bbox.width),
                                "height": int(bbox.height),
                            },
                        })

                await websocket.send_json({"detections": faces_list})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")


@app.post("/process-image")
async def detect_image_endpoint(file: UploadFile = File(...)) -> Response:
    logger.info("Image processing request received: %s", file.filename)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        capture = cv.imdecode(nparr, cv.IMREAD_COLOR)

        if capture is None:
            logger.warning("Invalid image file: %s", file.filename)
            raise HTTPException(status_code=400, detail="Invalid image file")

        frame = resize(capture)

        with PoliInputFaceDetector(MODEL_PATH, running_mode=IMAGE) as detector:
            detection_result = detector.detection(frame)
            count = len(detection_result.detections) if detection_result and detection_result.detections else 0
            logger.info("Detection complete: %d face(s) found", count)
            annotated_img = visualize(frame, detection_result)

        success, encoded_image = cv.imencode(".jpg", annotated_img)
        if not success:
            logger.error("Failed to encode result image")
            raise HTTPException(status_code=500, detail="Image encoding failed")

        response = Response(content=encoded_image.tobytes(), media_type="image/jpeg")
        response.headers["X-Detection-Count"] = str(count)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during image processing")
        raise HTTPException(status_code=500, detail=str(e))
