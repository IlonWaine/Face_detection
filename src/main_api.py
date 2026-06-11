from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


from config import ROOT, MODEL_PATH

import cv2 as cv
import numpy as np
import base64

from detector import PoliInputFaceDetector, IMAGE

# Використовуємо менеджер життєвого циклу додатку для безпечного запуску та зупинки камери
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Дії під час старту сервера
#     video_stream_service.start_stream()
#     yield
#     # Дії під час вимкнення сервера
#     video_stream_service.stop_stream()

app = FastAPI(
    title="Vision Agent API",
    version="1.0.0",
    # lifespan=lifespan
)

def get_html_response(file_name: str) -> HTMLResponse:
    with open(ROOT / 'templates' / file_name, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/", response_class=HTMLResponse)
async def home_page():
    return get_html_response("index.html")

@app.websocket("/api/v1/stream-detect")
async def websocket_stream_detect(websocket: WebSocket):
    # Приймаємо з'єднання від браузера
    await websocket.accept()
    
    # Ініціалізуємо ваш менеджер контексту детектора
    with PoliInputFaceDetector(MODEL_PATH, running_mode=IMAGE) as detector:
        try:
            while True:
                # 1. Отримуємо Base64 рядок зображення від клієнта  
                base64_data = await websocket.receive_text()
                
                # 2. Декодуємо рядок назад у бінарні байти зображення
                image_bytes = base64.b64decode(base64_data)
                
                # 3. Перетворюємо байти в масив NumPy для OpenCV
                np_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv.imdecode(np_array, cv.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # 4. Викликаємо ваш метод обробки зображення
                detection_result = detector.detection(frame)
                
                # 5. Парсимо результат об'єкта MediaPipe у чистий JSON формат
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
                                "height": int(bbox.height)
                            }
                        })
                
                # 6. Повертаємо координати у браузер клієнта
                await websocket.send_json({"detections": faces_list})
                
        except WebSocketDisconnect:
            print("Користувач закрив сторінку або вимкнув камеру (WebSocket розірвано).")


