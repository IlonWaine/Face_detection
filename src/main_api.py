from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Response, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware # Додано імпорт

from detection_visualization import visualize, resize

from config import ROOT, MODEL_PATH

import cv2 as cv
import numpy as np
import base64

from detector import PoliInputFaceDetector, IMAGE

app = FastAPI(
    title="Vision Agent API",
    version="1.0.0",
    # lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Дозволяє запити з будь-яких доменів (для розробки)
    allow_credentials=False,
    allow_methods=["*"],  # Дозволяє всі методи (GET, POST і т.д.)
    allow_headers=["*"],  # Дозволяє всі заголовки
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


@app.post("/process-image")
async def detect_image_endpoint(file: UploadFile = File(...)):
    try:
        print("Отримано запит на обробку фото!") # Перевіряємо, чи доходить запит
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        capture = cv.imdecode(nparr, cv.IMREAD_COLOR)

        
        if capture is None:
            raise HTTPException(status_code=400, detail="Недійсний файл зображення")

        # --- ТУТ ПОЧИНАЄТЬСЯ ВАША ЛОГІКА ---
        print("Починаємо ресайз і детекцію...")
        frame = resize(capture)
        
        with PoliInputFaceDetector(MODEL_PATH, running_mode=IMAGE) as detector:
            detection_result = detector.detection(frame)
            count = len(detection_result.detections) if detection_result and detection_result.detections else 0
            print(count)
            drawen_img = visualize(frame, detection_result)
        # --- ТУТ ЗАКІНЧУЄТЬСЯ ВАША ЛОГІКА ---
            
        print("Кодуємо результат...")
        success, encoded_image = cv.imencode('.jpg', drawen_img)
        if not success:
            raise HTTPException(status_code=500, detail="Помилка кодування")
            
        response = Response(content=encoded_image.tobytes(), media_type="image/jpeg")
        response.headers["X-Detection-Count"] = str(count)
        return response
        
    except Exception as e:
        # ЦЕ ВРЯТУЄ НАС: виведе повну помилку в термінал замість того, щоб впасти
        raise HTTPException(status_code=500, detail=str(e))