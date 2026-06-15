from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(ROOT / Path(os.getenv("MODEL_PATH", "models/face_detector.tflite")))
