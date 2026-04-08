from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import numpy as np
import cv2
import tensorflow as tf
import pytesseract
from PIL import Image
import io

# ================= CREATE APP FIRST =================
app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODELS =================

# spaCy NLP
nlp = spacy.load("en_core_web_sm")

# CNN Model
model = tf.keras.models.load_model("cnn_model.h5")

# Labels (Sign MNIST → no J, Z)
labels = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S',
    'T','U','V','W','X','Y'
]

# OCR path (CHANGE if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= TEXT → SIGN =================

class TextInput(BaseModel):
    text: str

@app.post("/process-text")
def process_text(data: TextInput):
    doc = nlp(data.text.lower())

    words = []
    for token in doc:
        if not token.is_punct:
            words.append(token.text)

    return {"words": words}

# ================= OCR IMAGE → SIGN =================

@app.post("/ocr-to-sign")
async def ocr_to_sign(file: UploadFile):
    contents = await file.read()

    # Convert to OpenCV image
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 🔥 STEP 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 STEP 2: Remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 🔥 STEP 3: Threshold (make text clear)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 🔥 STEP 4: OCR config (IMPORTANT)
    custom_config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(thresh, config=custom_config)

    # spaCy cleaning
    doc = nlp(text.lower())

    words = []
    for token in doc:
        if not token.is_punct:
            words.append(token.text)

    return {
        "text": text.strip(),
        "words": words
    }

# ================= IMAGE → SIGN (CNN MODEL) =================

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(img)
    index = np.argmax(pred)

    return {"prediction": labels[index]}