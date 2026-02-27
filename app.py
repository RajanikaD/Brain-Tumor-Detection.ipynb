# app.py

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from preprocessing import crop_brain_contour

app = FastAPI(title="Brain Tumor Detection API")

model = load_model("brain_tumor_model.h5")

@app.get("/")
def root():
    return {"message": "Brain Tumor Detection API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Convert bytes to image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess
    image = crop_brain_contour(image)
    image = cv2.resize(image, (240, 240))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)[0][0]
    
    result = "Tumor" if prediction > 0.5 else "No Tumor"

    return {
        "prediction": result,
        "confidence": float(prediction)
    }