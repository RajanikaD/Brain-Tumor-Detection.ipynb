import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocessing import crop_brain_contour

model = load_model("brain_tumor_model.h5")

def predict(image_path):
    image = cv2.imread(image_path)
    image = crop_brain_contour(image)
    image = cv2.resize(image, (240, 240))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    return "Tumor" if prediction[0][0] > 0.5 else "No Tumor"

if __name__ == "__main__":
    print(predict("test.jpg"))