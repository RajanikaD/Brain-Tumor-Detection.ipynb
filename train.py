import numpy as np
from sklearn.model_selection import train_test_split
from model import build_model
from preprocessing import load_data

IMG_WIDTH, IMG_HEIGHT = (240, 240)

augmented_yes = "data/yes"
augmented_no = "data/no"

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

model = build_model((IMG_WIDTH, IMG_HEIGHT, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=22, validation_data=(X_val, y_val))

model.save("brain_tumor_model.h5")