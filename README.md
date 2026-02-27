# 🧠 Brain Tumor Detection API (Containerized ML Inference Service)

An end-to-end medical image classification system for detecting brain tumors from MRI scans.  
The project includes a custom CNN architecture, contour-based preprocessing, and a production-style FastAPI deployment wrapped in a Docker container.

---

## 🚀 Overview

This project implements a binary classification pipeline (Tumor / No Tumor) using TensorFlow and OpenCV-based preprocessing.  

Unlike a notebook-based prototype, this version modularizes training, preprocessing, and inference, and exposes the trained model as a REST API suitable for deployment.

---

## 🧩 System Architecture

**Pipeline:**

MRI Image → Contour-Based Cropping (OpenCV) → Resize & Normalize → CNN Inference → REST API Response

### Components

- Custom CNN model for binary classification
- Contour-based brain region isolation
- TensorFlow model persistence (`.h5`)
- FastAPI inference service
- Swagger-based interactive API docs
- Docker containerization for reproducible deployment

---

## 🧠 Model Architecture

- Convolution + pooling layers
- ReLU activation
- Binary classification with sigmoid output
- Optimized using Adam + Binary Cross-Entropy

Validation Accuracy: ~84% on held-out MRI dataset

---

## 📁 Project Structure
brain-tumor-detection-v2/
│

├── train.py # Training pipeline

├── model.py # CNN architecture

├── preprocessing.py # Contour-based image processing

├── inference.py # Standalone inference script

├── app.py # FastAPI deployment wrapper

├── brain_tumor_model.h5 # Trained model

├── Dockerfile # Container specification

└── requirements.txt


---

🔬 Engineering Highlights
Converted experimental notebook workflow into modular production-ready Python modules.
Implemented contour-based preprocessing to isolate relevant anatomical regions before inference.
Deployed trained TensorFlow model as a REST microservice using FastAPI.
Containerized the full inference stack using Docker for environment reproducibility.
Resolved Linux-level OpenCV dependencies inside container runtime.

---

📌 Key Technologies
Python
TensorFlow / Keras
OpenCV
FastAPI
Docker
Uvicorn

---

🧪 Future Improvements
ONNX model export for lightweight inference
GPU-enabled container runtime
Model performance monitoring integration
CI/CD automation for container builds

---

📜 Disclaimer
This project is for educational and research demonstration purposes and is not intended for clinical diagnosis.

---

## Run with Docker
docker build -t brain-tumor-api .
docker run -p 8000:8000 brain-tumor-api
