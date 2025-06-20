# import torch
import numpy as np
from pathlib import Path
import cv2
import os
import uuid
from ultralytics import YOLO  # For YOLOv8
import pathlib
from labels import LABELS  # <-- Import your labels list
import torch
pathlib.PosixPath = pathlib.WindowsPath  # Patch for Windows path issue

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === LOAD MODELS ===
yolov5_model = torch.hub.load('yolov5', 'custom', path='best-weights/bestYOLOv5s.pt', source='local')

try:
    yolov8_model = YOLO('best-weights/bestYOLOv8n.pt')
except Exception as e:
    yolov8_model = None
    print("YOLOv8 load failed:", e)

# === DETECTION FUNCTION ===

def detect_with_model(image_path, model_type='yolov5'):
    img0 = cv2.imread(image_path)
    assert img0 is not None, 'Image not found'

    result_label = "No Disease Detected"

    if model_type == 'yolov5':
        results = yolov5_model(image_path)
        preds = results.pandas().xyxy[0]

        for _, row in preds.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            conf = row['confidence']
            cls_id = int(row['class'])

            cls_name = LABELS[cls_id] if cls_id < len(LABELS) else f'class{cls_id}'
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            result_label = cls_name

    elif model_type == 'yolov8':
        if yolov8_model is None:
            raise ValueError("YOLOv8 model not loaded")

        results = yolov8_model(image_path)
        img0 = results[0].plot()

        if results[0].boxes:
            cls_index = int(results[0].boxes.cls[0].item())
            result_label = LABELS[cls_index] if cls_index < len(LABELS) else f'class{cls_index}'

    else:
        raise ValueError("Invalid model type. Use 'yolov5' or 'yolov8'.")

    # Save annotated image
    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join('static', result_filename)
    cv2.imwrite(result_path, img0)

    return result_path, result_label
