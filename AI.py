# ia.py
import torch
from ultralytics import YOLO

def load_yolov5_model(model_path="yolov5s.pt"):
    """
    Loads YOLOv5 using torch.hub.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    except Exception as e:
        raise Exception(f"Error loading YOLOv5 model: {e}")
    return model

def load_yolov8_model(model_path="yolov8x.pt"):
    """
    Loads YOLOv8 using the ultralytics API.
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise Exception(f"Error loading YOLOv8 model: {e}")
    return model
