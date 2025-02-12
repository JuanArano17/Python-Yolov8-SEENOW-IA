# ia.py
from ultralytics import YOLO

class IADetector:
    def __init__(self, model_path="yolov8x.pt"):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise Exception(f"Error al cargar el modelo: {e}")
    
    def predict(self, source, stream=True, show=False):
        """
        Realiza la predicción utilizando el modelo YOLO.
        Se configura show=False para permitir la personalización de la visualización.
        """
        return self.model.predict(source=source, stream=stream, show=show)
