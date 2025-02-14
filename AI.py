# ia.py
import torch

class IADetector:
    def __init__(self, model_path="yolov5s.pt"):
        """
        Carga el modelo YOLOv5 utilizando Torch Hub.
        En este ejemplo se utiliza la versión 's' (small), pero se puede cambiar por 'm', 'l', etc.
        """
        try:
            # Se descarga y carga el modelo YOLOv5s preentrenado
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        except Exception as e:
            raise Exception(f"Error al cargar el modelo YOLOv5: {e}")
    
    def predict(self, frame):
        """
        Realiza la detección sobre un frame.
        """
        results = self.model(frame)
        return results
