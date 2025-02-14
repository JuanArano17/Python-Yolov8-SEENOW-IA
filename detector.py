# detector.py
import time
import threading
import logging
import cv2
import queue
from AI import IADetector

class YOLODetector:
    def __init__(self, camera_source=0, detection_interval=0):
        """
        Inicializa el detector:
          - camera_source: Fuente de video (por defecto la cámara 0).
          - detection_interval: Tiempo mínimo (en segundos) entre detecciones. Si es 0 se procesa cada frame.
        """
        self.camera_source = camera_source
        self.detection_interval = detection_interval
        self.detector = IADetector("yolov5s.pt")
        self.frame_queue = queue.Queue()  # Cola para pasar frames anotados al hilo principal
        self.running = True

    def detection_worker(self):
        logging.info("Iniciando detección en streaming con YOLOv5...")
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            logging.error("No se pudo abrir la cámara.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.error("No se pudo leer frame de la cámara.")
                break

            # Voltear la imagen horizontalmente para que se vea como espejo
            frame = cv2.flip(frame, 1)

            start_time = time.time()

            # Realizar la predicción sobre el frame
            results = self.detector.predict(frame)
            annotated_frame = frame.copy()
            try:
                # Iterar sobre las detecciones. Cada detección es un tensor con [x1, y1, x2, y2, conf, cls]
                for detection in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = detection
                    x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                    class_name = results.names[int(cls.item())]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error al procesar las detecciones: {e}")

            # Enviar el frame anotado a la cola para su visualización
            self.frame_queue.put(annotated_frame)

            if self.detection_interval != 0:
                detection_time = time.time() - start_time
                wait_time = max(self.detection_interval - detection_time, 0)
                logging.info(f"Esperando {wait_time:.2f} segundos para la siguiente detección.")
                time.sleep(wait_time)

        cap.release()
        logging.info("Terminando detection_worker.")
    
    def start_detection(self):
        """
        Inicia el proceso de detección en un hilo en segundo plano.
        """
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        return detection_thread

    def stop(self):
        self.running = False
