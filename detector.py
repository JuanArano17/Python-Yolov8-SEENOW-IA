# detector.py
import time
import threading
import logging
import cv2  # OpenCV para la visualización y anotación
import queue
from AI import IADetector

class YOLODetector:
    def __init__(self, camera_source=0, detection_interval=1):
        """
        Inicializa el detector:
         - camera_source: Fuente de video (por defecto la cámara 0 como entero).
         - detection_interval: Tiempo mínimo (en segundos) entre cada procesamiento.
        """
        self.camera_source = camera_source
        self.detection_interval = detection_interval
        self.detector = IADetector("yolov8x.pt")
        self.frame_queue = queue.Queue()  # Cola para pasar frames al hilo principal
        self.running = True

    def detection_worker(self):
        logging.info("Iniciando detección en streaming...")
        try:
            # Se utiliza show=False para procesar la imagen y agregar anotaciones personalizadas
            results = self.detector.predict(source=self.camera_source, stream=True, show=False)
        except Exception as e:
            logging.error(f"Error al iniciar la predicción: {e}")
            return

        for result in results:
            if not self.running:
                break
            start_time = time.time()
            
            # Obtener la imagen del frame (puede venir en 'orig_img' o en 'imgs')
            if hasattr(result, 'orig_img'):
                frame = result.orig_img
            elif hasattr(result, 'imgs'):
                frame = result.imgs[0]
            else:
                logging.error("No se pudo obtener la imagen del frame.")
                continue

            # Hacer una copia para las anotaciones
            annotated_frame = frame.copy()
            
            try:
                if result.boxes.shape[0] > 0:
                    # Iterar sobre cada detección para extraer las coordenadas y la clase
                    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        # Convertir las coordenadas a enteros
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        class_name = self.detector.model.names[int(cls)]
                        # Dibujar el rectángulo y colocar el nombre
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error al procesar el frame: {e}")

            # Enviar el frame anotado a la cola para mostrarlo en el hilo principal
            self.frame_queue.put(annotated_frame)

            detection_time = time.time() - start_time
            wait_time = max(self.detection_interval - detection_time, 0)
            logging.info(f"Esperando {wait_time:.2f} segundos para la siguiente detección.")
            time.sleep(wait_time)
        
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
