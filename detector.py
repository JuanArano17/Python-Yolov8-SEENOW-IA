import time
import threading
import logging
from AI import IADetector
from audio import AudioPlayer

class YOLOSpeechDetector:
    def __init__(self, camera_source="0", detection_interval=5, lang="en"):
        self.camera_source = camera_source
        self.detection_interval = detection_interval
        self.audio_player = AudioPlayer(lang)
        self.detector = IADetector("yolov8x.pt")
        self.last_announced = None
        self.announcement_lock = threading.Lock()
    
    def detection_worker(self):
        logging.info("Iniciando detección en streaming...")
        try:
            results = self.detector.predict(source=self.camera_source, stream=True, show=True)
        except Exception as e:
            logging.error(f"Error al iniciar la predicción: {e}")
            return

        for result in results:
            start_time = time.time()
            try:
                if result.boxes.shape[0] > 0:
                    names = [self.detector.model.names[int(cls)] for cls in result.boxes.cls]
                    logging.info(f"Objetos detectados: {names}")
                    
                    with self.announcement_lock:
                        if names == self.last_announced:
                            logging.debug("Anuncio repetido detectado; se omite.")
                        else:
                            self.last_announced = names
                            # En este ejemplo, se sintetiza de forma síncrona
                            text = ', '.join(names)
                            self.audio_player.speak(text)
            except Exception as e:
                logging.error(f"Error al procesar el frame: {e}")

            detection_time = time.time() - start_time
            wait_time = max(self.detection_interval - detection_time, 0)
            logging.info(f"Esperando {wait_time:.2f} segundos para la siguiente detección.")
            time.sleep(wait_time)
    
    def start_detection(self):
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        return detection_thread