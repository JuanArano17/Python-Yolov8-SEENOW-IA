# main.py
import time
import logging
import cv2
from detector import YOLODetector

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Se crea la instancia del detector utilizando YOLOv5 (modelo 's') y se procesa cada frame (detection_interval=0)
    detector = YOLODetector(camera_source=0, detection_interval=0)
    detection_thread = detector.start_detection()
    
    try:
        # Bucle principal: se encarga de mostrar los frames anotados en el hilo principal
        while detection_thread.is_alive():
            if not detector.frame_queue.empty():
                frame = detector.frame_queue.get()
                cv2.imshow("YOLOv5 Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Se presionó 'q'. Terminando detección.")
                detector.stop()
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logging.info("Interrupción por teclado. Terminando ejecución...")
        detector.stop()
    
    time.sleep(2)
    cv2.destroyAllWindows()
