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
    
    detector = YOLODetector()
    detection_thread = detector.start_detection()
    
    try:
        while detection_thread.is_alive():
            if not detector.frame_queue.empty():
                frame = detector.frame_queue.get()
                cv2.imshow("YOLO Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Se presionó 'q'. Terminando detección.")
                detector.stop()  # Detener de forma controlada el worker
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logging.info("Interrupción por teclado. Terminando ejecución...")
        detector.stop()
    
    # Esperar un momento para asegurar que la cámara se libere
    time.sleep(2)
    cv2.destroyAllWindows()
