import time
import logging
from detector import YOLOSpeechDetector

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    detector = YOLOSpeechDetector()
    detection_thread = detector.start_detection()

    try:
        while detection_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Interrupción por teclado. Terminando ejecución...")
