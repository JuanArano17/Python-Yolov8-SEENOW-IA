# main.py
import time
import logging
import cv2
from detector import YOLODetector

if __name__ == '__main__':
    # Configuración del logging para salida informativa
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Crear la instancia del detector (usando 0 como fuente de cámara)
    detector = YOLODetector(camera_source=0)
    detection_thread = detector.start_detection()
    
    try:
        # Bucle principal: se encarga de mostrar los frames en el hilo principal
        while detection_thread.is_alive():
            # Verificar si hay un frame disponible en la cola
            if not detector.frame_queue.empty():
                frame = detector.frame_queue.get()
                cv2.imshow("YOLO Detector", frame)
            
            # Permite salir presionando la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Se presionó 'q'. Terminando detección.")
                detector.stop()
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logging.info("Interrupción por teclado. Terminando ejecución...")
        detector.stop()
    
    cv2.destroyAllWindows()
