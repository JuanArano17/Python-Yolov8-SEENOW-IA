# main.py
import time
import logging
import cv2
import argparse
from detector import YOLODetectorV5, YOLODetectorV8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select YOLO version: v5 or v8")
    parser.add_argument("--version", type=str, choices=["v5", "v8"], default="v5",
                        help="Select YOLO version to use (default: v5)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Select the detector based on the argument
    if args.version == "v5":
        logging.info("Using YOLOv5 detector.")
        detector = YOLODetectorV5(camera_source=0, detection_interval=0)
    else:
        logging.info("Using YOLOv8 detector.")
        detector = YOLODetectorV8(camera_source=0, detection_interval=0)

    detection_thread = detector.start_detection()

    try:
        while detection_thread.is_alive():
            if not detector.frame_queue.empty():
                frame = detector.frame_queue.get()
                # Display the annotated frame
                window_title = f"YOLO {args.version.upper()} Detector"
                cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Pressed 'q'. Stopping detection.")
                detector.stop()
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logging.info("Keyboard interruption. Stopping execution...")
        detector.stop()

    # Wait a moment to ensure resources are freed
    time.sleep(2)
    cv2.destroyAllWindows()
