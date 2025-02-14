# detector.py
import time
import threading
import logging
import cv2
import queue
from ia import load_yolov5_model, load_yolov8_model

class YOLODetectorV5:
    def __init__(self, camera_source=0, detection_interval=0):
        """
        Initializes the YOLOv5 detector.
         - camera_source: video source (default camera 0).
         - detection_interval: minimum time (in seconds) between processing frames.
           If 0, every frame is processed.
        """
        self.camera_source = camera_source
        self.detection_interval = detection_interval
        self.model = load_yolov5_model("yolov5s.pt")
        self.frame_queue = queue.Queue()  # Queue to pass annotated frames to the main thread
        self.running = True

    def detection_worker(self):
        logging.info("Starting streaming detection with YOLOv5...")
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            logging.error("Could not open the camera.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.error("Could not read frame from the camera.")
                break

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            start_time = time.time()

            # Run the prediction on the frame
            results = self.model(frame)
            annotated_frame = frame.copy()
            try:
                # Each detection is a tensor [x1, y1, x2, y2, conf, cls]
                for detection in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = detection
                    x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                    class_name = results.names[int(cls.item())]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error processing detections: {e}")

            self.frame_queue.put(annotated_frame)

            if self.detection_interval != 0:
                detection_time = time.time() - start_time
                wait_time = max(self.detection_interval - detection_time, 0)
                logging.info(f"Waiting {wait_time:.2f} seconds for next detection.")
                time.sleep(wait_time)
        cap.release()
        logging.info("Ending YOLOv5 detection worker.")

    def start_detection(self):
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        return detection_thread

    def stop(self):
        self.running = False


class YOLODetectorV8:
    def __init__(self, camera_source=0, detection_interval=0):
        """
        Initializes the YOLOv8 detector.
         - camera_source: video source (default camera 0).
         - detection_interval: minimum time (in seconds) between processing frames.
        """
        self.camera_source = camera_source
        self.detection_interval = detection_interval
        self.model = load_yolov8_model("yolov8x.pt")
        self.frame_queue = queue.Queue()  # Queue to pass annotated frames to the main thread
        self.running = True

    def detection_worker(self):
        logging.info("Starting streaming detection with YOLOv8...")
        try:
            # YOLOv8 internally captures the video stream when using a source and stream=True
            results = self.model.predict(source=self.camera_source, stream=True, show=False)
        except Exception as e:
            logging.error(f"Error starting YOLOv8 prediction: {e}")
            return

        for result in results:
            if not self.running:
                break
            start_time = time.time()

            # Get the frame from the result (it might be in 'orig_img' or in 'imgs')
            if hasattr(result, 'orig_img'):
                frame = result.orig_img
            elif hasattr(result, 'imgs'):
                frame = result.imgs[0]
            else:
                logging.error("Could not obtain frame image.")
                continue

            # Flip the frame horizontally to create a mirror effect
            frame = cv2.flip(frame, 1)
            frame_width = frame.shape[1]
            annotated_frame = frame.copy()

            try:
                if result.boxes.shape[0] > 0:
                    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        # Original coordinates from the unflipped image
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        # Adjust coordinates for the flipped frame
                        new_x1 = frame_width - x2
                        new_x2 = frame_width - x1
                        class_name = self.model.names[int(cls)]
                        cv2.rectangle(annotated_frame, (new_x1, y1), (new_x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, class_name, (new_x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error processing YOLOv8 frame: {e}")

            self.frame_queue.put(annotated_frame)

            if self.detection_interval != 0:
                detection_time = time.time() - start_time
                wait_time = max(self.detection_interval - detection_time, 0)
                logging.info(f"Waiting {wait_time:.2f} seconds for next detection.")
                time.sleep(wait_time)
        logging.info("Ending YOLOv8 detection worker.")

    def start_detection(self):
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        return detection_thread

    def stop(self):
        self.running = False
