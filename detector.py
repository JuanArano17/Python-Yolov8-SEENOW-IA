# detector.py
import time
import threading
import logging
import cv2  # OpenCV for visualization and annotation
import queue
from AI import IADetector

class YOLODetector:
    def __init__(self, camera_source=0, detection_interval=0):
        """
        Initializes the detector:
         - camera_source: Video source (by default camera 0).
         - detection_interval: Minimum time (in seconds) between each processing.
        """
        self.camera_source = camera_source
        self.detection_interval = detection_interval
        self.detector = IADetector("yolov8x.pt")
        self.frame_queue = queue.Queue()  # Queue for passing annotated frames to the main thread
        self.running = True

    def detection_worker(self):
        logging.info("Starting streaming detection with YOLOv8...")
        try:
            # Use show=False to process the image and add custom annotations
            results = self.detector.predict(source=self.camera_source, stream=True, show=False)
        except Exception as e:
            logging.error(f"Error starting prediction: {e}")
            return

        for result in results:
            if not self.running:
                break
            start_time = time.time()

            # Obtain the frame (it might be in 'orig_img' or in 'imgs')
            if hasattr(result, 'orig_img'):
                frame = result.orig_img
            elif hasattr(result, 'imgs'):
                frame = result.imgs[0]
            else:
                logging.error("Could not obtain frame image.")
                continue

            # Flip the frame horizontally so it behaves like a mirror
            frame = cv2.flip(frame, 1)

            # Get the width of the frame to adjust bounding boxes
            frame_width = frame.shape[1]

            # Make a copy for annotations
            annotated_frame = frame.copy()

            try:
                if result.boxes.shape[0] > 0:
                    # Iterate over each detection to extract coordinates and class
                    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                        # Original coordinates from the unflipped image
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        # Adjust coordinates for the flipped frame:
                        new_x1 = frame_width - x2
                        new_x2 = frame_width - x1
                        class_name = self.detector.model.names[int(cls)]
                        # Draw a rectangle with the adjusted coordinates and put the class name
                        cv2.rectangle(annotated_frame, (new_x1, y1), (new_x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, class_name, (new_x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error processing frame: {e}")

            # Send the annotated frame to the queue for display in the main thread
            self.frame_queue.put(annotated_frame)

            if self.detection_interval != 0:
                detection_time = time.time() - start_time
                wait_time = max(self.detection_interval - detection_time, 0)
                logging.info(f"Waiting {wait_time:.2f} seconds for the next detection.")
                time.sleep(wait_time)
        
        logging.info("Ending detection_worker.")
    
    def start_detection(self):
        """
        Starts the detection process in a background thread.
        """
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()
        return detection_thread

    def stop(self):
        self.running = False
