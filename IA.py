from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict(
    source="0", 
    show=True)

print(results)
