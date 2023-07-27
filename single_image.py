# Description: This file is used to test the YOLOv8n model on a single image.

# Import the YOLO model
from ultralytics import YOLO

model = YOLO("yolov8n.pt") # Initialize model

results = model.predict(source="bus.jpg", conf=0.5, show=True, verbose=False, stream=True, save=True) # Predict 1 image


for result in results: # Results object with detected classes, boxes, and scores
    boxes = result[0].boxes.numpy() # Boxes object for bbox outputs
    for box in boxes: # There could be more than one detection
        print("class", box.cls)
        print("xyxy", box.xyxy)
        print("conf", box.conf)



   

