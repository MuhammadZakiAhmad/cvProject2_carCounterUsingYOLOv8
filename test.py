import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

# Define the region of interest (ROI) area
area = [(270, 238), (285, 262), (592, 226), (552, 207)]

# Initialize the area_c set for counting cars
area_c = set()

# Create a dictionary to store car counts
car_counts = {}

# Tracking Mouse Movements
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window for displaying the video
cv2.namedWindow('Car Counting')
cv2.setMouseCallback('Car Counting', RGB)

# Load the video
cap = cv2.VideoCapture('vidyolov8.mp4')

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Read class labels from the COCO file
with open("D:\Learning\Computer Vision\CarCounter\yolov8-opencv-win11-main\yolov8-opencv-win11-main\coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize counters and the object tracker
count = 0
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # Process every 10th frame
    if count % 10 != 0:
        continue

    # Resize the frame
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame)
    boxes = results[0].boxes.data

    # Extract car bounding boxes
    car_boxes = []
    for box in boxes:
        x1, y1, x2, y2, _, class_id = box
        if 'car' in class_list[int(class_id)]:
            car_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    # Update the object tracker with car bounding boxes
    bbox_ids = tracker.update(car_boxes)

    # Draw bounding boxes and count cars within the ROI
    for bbox in bbox_ids:
        x1, y1, x2, y2, obj_id = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Check if the center of the bounding box is inside the ROI
        results = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if results >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            area_c.add(obj_id)

    # Count cars and display the count on the frame with a background rectangle
    car_count = len(area_c)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 0), 3)  # Make the polygon black
    cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)  # Add a background rectangle
    cv2.putText(frame, f'Car Count: {car_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Car Counting", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
