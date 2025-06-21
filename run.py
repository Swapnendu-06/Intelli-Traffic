# Optional: Install required packages (uncomment each line if running for the first time)

# Install the YOLOv8 framework from Ultralytics — includes YOLO models and inference tools
# !pip install ultralytics

# Install OpenCV for video processing, frame capture, and drawing functions like cv2.imshow and cv2.polylines
# !pip install opencv-python

# Install Supervision library — provides annotation utilities for YOLO results (used for overlays, tracking, etc.)
# import subprocess
# subprocess.check_call(["pip", "install", "supervision"])

# Install NumPy — used for numerical operations and for polygon creation with cv2
# !pip install numpy

# Optional: If you plan to log or analyze data using Pandas (e.g., for velocity, counts, etc.)
# !pip install pandas

# Optional: If using OpenCV features from the contrib repository (more advanced modules)
# !pip install opencv-contrib-python

# Optional: Install data analysis and plotting tools for logging and visualization
# Includes numpy (math), pandas (data tables), matplotlib/seaborn (graphs)
# !pip install numpy pandas matplotlib seaborn

import ultralytics
import subprocess
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np  # Required for polygon math and filtering
import math

# Detect on GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Define only traffic-related COCO class IDs (editable, each on its own line)
TRAFFIC_CLASSES = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7    # truck
  #   9,   # traffic light
  #   11   # stop sign
]

# Global dictionary to store last positions of each object (by track_id)
last_positions = {}

# Helper function to check if a point lies inside a polygon
def is_point_inside_polygon(point, polygon):
    # Returns True if the point is inside the polygon
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# Count objects inside a polygonal zone in the current frame
# Only count those that belong to traffic-related classes
def count_objects_in_frame(frame, detections):
    polygon_points = [(300, 200), (600, 220), (620, 400), (280, 390)]  # Irregular polygon points
    object_count = 0  # Counter for valid objects

    for result in detections:
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # Get box coordinates
            cls = boxes.cls.cpu().numpy().astype(int)  # Get class IDs
            for box, cls_id in zip(xyxy, cls):
                if cls_id not in TRAFFIC_CLASSES:
                    continue  # Skip non-traffic classes
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)  # X coordinate of center
                cy = int((y1 + y2) / 2)  # Y coordinate of center
                if is_point_inside_polygon((cx, cy), polygon_points):
                    object_count += 1
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)  # Optional debug dot

    # Draw polygon zone
    polygon_np = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon_np], isClosed=True, color=(255, 0, 0), thickness=2)

    # Annotate count on frame
    cv2.putText(
        frame,
        f"Objects in zone: {object_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return frame

# Draw bounding boxes and velocity using previous center points
def instantvel(frame, detections, p, fps):
    global last_positions

    for result in detections:
        if result.boxes.id is None:
            continue  # Skip if no ID assigned

        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int)  # Track IDs
        classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs
        xyxy = boxes.xyxy.cpu().numpy()  # Box coordinates

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            track_id = ids[i]
            cls_id = classes[i]

            cx = (x1 + x2) / 2  # X center of box
            cy = (y1 + y2) / 2  # Y center of box
            current_point = (cx, cy)

            if track_id in last_positions:
                prev_point = last_positions[track_id]
                dx = current_point[0] - prev_point[0]  # Change in x
                dy = current_point[1] - prev_point[1]  # Change in y
                pixel_dist = math.sqrt(dx**2 + dy**2)  # Euclidean distance
                velocity = (pixel_dist / fps) * p  # Convert to real-world speed
            else:
                velocity = 0  # No previous point

            last_positions[track_id] = current_point  # Update last position

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Annotate box with ID and velocity
            label = f"ID: {track_id}, V: {velocity:.2f} m/s"
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return frame

# Performs tracking, prediction, counting, and velocity measurement
def trackandpredict(source_path):
    cap = cv2.VideoCapture(source_path)  # Open video file
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Read FPS; fallback to 30 if unknown
    cap.release()

    p = 0.05  # Meters per pixel ratio for velocity calculation (adjust based on scene)

    # --- PREDICTION STEP ---
    model.predict(
        source=source_path,       # Path to video or image file
        save=True,                # Save output visualizations (frames with boxes)
        imgsz=1088,               # Inference image size (adjust for performance/accuracy tradeoff)

        iou=0.5,                  # IOU threshold for Non-Maximum Suppression (NMS)
                                 # IOU (Intersection over Union) measures how much two boxes overlap
                                 # Boxes with IOU > 0.5 are considered overlapping; lower threshold = more boxes kept

        conf=0.15,                # Confidence threshold — only predictions above this score are kept
                                 # Lower values increase recall (more detections) but may increase false positives

     #  verbose=True,            # Optional: display detailed output per frame
     #  show=True,               # Optional: display prediction window (only in interactive mode)

        save_txt=True,           # Save results in YOLO format (.txt files)
                                 # Each line: class_id, center_x, center_y, width, height (all normalized)

        save_conf=True,          # Save confidence scores along with detections in the .txt files
                                 # Useful for later filtering or analysis

     #  save_crop=True,         # Optional: save cropped images of detected objects
     #  line_width=2,           # Optional: thickness of box borders in saved image
     #  hide_labels=False,      # Optional: show/hide class labels
     #  hide_conf=False,        # Optional: show/hide confidence values

        agnostic_nms=True,       # Use class-agnostic NMS — suppress overlapping boxes regardless of class
                                 # Useful when overlapping objects may be from different classes

     #  max_det=1000,           # Optional: maximum number of detections per frame

        device=device            # Run on CUDA GPU if available, else CPU
    )

    # --- TRACKING STEP ---
    results = model.track(
        source=source_path,       # Video or image input

        conf=0.15,                # Minimum confidence threshold for detections

        imgsz=1088,               # Input image size for inference

        iou=0.5,                  # IOU threshold for NMS during tracking

        save=True,                # Save output video/images with tracking annotations

     #  verbose=True,            # Optional: log details to console

        tracker="bytetrack.yaml",# Tracker configuration file — ByteTrack in this case

        persist=True,            # Retain object IDs between frames
                                 # Crucial for tracking the same object over time and calculating velocity

        save_txt=True,           # Save tracking output in YOLO txt format with IDs
                                 # Each detection line includes ID in addition to class and box info

        save_conf=True,          # Save confidence scores in the .txt output

        stream=True,             # Return a generator to process results frame by frame
                                 # Useful for real-time processing and per-frame access to outputs

        device=device             # Use GPU/CPU based on availability
    )

    try:
        for result in results:
            # Filter out non-traffic classes
            result.boxes = result.boxes[
                np.isin(result.boxes.cls.cpu().numpy().astype(int), TRAFFIC_CLASSES)
            ]

            frame = result.orig_img                          # Original frame from video/image
            frame = count_objects_in_frame(frame, [result])  # Count traffic-relevant objects in polygon zone
            frame = instantvel(frame, [result], p, fps)      # Estimate velocity using pixel displacement
            cv2.imshow("Object Count View", frame)           # Display current frame

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Exit loop when 'q' is pressed
    finally:
        cv2.destroyAllWindows()  # Clean up and close any OpenCV display windows


# Main function to launch tracking
def main():
    source = r"testing\street2.mp4"  # Replace with video for live tracking
    trackandpredict(source)

# Run main
if __name__ == "__main__":
    main()