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
# Added for output directory creation and video saving
import os  # NEW: Required for directory and path handling
from datetime import datetime  # NEW: For unique output file names

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
    polygon_points = [(773, 14), (3452, 14), (3800, 2107), (34, 2088)]  # Irregular polygon points
    object_count = 0  # Counter for valid objects

    for result in detections:
        boxes = result.boxes
        if boxes is not None:
            # FIX 1: Check if boxes have any detections before proceeding
            if len(boxes.xyxy) == 0:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()  # Get box coordinates
            cls = boxes.cls.cpu().numpy().astype(int)  # Get class IDs
            for box, cls_id in zip(xyxy, cls):
                # FIX 9: Filter traffic classes here instead of modifying result object
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

    # Annotate count on frame - MODIFIED: Changed color to blue and moved position down
    cv2.putText(
        frame,
        f"Objects in zone: {object_count}",
        (10, 60),  # MODIFIED: Changed Y position from 30 to 60 to move text down
        cv2.FONT_HERSHEY_SIMPLEX,
        4.5,  # NEW: Increased font scale from 1.5 to 4.5 (300% increase) for larger object counter text
        (255, 0, 0),  # MODIFIED: Changed color to blue (BGR: 255, 0, 0) to match bounding boxes
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
        # FIX 2: Check if boxes have any detections before proceeding
        if len(boxes.xyxy) == 0:
            continue
            
        ids = boxes.id.cpu().numpy().astype(int)  # Track IDs
        classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs
        xyxy = boxes.xyxy.cpu().numpy()  # Box coordinates

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            track_id = ids[i]
            cls_id = classes[i]
            
            # FIX 9: Filter traffic classes here instead of modifying result object
            if cls_id not in TRAFFIC_CLASSES:
                continue  # Skip non-traffic classes

            # NEW: Slightly increase box size by expanding coordinates
            box_expand = 10  # NEW: Increase box size by 10 pixels on each side
            x1 = x1 - box_expand  # NEW: Expand left
            y1 = y1 - box_expand  # NEW: Expand top
            x2 = x2 + box_expand  # NEW: Expand right
            y2 = y2 + box_expand  # NEW: Expand bottom

            cx = (x1 + x2) / 2  # X center of box
            cy = (y1 + y2) / 2  # Y center of box
            current_point = (cx, cy)

            if track_id in last_positions:
                prev_point = last_positions[track_id]
                dx = current_point[0] - prev_point[0]  # Change in x
                dy = current_point[1] - prev_point[1]  # Change in y
                pixel_dist = math.sqrt(dx**2 + dy**2)  # Euclidean distance
                velocity = (pixel_dist / fps) * p * 3.6  # NEW: Convert to km/hr (m/s * 3.6 = km/hr)
            else:
                velocity = 0  # No previous point

            last_positions[track_id] = current_point  # Update last position

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 6)  # NEW: Changed color to blue (BGR: 255, 0, 0) and thickness from 2 to 6 (300%)
            # Annotate box with ID and velocity - MODIFIED: Changed text color to blue
            label = f"ID: {track_id}, V: {velocity:.2f} km/h"  # NEW: Changed label to display km/h
            # NEW: Increased font scale from 0.8 to 2.4 (300% increase) for larger velocity text
            # MODIFIED: Changed text color to blue (255, 0, 0) to match bounding boxes
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 0, 0), 1)

    return frame

# Performs tracking, prediction, counting, and velocity measurement
def trackandpredict(source_path):
    # NEW: Create output directory for custom video
    output_dir = "runs/custom_output"  # NEW: Custom directory for saving output video
    os.makedirs(output_dir, exist_ok=True)  # NEW: Create directory if it doesn't exist
    
    # NEW: Define output video path with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # NEW: Unique timestamp
    output_video_path = os.path.join(output_dir, f"output_{timestamp}.mp4")  # NEW: Output video path

    cap = cv2.VideoCapture(source_path)  # Open video file
    # FIX 3: Add error handling for video file opening
    if not cap.isOpened():
        print(f"Error: Could not open video file {source_path}")
        return
        
    # NEW: Get video properties for VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Read FPS; fallback to 30 if unknown
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # NEW: Video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # NEW: Video height

    # NEW: Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # NEW: Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # NEW: Video writer object

    p = 5  # MODIFIED: Changed from 300 to 5 for velocity calculation

    # FIX 10: First run prediction step to completion and save results
    print("Running prediction step...")
    prediction_results = model.predict(
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
                                 # Useful for filtering or analysis

     #  save_crop=True,         # Optional: save cropped images of detected objects
     #  line_width=2,           # Optional: thickness of box borders in saved image
     #  hide_labels=False,      # Optional: show/hide class labels
     #  hide_conf=False,        # Optional: show/hide confidence values

        agnostic_nms=True,       # Use class-agnostic NMS — suppress overlapping boxes regardless of class
                                 # Useful when overlapping objects may be from different classes

     #  max_det=1000,           # Optional: maximum number of detections per frame

        device=device            # Run on CUDA GPU if available, else CPU
    )
    print(f"Prediction completed. Results saved to: {prediction_results[0].save_dir if prediction_results else 'Unknown'}")

    # --- TRACKING STEP ---
    print("Running tracking step...")
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
            # FIX 4: Add check to ensure result.boxes exists and has content before filtering
            if result.boxes is None or len(result.boxes.xyxy) == 0:
                frame = result.orig_img
                frame = count_objects_in_frame(frame, [])  # Pass empty list for zero count
                out.write(frame)  # NEW: Save empty frame to video
                cv2.imshow("Object Count View", frame)           # Display current frame
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            
            frame = result.orig_img                          # Original frame from video/image
            frame = count_objects_in_frame(frame, [result])  # Count traffic-relevant objects in polygon zone
            frame = instantvel(frame, [result], p, fps)      # Estimate velocity using pixel displacement
            out.write(frame)  # NEW: Save modified frame to video
            cv2.imshow("Object Count View", frame)           # Display current frame

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Exit loop when 'q' is pressed
    except Exception as e:
        # FIX 7: Add exception handling for tracking errors
        print(f"Error during tracking: {e}")
    finally:
        # NEW: Release video resources
        cap.release()  # NEW: Release input video capture
        out.release()  # NEW: Release VideoWriter
        cv2.destroyAllWindows()  # Clean up and close any OpenCV display windows
        print(f"Output video saved to: {output_video_path}")  # NEW: Inform user of saved video location

# Main function to launch tracking
def main():
    source = r"testing\street2.mp4"  # Replace with video for live tracking
    # FIX 8: Add file existence check
    import os
    if not os.path.exists(source):
        print(f"Error: Video file '{source}' not found. Please check the path.")
        return
    
    trackandpredict(source)

# Run main
if __name__ == "__main__":
    main()
