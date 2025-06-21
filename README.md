# IntelliTraffic: Smart Traffic Monitoring and Violation Detection System

## Overview

IntelliTraffic is an advanced AI-powered computer vision system designed for intelligent traffic monitoring, object tracking, and rule violation detection. It utilizes YOLOv8, OpenCV, and Supervision to detect vehicles and pedestrians, estimate real-time speeds, and identify movement within defined zones of a traffic scene. The system is scalable and modular to support upcoming features such as automatic challan issuance, license plate recognition, and integration with India's Defense Satellite APIs for broader surveillance.

## Project Objective

With increasing traffic congestion, law violations, and road accidents, a scalable, AI-driven solution is needed to:

* Automate vehicle detection and classification.
* Track object motion and estimate velocities.
* Monitor restricted zones and detect intrusions or violations.
* Help traffic authorities with automated reports and actions.
* Lay the foundation for smart city infrastructure.

## Core Technologies Used

| Technology           | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| YOLOv8 (Ultralytics) | State-of-the-art object detection and tracking                 |
| ByteTrack            | Lightweight and accurate multi-object tracking                 |
| OpenCV               | Frame processing, drawing, and geometric calculations          |
| Supervision          | Utility layer for detection rendering and result visualization |
| NumPy & Math         | Efficient numerical and geometric operations                   |
| Torch (PyTorch)      | GPU/CPU acceleration and tensor management for YOLO            |

## Project Structure

```
intellitraffic/
|
├── main.py                    # Entry point for tracking and prediction
├── README.md                  # This documentation file
├── requirements.txt           # Package dependencies
├── testing/
│   └── street2.mp4            # Sample video file for testing
├── outputs/                   # Automatically generated tracked outputs
└── ...
```

## Installation

You can install the required libraries using pip:

```
pip install ultralytics opencv-python numpy supervision torch
```

Optional (for additional plotting/logging):

```
pip install pandas matplotlib seaborn opencv-contrib-python
```

## How It Works

### 1. Object Detection (YOLOv8)

* The model yolov8m.pt is loaded and used to detect traffic objects (cars, buses, persons, bikes, trucks).
* Confidence and IOU thresholds filter out low-quality predictions.
* Class-agnostic NMS ensures overlapping objects from different classes are properly handled.

### 2. Object Tracking (ByteTrack)

* Tracking assigns persistent IDs to objects across frames.
* This allows us to monitor motion over time (essential for speed and behavior analysis).
* persist=True keeps consistent IDs.

### 3. Counting in Polygonal Zone

* A user-defined polygonal area is defined (e.g., a no-parking or no-entry zone).
* The center of each detected object is checked if it lies within this polygon.
* The frame is annotated with the count.

### 4. Velocity Estimation

* For tracked objects, the system calculates pixel displacement across frames.
* Using frame rate (fps) and a meter-per-pixel scale (p), real-world velocity is computed.
* Velocity is displayed alongside the object's bounding box and ID.

## Configuration Parameters

| Parameter        | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| TRAFFIC\_CLASSES | List of COCO class IDs relevant to traffic                     |
| p                | Real-world scaling factor (meters per pixel)                   |
| imgsz            | Inference resolution for YOLO model (higher = better accuracy) |
| iou              | IOU threshold for box suppression                              |
| conf             | Minimum detection confidence                                   |
| polygon\_points  | Points defining the counting zone                              |

## How to Use

```
python main.py
```

Ensure your video file is correctly referenced in:

```python
source = r"testing\street2.mp4"
```

You can change this to any video or even a live webcam feed.

## Output

* Saved videos with bounding boxes and annotated velocities
* Real-time display of object counts inside defined zones
* Output .txt files with detection and tracking results (YOLO format)

## Planned Features

### License Plate Recognition (LPR)

* Integration of OCR models (e.g., EasyOCR or CRNN) to extract license plates
* Can be used to:

  * Identify vehicles violating speed or parking rules
  * Link to RTO databases for challan generation

### Automatic Challan System

* Integration with state-wise RTO APIs to:

  * Retrieve vehicle ownership info
  * Trigger challan generation
  * Send violation notices via SMS/email

### Integration with India's Defense Satellite API

* For high-altitude aerial traffic analytics
* Use satellite feeds to:

  * Identify traffic build-ups
  * Spot illegal gatherings
  * Detect unregistered or suspicious vehicle patterns

### AI-based Traffic Control System

* Smart traffic signal control based on:

  * Vehicle density per lane
  * Real-time velocity patterns
  * Emergency vehicle detection and prioritization

### Dashboard (Web + Mobile)

* Central dashboard with:

  * Live feed
  * Violation history
  * Statistics per zone/time interval
  * Admin tools to generate reports and alerts

## License

This project is open-source under the MIT License. You are free to use, modify, and distribute it with attribution. Commercial use requires permission for any integration with government bodies or surveillance authorities.

## Contribution Guidelines

We welcome contributions. Here's how you can help:

* Submit issues and feature requests
* Fork and raise pull requests
* Contribute to documentation
* Help build a web interface or REST API
* Integrate OCR and RTO APIs

## Contact and Credits

Author: Swapnendu Sikdar
Email: swapnendusikdar@gmail.com
Institution: Jadavpur University
Field: Electrical Engineering
Interests: AI/ML, Deep Learning,Machine Learning, Computer Vision, Robotics, Smart Infrastructure

## Acknowledgments

* Ultralytics for the YOLOv8 framework
* ByteTrack for object tracking
* OpenCV for image and video processing
* India Meteorological and Defense APIs (future integration)
