# Lane Detection & Vehicle Counting System

## Overview
This project detects and counts vehicles in different road lanes from a traffic video. It uses **YOLOv8** for object detection and **Norfair** for tracking.

---

## Features
- Vehicle detection for: car, bus, truck, motorcycle, bicycle
- Lane assignment using polygons
- Tracking with unique IDs across frames
- CSV and annotated video output

---

## install uv
1. run the command:
   ```bash
   pip install uv
## Setup (uv)
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lanedetection.git
   cd lanedetection
2. Install all packages:
   ```bash
   uv run sync
3. Or you can install via pip using uv
   ```bash
   uv pip install ultralytics norfair opencv-python pandas tqdm numpy
## Run the project
1. Place your traffic video as traffic.mp4 in the project root.
2. Execute:
   ```bash
   uv run main.py
3. Outputs:
- output/annotated.mp4 — video with detections and lanes
- output/vehicle_counts.csv — vehicle counts with timestamps