import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from norfair import Detection, Tracker, draw_tracked_objects

# Vehicle classes in COCO dataset
WANTED_CLASS_IDS = {2, 3, 4, 6, 8}  # bicycle, car, motorcycle, bus, truck

def xyxy_to_detection(xyxy, score):
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return Detection(points=np.array([[cx, cy]]), scores=np.array([score]))

def point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x), int(y)), False) >= 0

def main(video_path, output_dir="output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load YOLOv8 model (nano for speed)
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))

    out_video = cv2.VideoWriter(str(output_dir / "annotated.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (width, height))

    # Define lane polygons (adjust for your scene)
    lane1 = [(0, 0), (width // 3, 0), (width // 3, height), (0, height)]
    lane2 = [(width // 3, 0), (2 * width // 3, 0), (2 * width // 3, height), (width // 3, height)]
    lane3 = [(2 * width // 3, 0), (width, 0), (width, height), (2 * width // 3, height)]
    lanes = [lane1, lane2, lane3]

    tracker = Tracker(distance_function="euclidean", distance_threshold=50)

    id_to_lane = {}
    id_first_frame = {}
    id_frame_counts = defaultdict(int)

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
        detections = []
        for cls, xyxy, conf in zip(results.boxes.cls.cpu().numpy(),
                                   results.boxes.xyxy.cpu().numpy(),
                                   results.boxes.conf.cpu().numpy()):
            if int(cls) in WANTED_CLASS_IDS:
                detections.append(xyxy_to_detection(xyxy, conf))

        # Tracking
        tracked_objects = tracker.update(detections=detections)

        for t in tracked_objects:
            cx, cy = t.estimate[0]
            if t.id not in id_to_lane:
                for i, poly in enumerate(lanes, 1):
                    if point_in_polygon(cx, cy, poly):
                        id_to_lane[t.id] = i
                        break
            id_frame_counts[t.id] += 1
            if t.id not in id_first_frame:
                id_first_frame[t.id] = frame_idx

        # Draw lanes
        overlay = frame.copy()
        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]
        for i, poly in enumerate(lanes):
            cv2.polylines(overlay, [np.array(poly, np.int32)], True, colors[i], 2)
            cv2.putText(overlay, f"Lane {i+1}",
                        (poly[0][0] + 20, poly[0][1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

        draw_tracked_objects(overlay, tracked_objects)
        out_video.write(overlay)

    cap.release()
    out_video.release()

    # Save CSV
    data = [{
        "vehicle_id": vid,
        "lane": id_to_lane.get(vid, 0),
        "frame_count": id_frame_counts[vid],
        "first_timestamp": id_first_frame[vid] / fps
    } for vid in id_frame_counts]

    pd.DataFrame(data).to_csv(output_dir / "vehicle_counts.csv", index=False)

    print("Processing complete!")
    print("Annotated video:", output_dir / "annotated.mp4")
    print("CSV saved:", output_dir / "vehicle_counts.csv")

if __name__ == "__main__":
    main("traffic.mp4", output_dir="output")
