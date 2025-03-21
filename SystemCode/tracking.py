import cv2
import torch
import random
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from Models.deep_sort_pytorch.utils.parser import get_config
from Models.deep_sort_pytorch.deep_sort.deep_sort_modified import DeepSort

# Initialize YOLO model
model = YOLO("../Models/best.pt")

# Open video file
video_path = "../Dataset/final_test.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

shopping_cart = {}  # Dictionary to track objects and their counts

# Get YOLOv8 class names
class_names = model.names

# Allowed items to track
allowed_classes = [
    "AD Calcium Milk", "Coca-Cola", "Daliyuan Soft Bread",
    "Kiss Burn Braised Beef Flavor", "Kiss Burn Spicy Chicken Flavor",
    "Lai Yi Tong Instant Noodles", "RIO Lychee Flavor",
    "RIO Strawberry Flavor", "Tea Pi", "Want Want Senbei"
]
allowed_class_ids = [k for k, v in class_names.items() if v in allowed_classes]

# Generate random colors for each class
class_colors = {}

# Initialize DeepSORT
def init_tracker():
    """Initialize the DeepSORT tracker."""
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("../Models/deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )


# Function to process detections and update the shopping cart
def process_detections(results):
    """Process YOLO detection results and return valid detections."""
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            # Filter out non-relevant classes
            if cls not in allowed_class_ids:
                continue

            x_center, y_center, w, h = box.xywh[0]
            x_center, y_center, w, h = map(int, [x_center, y_center, w, h])

            x1 = x_center - w // 2
            y1 = y_center - h // 2
            x2 = x_center + w // 2
            y2 = y_center + h // 2

            width = int(x2 - x1)
            height = int(y2 - y1)
            conf = float(box.conf[0])
            detections.append(([x1, y1, width, height], conf, cls))
    return detections


# Function to update the tracker with detections
def update_tracker(detections, frame):
    """Update DeepSORT tracker and draw results."""
    bbox_xywh = []
    confidences = []
    class_ids = []

    for det in detections:
        (x1, y1, width, height), conf, cls_id = det
        bbox_xywh.append([x1 + width // 2, y1 + height // 2, width, height])
        confidences.append(conf)
        class_ids.append(cls_id)

    if bbox_xywh:
        bbox_xywh = torch.Tensor(bbox_xywh)
        confidences = torch.Tensor(confidences)
        outputs = deepsort.update(bbox_xywh, confidences, class_ids, frame)

        return outputs
    return []


# Function to handle object tracking and update shopping cart
def track_objects(outputs, frame):
    """Track objects and update the shopping cart."""
    for output in outputs:

        x1, y1, x2, y2, track_id, cls_id = output
        cls_name = model.names[cls_id]
        color = get_class_color(cls_name)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"ID: {track_id} {cls_name}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        # Update shopping cart with item class count
        if cls_name not in shopping_cart:
            shopping_cart[cls_name] = 1
        else:
            shopping_cart[cls_name] += 1


# Function to initialize the video capture
def init_video_capture(video_path):
    """Initialize video capture."""
    return cv2.VideoCapture(video_path)


# Function to get a random color for each object
def get_class_color(cls_name):
    """Generate a random color for each class."""
    if cls_name not in class_colors:
        class_colors[cls_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return class_colors[cls_name]


# Main processing loop
def tracking():
    """Main loop to process video frames and perform object tracking."""
    video_path = "../Dataset/final_test.mp4"  # Replace with your video file path
    cap = init_video_capture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame, conf=0.7, iou=0.5, agnostic_nms=True)

        # Process detections
        detections = process_detections(results)

        # Update the tracker with new detections
        outputs = update_tracker(detections, frame)

        # Track objects and update the shopping cart
        track_objects(outputs, frame)

        # Display results
        cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    init_tracker()
    tracking()