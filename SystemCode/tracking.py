import cv2
import random
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from Models.deep_sort_pytorch.utils.parser import get_config
from Models.deep_sort_pytorch.deep_sort.deep_sort_modified import DeepSort

# Load YOLOv8 model
model = YOLO("../Models/best.pt")

# Function to initialize DeepSORT
def init_tracker():
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

# Initialize DeepSORT tracker
init_tracker()
# tracker = DeepSort(max_age=50, n_init=3, embedder="mobilenet", max_iou_distance=0.7)


# Open video file
video_path = '../Dataset/final_test.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

shopping_cart = {}

# Frame skipping settings
frame_skip = 1
frame_count = 0

# Get YOLOv8 class names
class_names = model.names  # Dictionary mapping class IDs to names

allowed_classes = ['AD Calcium Milk', 'Coca-Cola', 'Daliyuan Soft Bread', 'Kiss Burn Braised Beef Flavor', 'Kiss Burn Spicy Chicken Flavor', 'Lai Yi Tong Instant Noodles', 'RIO Lychee Flavor', 'RIO Strawberry Flavor', 'Tea Pi', 'Want Want Senbei']
allowed_class_ids = [k for k, v in class_names.items() if v in allowed_classes]

class_colors = {}

def get_class_color(cls_name):
    if cls_name not in class_colors:
        class_colors[cls_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return class_colors[cls_name]

people_seq = [
    (68, 93, 523, 369, 1172),  # (id, x1, x2, y1, y2)
]


# Parameters for "pick-up" detection
N = 5  # Frames threshold to be considered as "picked up"
speed_threshold = 5  # Speed threshold to detect slow movement

# Define variables for tracking whether the item is "picked up"
picked_up_items = {}
# Action received flag
action_received = 'pick' # Can be 'pick', 'drop', 'fall', or 'none'

# Function to calculate the speed of the object
def calculate_speed(last_position, current_position, fps):
    distance = euclidean(last_position, current_position)
    return distance * fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if action_received == 'pick':

        results = model(frame, conf=0.7, iou=0.5, agnostic_nms=True)

        # Extract detection results (only for allowed classes)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                x_center, y_center, w, h = box.xywh[0]
                x_center, y_center, w, h = map(int, [x_center, y_center, w, h])

                x1 = x_center - w // 2
                y1 = y_center - h // 2
                x2 = x_center + w // 2
                y2 = y_center + h // 2

                width = abs(x2 - x1)
                height = abs(y2 - y1)
                conf = float(box.conf[0])
                detections.append(([x1, y1, width, height], conf, cls))

        # Perform object tracking using DeepSORT
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():  # Fixed spelling: is_confirmed
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            cls_id = getattr(track, 'det_class', None)
            if cls_id is None or cls_id not in allowed_class_ids:
                continue

            cls_name = class_names[cls_id]
            color = get_class_color(cls_name)

            # Initialize tracking for the item if not done already
            if track_id not in picked_up_items:
                picked_up_items[track_id] = {
                    "frames_in_box": 0,
                    "last_position": None,
                    "is_picked_up": False,
                    "last_speed": 0,
                    "person_frame": None
                }

            # Update tracking status
            item_status = picked_up_items[track_id]
            last_position = item_status["last_position"]
            current_position = (x_center, y_center)

            # Check if item is within person's box
            person_in_box = False
            for person in people_seq:
                person_x1, person_y1, person_x2, person_y2, _ = person
                if x1 > person_x1 and x2 < person_x2 and y1 > person_y1 and y2 < person_y2:
                    person_in_box = True
                    item_status["person_frame"] = frame_id  # Store the frame when the item is near a person
                    break

            # If the item is in the frame, increase the frame count
            if person_in_box:
                item_status["frames_in_box"] += 1

            # Check if item has been in the frame for more than N frames
            if item_status["frames_in_box"] >= N:
                # Check if item is moving with the person (by comparing position movement)
                if last_position:
                    speed = calculate_speed(last_position, current_position, fps=30)  # Assume 30 FPS
                    if speed < speed_threshold:  # If speed is slow, item might be picked up
                        item_status["is_picked_up"] = True

                item_status["last_position"] = current_position

            # Handle products has been picked up
            if item_status["is_picked_up"] and track_id not in shopping_cart:
                shopping_cart[track_id] = (cls_name, 1)
            elif item_status["is_picked_up"]:
                shopping_cart[track_id] = (cls_name, shopping_cart[track_id][1] + 1)

            # Draw bounding box and text (ID + Class Name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id} {cls_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    elif action_received == 'drop':
        for track_id in list(picked_up_items.keys()):
            if track_id in shopping_cart:
                del shopping_cart[track_id]
            picked_up_items[track_id]["is_picked_up"] = False
            picked_up_items[track_id]["frames_in_box"] = 0

    # Display results
    cv2.namedWindow("YOLOv8 + DeepSORT Tracking", cv2.WINDOW_NORMAL)  # Create a resizable window

    frame_height, frame_width = frame.shape[:2]
    window_scale = 0.8  # Set window scale
    cv2.resizeWindow("YOLOv8 + DeepSORT Tracking", int(frame_width * window_scale), int(frame_height * window_scale))
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Shopping Cart:", shopping_cart)