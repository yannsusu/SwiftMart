import sys
sys.path.append(r"C:\Users\xuchi\anaconda3\envs\PracticeModule\Lib\site-packages")

import torch
import json
import cv2
import numpy as np
import torch.nn as nn

from torch.serialization import add_safe_globals
from torchvision import transforms
from timesformer.models.vit import TimeSformer
from MoEMLP import MoEMLP

add_safe_globals([TimeSformer, MoEMLP])

def compute_optical_flow(prev_frame, next_frame):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_x = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    flow_y = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = np.stack([flow_x, flow_y, np.zeros_like(flow_x)], axis=-1)
    return flow_rgb.astype(np.uint8)

def load_video(video_path, num_frames=45, resize=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    cap = cv2.VideoCapture(video_path)
    frames, prev_frame = [], None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow_frame = compute_optical_flow(prev_frame, frame)
            frames.append(flow_frame)
        prev_frame = frame
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    frames = frames[:num_frames]
    frames = np.array(frames, dtype=np.float32) / 255.0
    video_tensor = torch.tensor(frames).permute(3, 0, 1, 2)
    normalize = transforms.Normalize(mean=mean, std=std)
    for t in range(video_tensor.shape[1]):
        video_tensor[:, t] = normalize(video_tensor[:, t])
    return video_tensor

def load_model(weight_path, device):
    model = TimeSformer(img_size=224, patch_size=16, num_classes=4, num_frames=45)
    for block in model.model.blocks:
        block.mlp = MoEMLP(in_features=768, hidden_features=3072, num_experts=4, top_k=2)
    model.head = nn.Linear(768, 4)
    model = torch.load(weight_path, map_location=device, weights_only=False)
    return model.to(device).eval()

def run(video_path="test.mp4"):
    """
    传入视频路径，返回预测的类别名和类别ID
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_tensor = load_video(video_path).unsqueeze(0).to(device)
    model = load_model("../Models/full_timeSformer.pth", device)

    with open("../Models/labels.json", "r", encoding="utf-8") as f:
        labels_map = json.load(f)

    with torch.no_grad():
        output = model(video_tensor)
        _, predicted = torch.max(output, 1)

    label_id = predicted.item()
    label_name = next((k for k, v in labels_map.items() if v == label_id), "未知")
    print(f"预测结果: 类别 {label_id} - {label_name}")
    return label_id, label_name
