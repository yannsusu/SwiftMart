import sys
sys.path.append(r"C:\Users\xuchi\anaconda3\envs\PracticeModule\Lib\site-packages")

import os
import json
import cv2
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from timesformer.models.vit import TimeSformer

from predict import run

if __name__ == "__main__":
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    #project_dir = os.path.join(script_dir, 'YOLOv8-DeepSORT', 'YOLOv8-DeepSORT', 'content',
    #            'YOLOv8-DeepSORT-Object-Tracking', 'ultralytics', 'yolo', 'v8',
    #            'detect')

    #os.chdir(project_dir)
    #sys.path.append(project_dir)
    #from predict_customer import predict

    #predict()

    video_path = "../../Dataset/final_test.mp4"
    label_id, label_name = run(video_path)

    print("预测结果为", label_id)

