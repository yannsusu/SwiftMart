import numpy as np
import torch
import os
import cv2

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']

class DeepSort(object):
    def __init__(self, model_path, output_dir, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.output_dir = self._create_unique_output_dir(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_count = 0  # 初始化 frame_count
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def _create_unique_output_dir(self, output_dir):
        suffix = 1
        unique_output_dir = output_dir
        while os.path.exists(unique_output_dir):
            unique_output_dir = f"{output_dir}_{suffix}"
            suffix += 1
        return unique_output_dir

    def update(self, bbox_xywh, confidences, oids, ori_img):
        self.frame_count += 1
        orig_h, orig_w = ori_img.shape[:2]  # 原始尺寸
        target_size = 224  # 目标尺寸

        # 直接拉伸图片到 224x224
        resized_img = cv2.resize(ori_img, (target_size, target_size))

        # 生成特征
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i], oid) for i, (conf, oid) in
                      enumerate(zip(confidences, oids)) if conf > self.min_confidence]

        # 运行 DeepSort 追踪器
        self.tracker.predict()
        self.tracker.update(detections)

        # 处理追踪输出
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box, orig_w, orig_h, target_size)

            track_id = track.track_id
            track_oid = track.oid
            outputs.append(np.array([x1, y1, x2, y2, track_id, track_oid], dtype=np.int64))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        # 保存 frame 和追踪信息
        self._save_track_info(outputs, resized_img)

        return outputs

    def _save_track_info(self, outputs, resized_img):
        """
        记录当前帧的所有 track_id 到 tracking.seq，并在 224x224 图片上绘制所有 bounding box
        """
        seq_path = os.path.join(self.output_dir, "tracking.seq")

        with open(seq_path, 'a') as f:
            for i in range(len(outputs)):
                x1, y1, x2, y2, track_id, track_oid = outputs[i]

                # 记录到 tracking.seq，已经是 224x224 的坐标
                f.write(f"{self.frame_count},{track_id},{x1},{y1},{x2},{y2}\n")

                # 在 224x224 图片上绘制 bounding box 和 ID
                cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边框
                cv2.putText(resized_img, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存 224x224 版本的当前帧
        img_path = os.path.join(self.output_dir, f"{self.frame_count}.jpg")
        cv2.imwrite(img_path, resized_img)

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box, ori_img.shape[1], ori_img.shape[0], 224)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh, img_width, img_height, target_size):
        """
        把 bbox 坐标从原始尺寸转换到 224x224
        """
        x, y, w, h = bbox_xywh
        x1 = max(int((x - w / 2) * target_size / img_width), 0)
        x2 = min(int((x + w / 2) * target_size / img_width), target_size - 1)
        y1 = max(int((y - h / 2) * target_size / img_height), 0)
        y2 = min(int((y + h / 2) * target_size / img_height), target_size - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh, img_width, img_height, target_size):
        """
        把 tlwh 格式 bbox 转换到 224x224
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x * target_size / img_width), 0)
        x2 = min(int((x + w) * target_size / img_width), target_size - 1)
        y1 = max(int(y * target_size / img_height), 0)
        y2 = min(int((y + h) * target_size / img_height), target_size - 1)
        return x1, y1, x2, y2
