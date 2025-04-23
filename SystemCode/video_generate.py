import cv2
import os
from glob import glob

def images_to_video(image_folder, output_path="output_video.mp4", fps=15):

    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))

    if not image_paths:
        raise ValueError("未找到图像，请检查路径或扩展名")

    # 读取第一帧来获取图像尺寸
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可改为 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"跳过无法读取的图像: {path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    img_folder = r'C:\Users\xuchi\Desktop\SwiftMart\practice_module_CV_v1\YOLOv8-DeepSORT\YOLOv8-DeepSORT\content\output_7'
    images_to_video(img_folder)