# SwiftMart: Intelligent Unmanned Shopping System

## 📌 Project Overview
SwiftMart is an automated shopping recognition system tailored for supermarket environments. It integrates object detection, multi-object tracking, and action recognition technologies to identify customer-product interactions in real time. By combining YOLOv8, DeepSORT, and TimeSformer models, the system enables seamless checkout without manual intervention. SwiftMart enhances retail efficiency, reduces labor costs, and improves the customer shopping experience.

## ✨ Key Features
- **Real-Time Person & Product Tracking**: Utilizes YOLOv8 and DeepSORT for accurate detection and consistent identity tracking.
- **Action Recognition Module**: Leverages TimeSformer and Optical Flow to detect pick-up, put-back, and drop actions.
- **Virtual Shopping Cart**: Automatically maintains a digital cart based on customer interactions.
- **Automated Checkout**: Calculates total price at store exit based on cart contents.
- **Anomaly Alerts**: Notifies staff when items are dropped or mishandled.

## 📊 Datasets
The system is trained on three main datasets:

1. **Person Detection & Tracking Dataset**:
   - Public source: Re-annotated OpenImagesV6
   - Self-recorded surveillance frames: 945 images, including real shopping mall scenes
   - Total: 3,609 images split into 70% train, 15% validation, 15% test
   - 🔗 [Kaggle - Person Detection Dataset](https://kaggle.com/datasets/337dc7559ca05aecf7c76c2101a9a01c84261b8eee9d89c54021346c65759822)

2. **Customer Behavior Recognition Dataset**:
   - Self-recorded video clips: 215 segments annotated for four key actions — Pick, Put, Drop, Take
   - Train/test split: 80/20
   - Used to train a TimeSformer-based spatiotemporal model
   - 🔗 [Google Drive - Behavior Recognition Dataset](https://drive.google.com/file/d/1gA42h3jOp9KR_11Ih98gv_eL6ttzXJqN/view?usp=drive_link)

3. **Product Detection & Tracking Dataset**:
   - Self-recorded videos: 3,228 images covering 10 distinct product classes
   - 4,280 total annotations; split into 70% train, 10% validation, 20% test
   - Trained using the YOLOv8 framework
   - 🔗 [Kaggle - Product Detection Dataset](https://www.kaggle.com/datasets/sunyanshu/dataset-for-yolov8-product-detection/data)

## 👨‍💻 Authors & Contributions

- **Sun Yanshu**  
  - Annotated product dataset  
  - Fine-tuned YOLOv8 for product detection  
  - Optimized DeepSORT parameters for product tracking

- **Huang Yifei**  
  - Annotated person dataset  
  - Fine-tuned YOLOv8 for person detection  
  - Optimized DeepSORT parameters for person tracking

- **Xu Chi**  
  - Implemented action recognition using Optical Flow and TimeSformer  
  - Integrated all components into a unified system

- **Zhou Danying**  
  - Prepared behavior recognition dataset  
  - Modified and trained the TimeSformer model

