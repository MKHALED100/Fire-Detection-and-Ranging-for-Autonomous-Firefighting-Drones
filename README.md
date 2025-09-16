
# FDAR – Fire Detection and Ranging

A modular vision-based system for real-time fire, smoke, and human detection with 3D localization, designed for autonomous firefighting drones. The pipeline integrates RGB and thermal sensing, object detection, monocular depth estimation, and safety-aware targeting to support autonomous fire suppression.

## Features

- **Multi-Modal Perception**
  - Supports RGB-only, thermal-only, and RGB–thermal fused (RGT) detection.
  - Early-fusion pipeline with pixel-aligned RGB–thermal data.
- **Real-Time Detection**
  - Includes YOLO (v5–v12) and transformer-based detectors (RT-DETRv2, RF-DETR).
  - Multi-class detection of fire, smoke, and humans.
- **Depth Estimation**
  - Monocular depth prediction using Depth Anything V2 with ZoeDepth scaling.
  - 3D localization of all detections.
- **Fire Targeting & Safety**
  - Lightweight fire source localization inside bounding boxes.
  - Human proximity checks with `safe-to-suppress` flags.
  - Optional pseudo-3D bounding box visualization.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

```bash
git clone <>
cd FDAR
pip install -r requirements.txt
```
(Optional) Calibrate cameras if using a new RGB–thermal setup:
```bash
python load_camera_params.py
```

## Usage

Run the main script:

```bash
python run.py
```

### Configuration Options

You can modify the following parameters in `run.py`:

- **Input/Output**:
  - `source`: Path to input video file or webcam index (0 for default camera)
  - `output_path`: Path to output video file

- **Model Settings**:
  - `yolo_model`: YOLOv5–YOLOv12, RT-DETRv2, or RF-DETR
  - `depth_model`:  Depth Anything v2 model size: "small", "base", "large"    `OR`   Zoedepth ['train', 'infer', 'eval']

- **Detection Settings**:
  - `conf_threshold`: Confidence threshold for object detection
  - `iou_threshold`: IoU threshold for NMS
  - `classes`: Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes

- **Feature Toggles**:
  - `enable_tracking`: Enable object tracking
  - `enable_bev`: Enable Bird's Eye View visualization
  - `enable_pseudo_3d`: Enable 3D visualization

## Project Structure

```
FDAR/
│── run.py                # Main entry point
│── detection_model.py    # Multi-modal object detection logic
│── depth_model.py        # Monocular depth estimation + scaling
│── bbox3d_utils.py       # 3D bounding box and projection utilities
│── load_camera_params.py # Camera intrinsics & extrinsics loader
├── requirements.txt      # Dependencies
└── README.md             # This file


## How It Works

1. **Object Detection**: Detects fire, smoke, and humans from RGB, thermal, or fused RGT images.
2. **Depth Estimation**: Generates relative depth maps and scales them to metric depth.
3. **3D Box Estimation**: Combines 2D boxes with depth information to create 3D boxes
4. **Visualization**: Renders 3D boxes and bird's eye view for better spatial understanding

