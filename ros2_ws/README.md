# YOLO ROS2 Detection Package

A ROS2 object detection package based on YOLO, providing real-time object detection functionality.

## 📁 Project Structure

```
src/
├── yolo_detection/           # Main detection package
│   ├── launch/                 # Launch files
│   │   └── yolo_detection.launch.py
│   ├── nodes/                  # Node files
│   │   └── detection_node.py
│   ├── config/                 # Configuration files
│   │   └── default_params.yaml
│   └── test/                   # Test files
│       └── test_detection.py
├── yolo_msgs/                # Message package
│   └── msg/                    # Message definitions
│       ├── BoundingBox.msg
│       ├── Mask.msg
│       ├── Point2D.msg
│       ├── YoloInference.msg
│       └── YoloInferenceMsg.msg
└── README.md                   # Project documentation
```

## 🚀 Quick Start

### 1. Build Workspace

```bash
cd ./ros2_ws
colcon build
source install/setup.bash
```

### 2. Run Detection Node

#### Camera Mode
```bash
ros2 launch yolo_detection yolo_detection.launch.py input_type:=camera camera_id:=0

ros2 launch yolo_detection yolo_detection.launch.py input_type:=camera weight:=weights/yolo11s-p2.engine save_video:=true save_path:=results width:=1920 height:=1080 show_result:=true camera_id:=0 
```

#### Video File Mode
```bash
ros2 launch yolo_detection yolo_detection.launch.py input_type:=video input_path:=/path/to/video.mp4


ros2 launch yolo_detection yolo_detection.launch.py input_type:=video input_path:=/home/ljq/ros2_ws/DJI_20250922112849_0002_S.MP4 weight:=weights/yolo11s-p2.engine width:=1920 height:=1080 save_path:=results save_video:=false show_result:=true save_result:=true save_original_video:=false
```

#### Image File Mode
```bash
ros2 launch yolo_detection yolo_detection.launch.py input_type:=image input_path:=/path/to/image.jpg
```



## ⚙️ Parameters

### Model Parameters
- `weight`: Path to model weights file
- `device`: Computation device (cuda:0, cpu)
- `conf_threshold`: Confidence threshold
- `iou_threshold`: IoU threshold

### Input Parameters
- `input_type`: Input type (camera, video, image)
- `input_path`: Path to video/image file
- `camera_id`: Camera device ID
- `width`, `height`: Input resolution
- `fps`: Frame rate

### Display Parameters
- `display_width`, `display_height`: Display window dimensions

### Video Saving Parameters
- `save_video`: Whether to save the detction video
- `save_path`: Save path
- `show_result`: Whether to show detection result on the video
- `save_result`: Whether to save detection result to the file
- `save_original_video`: Whether to save original video

## 📡 Topics

### Published Topics
- `/yolo_inference`: Detection result messages
- `/yolo_result`: Images with detection boxes
