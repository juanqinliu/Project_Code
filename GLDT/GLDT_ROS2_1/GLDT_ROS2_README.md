# GLDT ROS2 Tracking and Detection System

A ROS2-integrated version of the GLDT (Global-Local Detection and Tracking) system that provides real-time video processing capabilities with ROS2 topic publishing.

## 🚀 Quick Start

### 1. Build the Project

```bash
cd /home/developer/workspace/GLDT_ROS2
source /opt/ros/foxy/setup.bash
colcon build
source install/setup.bash
```

### 2. Launch Options

#### Using Launch Script

```bash
# Basic usage
./run_ros2.sh

# Process camera device file
./run_ros2.sh -v /dev/video0

# Process single video file
./run_ros2.sh -v /path/to/video.mp4

# Process all videos in directory
./run_ros2.sh -v /path/to/videos/

```

## 📡 ROS2 Topics

### Published Topics

- `/gldt/tracking_result` (gldt_msgs/msg/TrackingResult)
  - Complete tracking and detection results
  - Contains detection information, tracking data, and performance statistics

- `/gldt/tracking_image` (sensor_msgs/msg/Image)
  - Visualization image with overlays
  - Includes bounding boxes, tracking IDs, and trajectories

### Topic Inspection

```bash
# List all topics
ros2 topic list

# View tracking results
ros2 topic echo /gldt/tracking_result

# View image topic
ros2 topic echo /gldt/tracking_image

# Check topic frequency
ros2 topic hz /gldt/tracking_result


```

## 🎯 Usage Examples

### Process Single Video and View Results

```bash
# 1. Start processing
./run_ros2.sh -v /path/to/test_video.mp4

# 2. View results in another terminal
ros2 topic echo /gldt/tracking_result

# 3. View visualization
ros2 run rqt_image_view rqt_image_view
```



