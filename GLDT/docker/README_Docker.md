# GLDT Docker Guide

## Environment Summary
- Base image: `nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04`
- CUDA: `11.8`
- Python: `3.8.10`
- TensorRT: `8.4.1.5`
- PyTorch: `1.12.0+cu118`
- ROS2: `foxy`
- GStreamer
- FFmpeg

## Build

```bash
cd ./GLDT
docker build -f docker/Dockerfile -t gldt:v1 .
```

## Run
```bash
docker run -it --net=host --ipc=host --name gldt --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  -v /home/ljq/Share:/home/share \
  -v /home/ljq/Share/GLDT:/home/developer/workspace \
  -w /home/developer/workspace \
  gldt:v1 bash
```

## ROS2 workspace
```bash
source /opt/ros/foxy/setup.bash
cd /home/developer/workspace

colcon build
source install/setup.bash
```

## Environment checks
```bash
# PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# TensorRT
python3 -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"

# GStreamer
gst-inspect-1.0 --version

# FFmpeg
ffmpeg -version

# ROS2 distro
echo $ROS_DISTRO
```
