# CUDA 11.8 + cuDNN8 + Ubuntu 20.04 + TensorRT 8.5.2.2 + ffmpeg + GStreamer + ROS2 Foxy

ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Base OS, compilers, dev tools, Python3, OpenCV runtime, ffmpeg, GStreamer
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y universe \
    && add-apt-repository -y multiverse \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    gnupg \
    gnupg2 \
    curl \
    wget \
    git \
    nano \
    tzdata \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libsm6 \
    libxext6 \
    libopencv-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libgflags-dev \
    libfmt-dev \
    libgoogle-glog-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Optional: install extra GStreamer codecs if available (do not fail build)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    || true \
    && rm -rf /var/lib/apt/lists/*

# Install specific PyTorch wheels (CUDA 11.6) and dependencies
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install \
    torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Add missing runtime deps explicitly (avoid cache ambiguity)
RUN pip3 install --no-cache-dir typing_extensions packaging

# Install TensorRT 8.5.2.2 (CUDA 11.8) with all dependencies
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - \
    && curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer8=8.5.2-1+cuda11.8 \
    libnvinfer-dev=8.5.2-1+cuda11.8 \
    libnvinfer-plugin8=8.5.2-1+cuda11.8 \
    libnvinfer-plugin-dev=8.5.2-1+cuda11.8 \
    libnvparsers8=8.5.2-1+cuda11.8 \
    libnvparsers-dev=8.5.2-1+cuda11.8 \
    libnvonnxparsers8=8.5.2-1+cuda11.8 \
    libnvonnxparsers-dev=8.5.2-1+cuda11.8 \
    python3-libnvinfer=8.5.2-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*

# ROS2 Foxy (Ubuntu 20.04)
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends software-properties-common ca-certificates curl gnupg2 \
    && add-apt-repository -y universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list \
    && (apt-get update || apt-get update --fix-missing) \
    && (apt-get install -y --no-install-recommends \
         ros-foxy-desktop \
         python3-colcon-common-extensions \
         python3-rosdep \
         ros-foxy-cv-bridge \
         ros-foxy-image-transport \
         ros-foxy-vision-opencv \
         ros-foxy-rmw-fastrtps-cpp \
        || (apt-get update --fix-missing && apt-get install -y --no-install-recommends \
         ros-foxy-desktop \
         python3-colcon-common-extensions \
         python3-rosdep \
         ros-foxy-cv-bridge \
         ros-foxy-image-transport \
         ros-foxy-vision-opencv \
         ros-foxy-rmw-fastrtps-cpp)) \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update || true

# Create non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME/workspace

# Prepare optional ROS2 workspace structure (for deployment builds)
RUN mkdir -p /home/$USERNAME/workspace/src

# Copy ultralytics source code to container
COPY --chown=$USERNAME:$USERNAME ultralytics /home/$USERNAME/workspace/ultralytics

# Install ultralytics in editable mode with its dependencies
WORKDIR /home/$USERNAME/workspace/ultralytics
RUN pip3 install --user -e . \
    && pip3 install --user -r requirement.txt

# Add user's local bin to PATH
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Return to workspace root
WORKDIR /home/$USERNAME/workspace

# Entry
RUN echo '#!/usr/bin/env bash' > /home/$USERNAME/entrypoint.sh \
    && echo 'set -e' >> /home/$USERNAME/entrypoint.sh \
    && echo 'source /opt/ros/foxy/setup.bash || true' >> /home/$USERNAME/entrypoint.sh \
    && echo 'if [ -f /home/'"$USERNAME"'/workspace/install/setup.bash ]; then source /home/'"$USERNAME"'/workspace/install/setup.bash; fi' >> /home/$USERNAME/entrypoint.sh \
    && echo 'exec "$@"' >> /home/$USERNAME/entrypoint.sh \
    && chmod +x /home/$USERNAME/entrypoint.sh

ENTRYPOINT ["/home/developer/entrypoint.sh"]
CMD ["bash"] 