# Use ROS Noetic as base image
FROM osrf/ros:noetic-desktop-full@sha256:f2063147b3a7b63c69306ade84b814b3ee1073cd4c6f94de528e55260a4f78f5

# Install necessary dependencies for DepthAI and network tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    udev \
    wget \
    git \
    cmake \
    libusb-1.0-0-dev \
    unzip \
    iputils-ping \
    net-tools \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built depthai to avoid compilation issues
RUN pip3 install depthai==2.22.0

# Set up ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create workspace directory
WORKDIR /root/codes

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Keep container running
