# Use ROS Noetic as base image
FROM osrf/ros:noetic-desktop-full@sha256:f2063147b3a7b63c69306ade84b814b3ee1073cd4c6f94de528e55260a4f78f5

# Fix ROS GPG key issue by removing problematic repository
RUN rm -f /etc/apt/sources.list.d/ros1-latest.list

# Install a comprehensive set of system dependencies from both Dockerfiles
# This ensures all build tools and libraries are available for Python packages
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

# Install additional Python dependencies
RUN pip3 install \
    gradio_imageslider \
    gradio==4.29.0 \
    matplotlib \
    opencv-python \
    torch \
    torchvision


# Set up ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create and set the workspace directory
WORKDIR /root/codes

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Copy the entire project context into the working directory
# This includes install_requirements.py and requirements.txt
COPY . .

# Run the project's installation script to handle all Python dependencies.
# This replaces the previous manual 'pip install' commands.
RUN python3 install_requirements.py

# Keep container running for interactive use
CMD ["tail", "-f", "/dev/null"]

