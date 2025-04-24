#!/bin/bash

# Set the image name
IMAGE_NAME="depthai-ros-poe"
TAG="latest"

# Allow X11 connections
xhost +

# Run the container
docker run -it --rm \
  --name depthai-ros-container \
  --privileged \
  --network=host \
  --gpus all \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  -v /dev/bus/usb:/dev/bus/usb \
  --device-cgroup-rule='c 189:* rmw' \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/${USER}/codes:/root/codes:rw \
  -v /home/${USER}/codes/.ssh:/root/.ssh:rw \
  ${IMAGE_NAME}:${TAG} "$@"

# Reset X11 permissions after container exits
xhost +
