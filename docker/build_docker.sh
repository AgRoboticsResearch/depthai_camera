#!/bin/bash

# Set the image name
IMAGE_NAME="depthai-ros-poe"
TAG="latest"

# Build the Docker image
echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
docker build -t ${IMAGE_NAME}:${TAG} .

echo "Build complete!"
