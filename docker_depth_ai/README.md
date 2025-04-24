# Docker for robot development 
Ubuntu 22.04  
ROS Noetic  
CUDA Support :white_check_mark:

```bash
# Build docker 
./build_docker.sh
```
Create a folder codes under host machine home directory

#### If you want CUDA
[Install CUDA container-toolkit] https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```bash
# reload cuda
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
```bash
# Run docker with display 
xhost +
docker run -it --net=host --gpus all \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/${USER}/codes":"/codes":rw  \
    robot_ros:noetic \
    bash

```
