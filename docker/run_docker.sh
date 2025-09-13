#!/bin/bash

# --- 设置变量 ---
IMAGE_NAME="depthai-ros-poe"
TAG="latest"
CONTAINER_NAME="depthai-ros-container"

# --- 步骤 1: 检查并移除已存在的同名容器 ---
# 使用 docker ps -a 来查找所有容器（包括已停止的）
if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "发现已存在的容器 '$CONTAINER_NAME'，正在强制移除..."
    # 使用 rm -f 比先 stop 再 rm 更快
    docker rm -f $CONTAINER_NAME > /dev/null
    echo "旧容器已移除。"
fi

# --- 步骤 2: 为GUI应用临时授权X server访问权限 ---
echo "为Docker容器临时授权X server访问权限..."
xhost +local:docker

# --- 清理函数 ---
# 此函数将在脚本退出时被调用，以确保容器被停止且权限被重置
cleanup() {
    echo "脚本退出，正在停止容器并撤销X server权限..."
    # 检查容器是否存在，以避免在容器启动失败时报错
    if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
        docker stop $CONTAINER_NAME > /dev/null
    fi
    xhost -local:docker
}
# 注册清理函数，在脚本退出时（无论正常或异常）执行
trap cleanup EXIT

# --- 步骤 3: 在后台启动一个“服务”容器 ---
echo "正在后台启动服务容器: $CONTAINER_NAME"
docker run -d \
  --rm \
  --name $CONTAINER_NAME \
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
  -v "/home/${USER}/.Xauthority:/root/.Xauthority:rw" \
  -v "/home/${USER}/codes:/root/codes:rw" \
  -v "/home/${USER}/codes/.ssh:/root/.ssh:rw" \
  ${IMAGE_NAME}:${TAG} tail -f /dev/null > /dev/null

echo "容器已启动。正在准备执行命令..."
sleep 2 # 短暂等待，确保容器内部服务完全就绪

# --- 步骤 4: 在容器内执行命令 ---
# 如果用户从命令行提供了参数 (例如, ./run_docker.sh python3 my_script.py), 就执行这些参数。
# 如果没有提供参数，就启动一个交互式的bash shell。
if [ $# -gt 0 ]; then
    echo "正在执行提供的命令: $@"
    docker exec -it --workdir /root/codes $CONTAINER_NAME "$@"
else
    echo "未提供命令，正在启动交互式 bash shell..."
    docker exec -it --workdir /root/codes $CONTAINER_NAME bash
fi

# 'trap' 会在脚本执行到这里或被中断时，自动调用 cleanup 函数


