#!/bin/bash

xhost +local:docker
sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/curriculum:/workspace/camp \
  -v ~/ros2_ws:/workspace/ros2_ws \
  -v ~/isaac_ros_ws:/workspace/isaac_ros_ws \
  -v /opt/ros/humble:/opt/ros/humble:ro \
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  --name jetsoncamp \
  jetsoncamp:latest
