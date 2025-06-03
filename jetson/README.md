# Jetson Docker Environment (with ROS 2 + Isaac ROS)

This Docker image sets up a complete AI, Cybersecurity, and Robotics development environment for Jetson Orin Nano.

## 🔧 Build the Image

Base container image: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

```bash
git clone https://github.com/lkk688/edgeAI.git
cd edgeAI/jetson
sudo docker build -t sjsucyberai_jetson:latest .
```

## 🚀 Run the Container

```bash
bash run.sh
```

## 🧰 Includes:

- Python, CUDA, PyTorch
- OpenCV, Transformers, LangChain
- Cyber tools: tcpdump, nmap, ufw, fail2ban
- ROS 2 workspace support
- Isaac ROS workspace mount-ready
- JupyterLab (optional)

## 🤖 ROS 2 Workflow

```bash
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build
source install/setup.bash
ros2 run your_pkg your_node
```

## ⚙️ Isaac ROS Integration

Mount your `isaac_ros_ws` and GPU libraries. Most Isaac ROS demos will run natively inside this container using mounted hardware access.

## 📁 Volumes Mounted

- `~/ros2_ws` → `/workspace/ros2_ws`
- `~/isaac_ros_ws` → `/workspace/isaac_ros_ws`
- `./curriculum/` → `/workspace/camp`
