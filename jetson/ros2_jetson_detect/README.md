# ros2_jetson_detect

A minimal ROS 2 (Humble) package that wraps
[`jetson_object_detection_toolkit.py`](../jetson_object_detection_toolkit.py) as a node.

It subscribes to a camera/image topic, runs any toolkit detector on the Jetson
GPU (`yolo`, `faster-rcnn`, `maskrcnn`, `detr`, `owl-vit`, `grounding-dino`), and
publishes:

| Topic | Type | Description |
|---|---|---|
| `~/detections` | `vision_msgs/Detection2DArray` | boxes + class + score |
| `~/detections_image` | `sensor_msgs/Image` | annotated frame (view in `rqt_image_view`/RViz) |

## Parameters
`model` · `image_topic` (default `/image_raw`) · `confidence` · `iou` · `prompts`
(owl-vit/grounding-dino) · `device` (`cuda`/`cpu`) · `toolkit_path`
(default `/Developer/edgeAI/jetson`) · `publish_annotated`.

## Build & run (inside a ROS 2 + PyTorch container — see `docs/curriculum/05c_ros2_isaac_ros_jetson.md`)
```bash
mkdir -p ~/ros2_ws/src && ln -s /Developer/edgeAI/jetson/ros2_jetson_detect ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select ros2_jetson_detect && source install/setup.bash

# feed it an image and run the detector
ros2 launch ros2_jetson_detect detect.launch.py model:=yolo image:=/Developer/models/bus.jpg
# inspect:
ros2 topic echo /jetson_detector/detections --once
ros2 run rqt_image_view rqt_image_view /jetson_detector/detections_image
```

> The container must provide the toolkit's deps (torch, ultralytics, opencv,
> transformers) **and** ROS deps (`vision_msgs`, `cv_bridge`). See the tutorial.
