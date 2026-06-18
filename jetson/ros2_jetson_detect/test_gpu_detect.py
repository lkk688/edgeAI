#!/usr/bin/env python3
"""Self-contained end-to-end check of the detector node (no external publisher).

Creates the JetsonDetectorNode (yolo/cuda), publishes one image on /image_raw,
and prints the vision_msgs/Detection2DArray it publishes back. Run inside the
ROS 2 + PyTorch container:  python3 test_gpu_detect.py /Developer/models/bus.jpg
"""
import sys
import time

HERE = '/Developer/edgeAI/jetson/ros2_jetson_detect/ros2_jetson_detect'
sys.path.insert(0, HERE)
sys.path.insert(0, '/Developer/edgeAI/jetson')

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from detector_node import JetsonDetectorNode

img_path = sys.argv[1] if len(sys.argv) > 1 else '/Developer/models/bus.jpg'

rclpy.init()
det = JetsonDetectorNode()                      # defaults: model=yolo, device=cuda
bridge = CvBridge()
frame = cv2.imread(img_path)
assert frame is not None, f"could not read {img_path}"

tester = Node('tester')
pub = tester.create_publisher(Image, '/image_raw', qos_profile_sensor_data)
got = []
tester.create_subscription(Detection2DArray, '/jetson_detector/detections', lambda m: got.append(m), 10)

exe = SingleThreadedExecutor()
exe.add_node(det)
exe.add_node(tester)

msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
t0 = time.time()
while time.time() - t0 < 20 and not got:
    pub.publish(msg)
    exe.spin_once(timeout_sec=0.3)

if got:
    arr = got[-1]
    print(f"\nRECEIVED Detection2DArray with {len(arr.detections)} detections:")
    for d in arr.detections:
        h = d.results[0].hypothesis
        b = d.bbox
        print(f"  {h.class_id:12s} {h.score:.2f}  center=({b.center.position.x:.0f},{b.center.position.y:.0f}) "
              f"size=({b.size_x:.0f}x{b.size_y:.0f})")
else:
    print("\nNO detections received")

rclpy.shutdown()
