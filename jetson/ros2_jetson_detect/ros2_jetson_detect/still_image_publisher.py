#!/usr/bin/env python3
"""
still_image_publisher.py — publish one image file repeatedly as sensor_msgs/Image.

A reliable alternative to ROS's `image_publisher` for a SINGLE still image
(`image_publisher` uses OpenCV VideoCapture, which treats a lone `.jpg` as an
image *sequence* and fails). Uses cv2.imread + cv_bridge.

  ros2 run ros2_jetson_detect still_image_publisher --ros-args -p image:=/path/to.jpg
"""
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class StillImagePublisher(Node):
    def __init__(self):
        super().__init__('still_image_publisher')
        self.declare_parameter('image', '/Developer/models/bus.jpg')
        self.declare_parameter('topic', '/image_raw')
        self.declare_parameter('rate', 2.0)

        path = self.get_parameter('image').value
        topic = self.get_parameter('topic').value
        rate = float(self.get_parameter('rate').value)

        self.frame = cv2.imread(path)
        if self.frame is None:
            raise RuntimeError(f"could not read image: {path}")
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"Publishing '{path}' on '{topic}' at {rate} Hz")

    def _tick(self):
        msg = self.bridge.cv2_to_imgmsg(self.frame, encoding='bgr8')
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StillImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
