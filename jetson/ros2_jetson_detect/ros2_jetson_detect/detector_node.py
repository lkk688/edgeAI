#!/usr/bin/env python3
"""
detector_node.py — ROS 2 node that wraps jetson_object_detection_toolkit.py.

Subscribes to a sensor_msgs/Image topic, runs the chosen detector from the
toolkit (yolo | faster-rcnn | maskrcnn | detr | owl-vit | grounding-dino) on the
Jetson GPU, and publishes:
  * ~/detections        vision_msgs/Detection2DArray   (boxes + class + score)
  * ~/detections_image  sensor_msgs/Image              (annotated frame, for RViz/rqt)

Parameters (ros2 run ... --ros-args -p name:=value):
  model             str    detector to use                         (default: yolo)
  image_topic       str    input image topic                       (default: /image_raw)
  confidence        float  confidence / box threshold              (default: 0.25)
  iou               float  NMS IoU threshold (yolo)                (default: 0.45)
  prompts           str    comma-separated text prompts (owl-vit / grounding-dino)
  device            str    cuda | cpu                              (default: cuda)
  toolkit_path      str    dir with jetson_object_detection_toolkit.py
                           (default: /Developer/edgeAI/jetson)
  publish_annotated bool   also publish the annotated image        (default: true)

Tested against ROS 2 Humble (the jetson-containers / Isaac ROS default).
"""
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge


class JetsonDetectorNode(Node):
    def __init__(self):
        super().__init__('jetson_detector')

        self.declare_parameter('model', 'yolo')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('prompts', 'person,car,dog')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('toolkit_path', '/Developer/edgeAI/jetson')
        self.declare_parameter('publish_annotated', True)

        self.model_type = self.get_parameter('model').value
        image_topic = self.get_parameter('image_topic').value
        self.confidence = float(self.get_parameter('confidence').value)
        self.iou = float(self.get_parameter('iou').value)
        self.prompts = self.get_parameter('prompts').value
        device = self.get_parameter('device').value
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        toolkit_path = self.get_parameter('toolkit_path').value

        # Reuse the EXACT detectors from the curriculum toolkit.
        if toolkit_path and toolkit_path not in sys.path:
            sys.path.insert(0, toolkit_path)
        from jetson_object_detection_toolkit import ObjectDetectionToolkit

        kwargs = {}
        if self.model_type == 'yolo':
            kwargs['model_path'] = 'yolov8n.pt'
        self.get_logger().info(f"Loading detector '{self.model_type}' on {device} ...")
        self.toolkit = ObjectDetectionToolkit(self.model_type, device, **kwargs)
        self.get_logger().info('Detector ready.')

        self.bridge = CvBridge()
        self.det_pub = self.create_publisher(Detection2DArray, '~/detections', 10)
        self.img_pub = self.create_publisher(Image, '~/detections_image', 10) \
            if self.publish_annotated else None
        # sensor-data QoS (best-effort) matches typical camera publishers
        # (image_publisher, usb_cam, Isaac ROS) and also accepts reliable ones.
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribed to '{image_topic}', publishing '~/detections'.")

    def _detect_kwargs(self):
        m = self.model_type
        if m == 'yolo':
            return {'conf_threshold': self.confidence, 'iou_threshold': self.iou}
        if m == 'owl-vit':
            return {'confidence_threshold': self.confidence,
                    'text_prompts': [p.strip() for p in self.prompts.split(',')]}
        if m == 'grounding-dino':
            return {'box_threshold': self.confidence, 'text_prompt': self.prompts}
        return {'confidence_threshold': self.confidence}

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        results = self.toolkit.detect(frame, **self._detect_kwargs())

        arr = Detection2DArray()
        arr.header = msg.header
        for box, score, name in zip(results['boxes'], results['scores'], results['class_names']):
            x1, y1, x2, y2 = (float(v) for v in box)
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2.0   # vision_msgs Humble layout
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(name)
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)
            arr.detections.append(det)
        self.det_pub.publish(arr)

        if self.img_pub is not None:
            annotated = self.toolkit.visualize_results(frame, results)
            out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header = msg.header
            self.img_pub.publish(out)

        self.get_logger().info(
            f"{len(arr.detections)} detections "
            f"({results.get('inference_time', 0.0):.0f} ms)")


def main(args=None):
    rclpy.init(args=args)
    node = JetsonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
