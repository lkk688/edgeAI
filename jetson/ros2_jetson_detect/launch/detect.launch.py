"""Launch the Jetson detector node, optionally with an image_publisher source.

Example:
  ros2 launch ros2_jetson_detect detect.launch.py model:=yolo image:=/Developer/models/bus.jpg
  # then view:  ros2 run rqt_image_view rqt_image_view /jetson_detector/detections_image
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    model = LaunchConfiguration('model')
    confidence = LaunchConfiguration('confidence')
    image_topic = LaunchConfiguration('image_topic')
    image = LaunchConfiguration('image')          # optional file to publish on image_topic

    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='yolo'),
        DeclareLaunchArgument('confidence', default_value='0.25'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('image', default_value='',
                              description='If set, publish this image file on image_topic'),

        # Optional: publish a still image as sensor_msgs/Image (reliable for a
        # single file; ROS's image_publisher treats a lone .jpg as a sequence).
        Node(
            package='ros2_jetson_detect', executable='still_image_publisher',
            name='still_image_publisher', output='screen',
            parameters=[{'image': image, 'topic': image_topic}],
            condition=IfCondition(PythonExpression(["'", image, "' != ''"])),
        ),

        Node(
            package='ros2_jetson_detect', executable='detector_node',
            name='jetson_detector', output='screen',
            parameters=[{
                'model': model,
                'confidence': confidence,
                'image_topic': image_topic,
            }],
        ),
    ])
