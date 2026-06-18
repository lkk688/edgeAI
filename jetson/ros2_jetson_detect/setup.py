import os
from glob import glob
from setuptools import setup

package_name = 'ros2_jetson_detect'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kaikai Liu',
    maintainer_email='kaikai.liu@sjsu.edu',
    description='ROS 2 node wrapping jetson_object_detection_toolkit.py.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node = ros2_jetson_detect.detector_node:main',
            'still_image_publisher = ros2_jetson_detect.still_image_publisher:main',
        ],
    },
)
