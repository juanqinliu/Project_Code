from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'yolo_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='YOLO object detection package for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detection_node = yolo_detection.nodes.detection_node:main',
        ],
    },
)
