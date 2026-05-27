from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ur3_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jackson Russell',
    maintainer_email='jackfruittt@users.noreply.github.com',
    description='YOLO object detection node for UR3e XR pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = ur3_perception.yolo_detector:main',
            'handeye_solver = ur3_perception.handeye_solver:main',
            'colour_button_detector = ur3_perception.colour_button_detector:main',
            'hsv_tuner = ur3_perception.hsv_tuner:main',
        ],
    },
)
