from setuptools import setup
from glob import glob
import os

package_name = 'wayp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'resource'), glob(os.path.join(package_name, 'tag_map.json')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zachary Chew',
    maintainer_email='zachary-chew@example.com',
    description='Publishes ArUco TFs and drone localisation using transforms.',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_tf_node = wayp.aruco_tf_node:main',
            'marker_tf_node = wayp.marker_tf_node:main',
            'drone_localiser_node = wayp.drone_localiser_node:main',
            'aruco_pose_estimator_node = wayp.aruco_pose_estimator_node:main',
            'aruco_tf_node_copy = wayp.aruco_tf_node_copy:main',
        ],
    },
)


"""
from setuptools import setup
import os
from glob import glob

package_name = 'wayp'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'resource'), glob(os.path.join(package_name, 'tag_map.json')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zachary-chew',
    maintainer_email='your@email.com',
    description='Aruco TF and localisation tools',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_tf_node = aruco_tf_publisher.aruco_tf_node:main',
            'marker_tf_node = aruco_tf_publisher.marker_tf_node:main',
            'drone_localiser_node = aruco_tf_publisher.drone_localiser_node:main',
        ],
    },
)
"""