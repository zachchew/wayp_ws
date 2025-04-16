from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wayp',
            executable='aruco_tf_node',
            name='aruco_tf_node',
            output='screen'
        ),
        Node(
            package='wayp',
            executable='marker_tf_node',
            name='marker_tf_node',
            output='screen'
        ),
        Node(
            package='wayp',
            executable='drone_localiser_node',
            name='drone_localiser_node',
            output='screen'
        ),
    ])
