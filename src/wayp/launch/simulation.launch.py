from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Environment variables
    env = {
        'PX4_GZ_WORLD': 'default',
        'PX4_GZ_MODEL': 'gz_x500_mono_cam',
        'PX4_GZ_SIM_RENDER_ENGINE': 'ogre',
    }
    
    # Launch PX4 SITL with Gazebo
    # px4_sitl_cmd = ExecuteProcess(
    #     cmd=[
    #         'bash', '-c',
    #         'cd ~/PX4-Autopilot && make px4_sitl gz_x500_mono_cam'
    #     ],
    #     env=env,
    #     output='screen'
    # )
    
    # Launch ROS-Gazebo image bridge for camera
    camera_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=[
            '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/Image'
        ],
        remappings=[
            ('/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/Image', '/camera')
        ],
        output='screen'
    )

    # Launch ROS-Gazebo bridge for camera info
    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
        ],
        remappings=[
            ('/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info', '/camera_info')
        ],
        output='screen'
    )
    
    # Create camera_frame to camera_link transform (camera_frame as parent)
    camera_frame_to_camera_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--frame-id', 'camera_frame',
            '--child-frame-id', 'x500_mono_cam_0/camera_link'
        ],
        output='screen'
    )
    
    # Create camera_link to drone base transform (camera_link as parent)
    camera_link_to_drone = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--frame-id', 'x500_mono_cam_0/camera_link',
            '--child-frame-id', 'x500_mono_cam_0/base_link'
        ],
        output='screen'
    )

    
    # Launch ArUco TF node
    aruco_tf_node = Node(
        package='wayp',
        executable='aruco_tf_node_gz',
        output='screen'
    )
    
    # Launch marker TF node
    marker_tf_node = Node(
        package='wayp',
        executable='marker_tf_node',
        output='screen'
    )
    
    # Launch drone localiser node
    drone_localiser_node = Node(
        package='wayp',
        executable='drone_localiser_node',
        output='screen'
    )
    
    # Launch drone controller node
    drone_control_node = Node(
        package='wayp',
        executable='drone_control_node',
        output='screen'
    )
    
    return LaunchDescription([
        # px4_sitl_cmd,
        camera_bridge,
        camera_info_bridge,
        # world_to_camera_frame,
        camera_frame_to_camera_link,
        camera_link_to_drone,
        aruco_tf_node,
        marker_tf_node,
        drone_localiser_node,
        drone_control_node,
    ])
