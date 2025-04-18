#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
import math
import time
import transforms3d as tf3d

class GazeboControlNode(Node):
    def __init__(self):
        super().__init__('gazebo_control_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Initialize drone state variables
        self.drone = None
        self.offboard_started = False
        self.current_position = [0.0, 0.0, 0.0]  # [North, East, Down]
        self.current_orientation = 0.0  # Yaw in radians
        self.target_position = None
        self.detected_markers = {}
        self.flying_to_marker = False
        self.target_marker_id = None
        
        # ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.3  # Size of ArUco marker in meters
        
        # Subscribe to camera image and camera info
        self.create_subscription(
            Image,
            '/camera',  # This will be remapped from the Gazebo topic
            self.image_callback,
            10
        )
        
        self.create_subscription(
            CameraInfo,
            '/camera_info',  # This will be remapped from the Gazebo topic
            self.camera_info_callback,
            10
        )
        
        # Create a publisher for visualization
        self.marker_pose_pub = self.create_publisher(
            PoseStamped,
            '/detected_marker_pose',
            10
        )
        
        # Start the drone control loop
        asyncio.ensure_future(self.run())
        
        self.get_logger().info('Gazebo control node initialized')
    
    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Camera calibration parameters received')
    
    def image_callback(self, msg):
        """Process incoming camera images and detect ArUco markers"""
        if self.camera_matrix is None:
            self.get_logger().warn('Camera calibration not yet received')
            return
        
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                cv_image, self.aruco_dict, parameters=self.aruco_params
            )
            
            # Clear previous detections
            self.detected_markers = {}
            
            if ids is not None:
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                for i, marker_id in enumerate(ids.flatten()):
                    # Get marker position in camera frame
                    tvec = tvecs[i][0]
                    rvec = rvecs[i][0]
                    
                    # Convert rotation vector to matrix
                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Store marker position and orientation
                    self.detected_markers[marker_id] = {
                        'position': tvec,
                        'rotation': rot_matrix
                    }
                    
                    # Publish marker pose for visualization
                    marker_pose = PoseStamped()
                    marker_pose.header.stamp = self.get_clock().now().to_msg()
                    marker_pose.header.frame_id = 'camera_frame'
                    marker_pose.pose.position.x = tvec[0]
                    marker_pose.pose.position.y = tvec[1]
                    marker_pose.pose.position.z = tvec[2]
                    
                    # Convert rotation matrix to quaternion
                    quat = self.rotation_matrix_to_quaternion(rot_matrix)
                    marker_pose.pose.orientation.x = quat[0]
                    marker_pose.pose.orientation.y = quat[1]
                    marker_pose.pose.orientation.z = quat[2]
                    marker_pose.pose.orientation.w = quat[3]
                    
                    self.marker_pose_pub.publish(marker_pose)
                    
                    self.get_logger().info(f'Detected marker {marker_id} at position {tvec}')
                
                # If we're not already flying to a marker, select the first detected one
                if not self.flying_to_marker and len(self.detected_markers) > 0:
                    self.target_marker_id = list(self.detected_markers.keys())[0]
                    self.flying_to_marker = True
                    self.get_logger().info(f'Setting target to marker {self.target_marker_id}')
            
            else:
                if self.flying_to_marker:
                    self.get_logger().info('Lost sight of markers')
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """Convert a rotation matrix to quaternion"""
        return tf3d.quaternions.mat2quat(rotation_matrix)
    
    async def run(self):
        """Main drone control loop"""
        # Connect to the drone
        self.drone = System()
        await self.drone.connect(system_address="udp://:14540")
        
        self.get_logger().info("Waiting for drone connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info("Drone connected!")
                break
        
        # Wait for position estimate
        self.get_logger().info("Waiting for position estimate...")
        async for health in self.drone.telemetry.health():
            if health.is_local_position_ok:
                self.get_logger().info("Position estimate OK")
                break
        
        # Start position updates
        asyncio.ensure_future(self.position_update_loop())
        
        # Arm the drone
        self.get_logger().info("Arming drone...")
        await self.drone.action.arm()
        
        # Take off to 2.5m height
        self.get_logger().info("Taking off to 2.5m...")
        
        # Set initial setpoint before starting offboard mode
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, -2.5, 0.0)  # -2.5m in NED frame is 2.5m above ground
        )
        
        # Start offboard mode
        try:
            await self.drone.offboard.start()
            self.offboard_started = True
            self.get_logger().info("Offboard mode started")
        except OffboardError as error:
            self.get_logger().error(f"Starting offboard mode failed with error: {error}")
            await self.drone.action.land()
            return
        
        # Wait for takeoff to complete
        for _ in range(10):  # Wait for approximately 5 seconds
            await asyncio.sleep(0.5)
        
        # Main control loop
        while True:
            if self.flying_to_marker and self.target_marker_id is not None:
                if self.target_marker_id in self.detected_markers:
                    await self.fly_to_marker(self.target_marker_id)
                else:
                    # If we lost sight of the target marker, hover in place
                    self.get_logger().info("Target marker not currently visible")
                    await self.hover()
            else:
                # If no marker is targeted, just hover
                await self.hover()
            
            await asyncio.sleep(0.1)
    
    async def position_update_loop(self):
        """Update current position from telemetry"""
        async for position in self.drone.telemetry.position():
            self.current_position = [position.north_m, position.east_m, -position.relative_altitude_m]
        
        async for angle in self.drone.telemetry.attitude_euler():
            self.current_orientation = angle.yaw_deg * math.pi / 180.0
    
    async def hover(self):
        """Command the drone to hover at current position"""
        if self.offboard_started:
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(
                    self.current_position[0],
                    self.current_position[1],
                    self.current_position[2],
                    self.current_orientation
                )
            )
    
    async def fly_to_marker(self, marker_id):
        """Fly to position 1m in front of the marker"""
        if marker_id not in self.detected_markers:
            return
        
        marker_data = self.detected_markers[marker_id]
        marker_pos = marker_data['position']
        marker_rot = marker_data['rotation']
        
        # Calculate position 1m in front of the marker
        # The marker's z-axis points outward from the marker
        marker_normal = marker_rot[:, 2]  # Third column is z-axis
        
        # Calculate target position 1m away from marker along its normal
        target_pos_camera_frame = marker_pos - marker_normal * 1.0
        
        # Transform from camera frame to drone body frame
        # This is a simplified transformation assuming camera is aligned with drone
        # In a real system, you would need the exact camera-to-drone transform
        target_pos_body_frame = [
            target_pos_camera_frame[2],   # X in camera is Z in body
            -target_pos_camera_frame[0],  # Y in camera is -X in body
            -target_pos_camera_frame[1]   # Z in camera is -Y in body
        ]
        
        # Transform from body frame to NED frame
        # This requires the current drone orientation
        cos_yaw = math.cos(self.current_orientation)
        sin_yaw = math.sin(self.current_orientation)
        
        target_pos_ned = [
            self.current_position[0] + target_pos_body_frame[0] * cos_yaw - target_pos_body_frame[1] * sin_yaw,
            self.current_position[1] + target_pos_body_frame[0] * sin_yaw + target_pos_body_frame[1] * cos_yaw,
            self.current_position[2] + target_pos_body_frame[2]
        ]
        
        # Calculate yaw to face the marker
        dx = marker_pos[0]
        dy = marker_pos[1]
        target_yaw = math.atan2(dy, dx)
        
        # Command the drone to move to the target position
        self.get_logger().info(f"Flying to position in front of marker {marker_id}")
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(
                target_pos_ned[0],
                target_pos_ned[1],
                target_pos_ned[2],
                target_yaw
            )
        )

def main(args=None):
    rclpy.init(args=args)
    
    # Create the ROS2 node
    gazebo_control_node = GazeboControlNode()
    
    # Set up the event loop
    loop = asyncio.get_event_loop()
    
    try:
        # Run the asyncio event loop and ROS2 spin in parallel
        loop.run_until_complete(
            asyncio.gather(
                rclpy.spin(gazebo_control_node),
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        rclpy.shutdown()

if __name__ == '__main__':
    main()
