import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
import transforms3d.euler as euler
import transforms3d.quaternions as quat
import transforms3d.affines as affines
import numpy as np

class DroneLocaliser(Node):
    def __init__(self):
        super().__init__('drone_localiser')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)

        # Fixed transform: drone_frame → camera_frame
        self.T_drone_camera = self.build_static_transform_matrix(
            translation=[0.05, -0.03, 0.15],    
            rotation_euler=[-(np.pi)/2, 0.0, 0.0] # Roll, Pitch, Yaw
        )

        # Get inverse: camera → drone
        self.T_camera_drone = np.linalg.inv(self.T_drone_camera)
        
        # Define ENU to FRD transformation matrix
        # This transforms from ROS ENU to PX4 FRD coordinate system
        # ENU to FRD: X→Y, Y→X, Z→-Z with a 90-degree rotation about Z
        self.T_enu_to_frd = np.array([
            [0, 1, 0, 0],  # ENU Y → FRD X (forward)
            [1, 0, 0, 0],  # ENU X → FRD Y (right)
            [0, 0, -1, 0], # ENU Z → FRD -Z (down)
            [0, 0, 0, 1]
        ])

        self.marker_ids_to_check = [f'marker_{i}' for i in range(1, 5)]

    def build_static_transform_matrix(self, translation, rotation_euler):
        R = euler.euler2mat(*rotation_euler)
        return affines.compose(translation, R, [1, 1, 1])

    def tf_to_matrix(self, tf_msg: TransformStamped):
        t = tf_msg.transform.translation
        r = tf_msg.transform.rotation
        translation = [t.x, t.y, t.z]
        rotation = quat.quat2mat([r.w, r.x, r.y, r.z])
        return affines.compose(translation, rotation, [1, 1, 1])

    def timer_callback(self):
        now = rclpy.time.Time()
        chosen_marker_id = None

        # ✅ Find the first visible marker
        for marker_id in self.marker_ids_to_check:
            try:
                self.tf_buffer.lookup_transform('world', marker_id, now, timeout=rclpy.duration.Duration(seconds=0.05))
                chosen_marker_id = marker_id
                break
            except Exception:
                continue

        if not chosen_marker_id:
            self.get_logger().warn("No visible markers found.")
            return

        try:
            # T_world_marker
            tf_world_marker = self.tf_buffer.lookup_transform('world', chosen_marker_id, now)
            T_world_marker = self.tf_to_matrix(tf_world_marker)

            # T_marker_camera
            tf_marker_camera = self.tf_buffer.lookup_transform(chosen_marker_id, 'camera_frame', now)
            T_marker_camera = self.tf_to_matrix(tf_marker_camera)

            # T_camera_drone: fixed transform
            T_camera_drone = self.T_camera_drone

            # Calculate the drone pose in ENU world frame
            T_world_drone_enu = T_world_marker @ T_marker_camera @ T_camera_drone
            
            # Convert from ENU to FRD coordinate system for PX4
            T_world_drone_frd = T_world_drone_enu @ self.T_enu_to_frd
            
            # Extract position and orientation from the transformed matrix
            translation, rotation_matrix, _, _ = affines.decompose(T_world_drone_frd)
            quaternion = quat.mat2quat(rotation_matrix)

            # Create and publish the pose message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = translation[0]
            pose_msg.pose.position.y = translation[1]
            pose_msg.pose.position.z = translation[2]
            pose_msg.pose.orientation.x = quaternion[1]
            pose_msg.pose.orientation.y = quaternion[2]
            pose_msg.pose.orientation.z = quaternion[3]
            pose_msg.pose.orientation.w = quaternion[0]

            self.pose_pub.publish(pose_msg)

            self.get_logger().info(f"[POSE] Drone in world using {chosen_marker_id}:")
            self.get_logger().info(f"Position: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")
            self.get_logger().info(f"Orientation (FRD): qw={quaternion[0]:.3f}, qx={quaternion[1]:.3f}, qy={quaternion[2]:.3f}, qz={quaternion[3]:.3f}")

        except Exception as e:
            self.get_logger().warn(f"[WARN] Could not compute transform: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DroneLocaliser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


"""
----------------WORKING COPY THAT PUBLISHES ROS2 ENU FORMAT FOR /ROBOT_POSE----------------
---------------------------------------DO NOT DELETE---------------------------------------

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
import transforms3d.euler as euler
import transforms3d.quaternions as quat
import transforms3d.affines as affines
import numpy as np

class DroneLocaliser(Node):
    def __init__(self):
        super().__init__('drone_localiser')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)

        # Fixed transform: drone_frame → camera_frame
        self.T_drone_camera = self.build_static_transform_matrix(
            translation=[0.05, -0.03, 0.15],    
            rotation_euler=[-(np.pi)/2, 0.0, 0.0] # Roll, Pitch, Yaw
        )

        # Get inverse: camera → drone
        self.T_camera_drone = np.linalg.inv(self.T_drone_camera)

        self.marker_ids_to_check = [f'marker_{i}' for i in range(1, 5)]

    def build_static_transform_matrix(self, translation, rotation_euler):
        R = euler.euler2mat(*rotation_euler)
        return affines.compose(translation, R, [1, 1, 1])

    def tf_to_matrix(self, tf_msg: TransformStamped):
        t = tf_msg.transform.translation
        r = tf_msg.transform.rotation
        translation = [t.x, t.y, t.z]
        rotation = quat.quat2mat([r.w, r.x, r.y, r.z])
        return affines.compose(translation, rotation, [1, 1, 1])

    def timer_callback(self):
        now = rclpy.time.Time()
        chosen_marker_id = None

        # ✅ Find the first visible marker
        for marker_id in self.marker_ids_to_check:
            try:
                self.tf_buffer.lookup_transform('world', marker_id, now, timeout=rclpy.duration.Duration(seconds=0.05))
                chosen_marker_id = marker_id
                break
            except Exception:
                continue

        if not chosen_marker_id:
            self.get_logger().warn("No visible markers found.")
            return

        try:
            # T_world_marker
            tf_world_marker = self.tf_buffer.lookup_transform('world', chosen_marker_id, now)
            T_world_marker = self.tf_to_matrix(tf_world_marker)

            # T_marker_camera
            tf_marker_camera = self.tf_buffer.lookup_transform(chosen_marker_id, 'camera_frame', now)
            T_marker_camera = self.tf_to_matrix(tf_marker_camera)

            # T_camera_drone: fixed transform
            T_camera_drone = self.T_camera_drone

            # Final transform chain: T_world_drone = T_world_marker × T_marker_camera × T_camera_drone
            T_world_drone = T_world_marker @ T_marker_camera @ T_camera_drone

            translation, rotation_matrix, _, _ = affines.decompose(T_world_drone)
            quaternion = quat.mat2quat(rotation_matrix)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = translation[0]
            pose_msg.pose.position.y = translation[1]
            pose_msg.pose.position.z = translation[2]
            pose_msg.pose.orientation.x = quaternion[1]
            pose_msg.pose.orientation.y = quaternion[2]
            pose_msg.pose.orientation.z = quaternion[3]
            pose_msg.pose.orientation.w = quaternion[0]

            self.pose_pub.publish(pose_msg)

            self.get_logger().info(f"[POSE] Drone in world using {chosen_marker_id}:")
            self.get_logger().info(f"Position: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}")

        except Exception as e:
            self.get_logger().warn(f"[WARN] Could not compute transform: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DroneLocaliser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

"""