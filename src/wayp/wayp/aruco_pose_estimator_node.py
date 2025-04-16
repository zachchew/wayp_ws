import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import json
import os
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory

ARUCO_DICT = cv2.aruco.DICT_5X5_50
aruco_marker_length = 0.105  # in meters
RTSP_STREAM = "https://192.168.1.93:8080/video"

class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator_node')
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.marker_pose_pub = self.create_publisher(PoseStamped, '/detected_marker_pose', 10)

        self.mtx = np.array([
            [1.09531983e+03, 0.00000000e+00, 4.16403376e+02],
            [0.00000000e+00, 1.07790562e+03, 3.53302770e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.dst = np.array([-0.6675, 2.1282, 0.0021, -0.0021, -3.0821])

        pkg_path = get_package_share_directory('wayp')
        self.json_path = os.path.join(pkg_path, "resource", 'tag_map.json')

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.cap = cv2.VideoCapture(RTSP_STREAM)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open RTSP stream.")
            return

        self.timer = self.create_timer(0.2, self.timer_callback)  # 5Hz

    def load_marker_pose(self, marker_id):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        entry = data[str(marker_id)]
        pos = np.array(entry["position"])
        rot = np.array(entry["rotation"])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def build_static_transform(self, translation, euler_rpy):
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', euler_rpy).as_matrix()
        T[:3, 3] = translation
        return T

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No frame captured.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            self.get_logger().info("No markers detected.")
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, self.mtx, self.dst)

        marker_id = ids.flatten()[0]
        rvec = rvecs[0][0]
        tvec = tvecs[0][0]

        # OpenCV → ROS fix
        T_cv_to_ros = np.eye(4)
        T_cv_to_ros[:3, :3] = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        # Build T_marker_camera → ROS-convention
        R_cv, _ = cv2.Rodrigues(rvec)
        T_marker_camera = np.eye(4)
        T_marker_camera[:3, :3] = R_cv
        T_marker_camera[:3, 3] = tvec
        T_marker_camera_ros = T_cv_to_ros @ T_marker_camera @ np.linalg.inv(T_cv_to_ros)

        # Invert to get camera → marker
        T_camera_marker = np.linalg.inv(T_marker_camera_ros)

        # Static camera → drone
        T_camera_drone = self.build_static_transform([0.05, -0.03, 0.15], [-np.pi/2, 0.0, 0.0])
        T_drone_camera = np.linalg.inv(T_camera_drone)

        # Get T_world_marker from tag map
        T_world_marker = self.load_marker_pose(marker_id)

        # Compute T_world_drone = T_world_marker * T_camera_marker * T_drone_camera
        T_world_drone = T_world_marker @ T_camera_marker @ T_drone_camera

        # → Publish robot pose
        translation = T_world_drone[:3, 3]
        rotation = T_world_drone[:3, :3]
        quaternion = R.from_matrix(rotation).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        self.pose_pub.publish(pose_msg)

        # → Publish marker pose
        marker_pos = T_world_marker[:3, 3]
        marker_rot = T_world_marker[:3, :3]
        marker_quat = R.from_matrix(marker_rot).as_quat()

        marker_msg = PoseStamped()
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.header.frame_id = 'world'
        marker_msg.pose.position.x = marker_pos[0]
        marker_msg.pose.position.y = marker_pos[1]
        marker_msg.pose.position.z = marker_pos[2]
        marker_msg.pose.orientation.x = marker_quat[0]
        marker_msg.pose.orientation.y = marker_quat[1]
        marker_msg.pose.orientation.z = marker_quat[2]
        marker_msg.pose.orientation.w = marker_quat[3]
        self.marker_pose_pub.publish(marker_msg)

        self.get_logger().info(f"[POSE] Published /robot_pose and /detected_marker_pose from marker {marker_id}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
