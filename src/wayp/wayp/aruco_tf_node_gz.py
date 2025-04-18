#!/usr/bin/env python3
# Gazebo camera capture
# aruco marker pose estimation
# publishes tf for each detected marker

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

# ArUco config
ARUCO_DICT = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
}
aruco_dictionary_name = "DICT_5X5_50"
aruco_marker_side_length = 0.3

class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.br = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.visible_ids_pub = self.create_publisher(Int32MultiArray, '/visible_marker_ids', 10)
        
        # Initialize camera parameters (will be updated from camera_info)
        self.mtx = None
        self.dst = None
        
        # Set up ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Create CvBridge
        self.bridge = CvBridge()
        
        # Create subscriptions to camera topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10)
            
        self.camera_sub = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10)
            
        self.get_logger().info("Aruco TF publisher node started. Waiting for camera data...")

    def camera_info_callback(self, msg):
        """Callback for camera info messages"""
        if self.mtx is None:
            # Extract camera matrix and distortion coefficients
            self.mtx = np.array(msg.k).reshape(3, 3)
            self.dst = np.array(msg.d)
            self.get_logger().info("Camera calibration parameters received")

    def image_callback(self, msg):
        """Callback for camera image messages"""
        if self.mtx is None:
            self.get_logger().info("Waiting for camera calibration parameters...")
            return
            
        try:
            # Convert ROS Image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info("Frame received successfully")
            
            # Process the frame
            self.process_frame(frame, msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {str(e)}")

    def process_frame(self, frame, timestamp):
        """Process a camera frame to detect ArUco markers"""
        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None:
            self.get_logger().info(f"Detected ArUco markers: {ids.flatten()}")

            # Publish visible marker IDs
            id_msg = Int32MultiArray()
            id_msg.data = ids.flatten().tolist()
            self.visible_ids_pub.publish(id_msg)

            # Estimate pose of markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, aruco_marker_side_length, self.mtx, self.dst
            )
            
            # Draw detected markers and axes
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                # Original transform (camera_frame → marker_X)
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]

                # Convert rotation vector to matrix and then to quaternion
                rot_matrix, _ = cv2.Rodrigues(rvec)
                quat = self.rotation_matrix_to_quaternion(rot_matrix)

                # Create and publish the INVERSE transform (marker_X → camera_frame)
                t = TransformStamped()
                t.header.stamp = timestamp
                t.header.frame_id = f'marker_{marker_id}'  # Now marker is the parent
                t.child_frame_id = 'camera_frame'          # Camera is the child

                # Invert the transformation
                # For rotation: transpose the rotation matrix
                inv_rot_matrix = rot_matrix.transpose()
                inv_quat = self.rotation_matrix_to_quaternion(inv_rot_matrix)

                # For translation: rotate the negative of the translation vector by the inverse rotation
                inv_tvec = -np.dot(inv_rot_matrix, tvec)

                # Set the inverted translation
                t.transform.translation.x = inv_tvec[0]
                t.transform.translation.y = inv_tvec[1]
                t.transform.translation.z = inv_tvec[2]

                # Set the inverted rotation
                t.transform.rotation.x = inv_quat[0]
                t.transform.rotation.y = inv_quat[1]
                t.transform.rotation.z = inv_quat[2]
                t.transform.rotation.w = inv_quat[3]

                # Broadcast the transform
                self.br.sendTransform(t)
                self.get_logger().info(f"Published TF: marker_{marker_id} → camera_frame")
                
                # Create and publish PoseStamped message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = timestamp
                pose_msg.header.frame_id = 'camera_frame'
                
                # Set position
                pose_msg.pose.position.x = tvec[0]
                pose_msg.pose.position.y = tvec[1]
                pose_msg.pose.position.z = tvec[2]
                
                # Set orientation (quaternion)
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
                
                # Publish the pose
                self.pose_pub.publish(pose_msg)
                self.get_logger().info(f"Published pose for marker_{marker_id} on /aruco_marker_pose")
        else:
            self.get_logger().info("No markers detected in this frame.")

        # Display the frame (optional - you might want to disable this in headless environments)
        cv2.imshow("ArUco Detection", frame)
        cv2.waitKey(1)

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """Convert a rotation matrix to quaternion."""
        try:
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(rotation_matrix)
            return r.as_quat()  # Returns x, y, z, w
        except ImportError:
            # Fallback method if scipy is not available
            q = np.zeros(4)
            trace = np.trace(rotation_matrix)
            
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                q[3] = 0.25 / s
                q[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
                q[1] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
                q[2] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
            else:
                if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                    q[3] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                    q[0] = 0.25 * s
                    q[1] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                    q[2] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                    q[3] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                    q[0] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                    q[1] = 0.25 * s
                    q[2] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                    q[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                    q[0] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                    q[1] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                    q[2] = 0.25 * s
            return q

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    print("[INFO] ArUco TF node started.")
    rclpy.init(args=args)
    node = ArucoTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



"""
# /usr/bin/env python3
# IP webcam rtsp capture
# aruco marker pose estimation
# publishes tf for each detected marker

import rclpy
from rclpy.node import Node
import time
import cv2
import numpy as np
import math
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# import os

# script_dir = os.path.dirname(os.path.abspath(__file__))
# camera_calibration_parameters_filename = os.path.join(script_dir, 'calibration.yaml')
# print(f"[INFO] Loading calibration from: {camera_calibration_parameters_filename}")

# ArUco config
ARUCO_DICT = {
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
}
aruco_dictionary_name = "DICT_5X5_50"
aruco_marker_side_length = 0.105
RTSP_STREAM = "https://192.168.1.93:8080/video"

# def load_camera_calibration(filename=camera_calibration_parameters_filename):
#     cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
#     mtx = cv_file.getNode('K').mat()
#     cv_file.getNode('D').mat()
#     dst = cv_file.release()
#     return mtx, dst


class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.br = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.visible_ids_pub = self.create_publisher(Int32MultiArray, '/visible_marker_ids', 10)

        mtx = np.array(
                [[1.09531983e+03, 0.00000000e+00, 4.16403376e+02],
                [0.00000000e+00, 1.07790562e+03, 3.53302770e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                )

        dst = np.array(
            [-6.67526108e-01, 2.12827140e+00, 2.15226318e-03, -2.13055254e-03, -3.08210370e+00]
            )

        # Load camera calibration
        # self.mtx, self.dst = load_camera_calibration(camera_calibration_parameters_filename)
        self.mtx, self.dst = mtx, dst

        # Set up ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Start the video stream
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(RTSP_STREAM)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open RTSP stream")
            return

        self.get_logger().info("Aruco TF publisher node started.")
        self.timer = self.create_timer(0.2, self.timer_callback)  # ~5 FPS

    def timer_callback(self):
        import time  # Ensure this is at the top
        start_time = time.time()  # Timestamp just before reading the frame

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[WARN] Failed to grab frame from video stream.")
        else:
            print("[INFO] Frame captured successfully.")

        latency_ms = (time.time() - start_time) * 1000
        self.get_logger().info(f"Estimated frame-to-detection latency: {latency_ms:.2f} ms")

        # ✅ Step 1: Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # ✅ Step 2: Use gray for detection
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is not None:
            print(f"[INFO] Detected ArUco markers: {ids.flatten()}")

            # ➕ Publish visible marker IDs
            id_msg = Int32MultiArray()
            id_msg.data = ids.flatten().tolist()
            self.visible_ids_pub.publish(id_msg)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, aruco_marker_side_length, self.mtx, self.dst
            )
            # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # for i, marker_id in enumerate(ids):
            #     print(f"[POSE] Marker ID {marker_id}: tvec = {tvecs[i][0]}, rvec = {rvecs[i][0]}")

            for i, marker_id in enumerate(ids.flatten()):
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]
                cv2.drawFrameAxes(frame, self.mtx, self.dst, rvec, tvec, 0.05)

                # Rotation: Rodrigues -> matrix -> quaternion
                rotation_matrix = cv2.Rodrigues(rvec)[0]
                quat = R.from_matrix(rotation_matrix).as_quat()

                # Create TransformStamped message
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = f'marker_{marker_id}'
                t.child_frame_id = 'camera_frame'

                t.transform.translation.x = tvec[0]
                t.transform.translation.y = tvec[1]
                t.transform.translation.z = tvec[2]

                t.transform.rotation.x = quat[0]
                t.transform.rotation.y = quat[1]
                t.transform.rotation.z = quat[2]
                t.transform.rotation.w = quat[3]

                # Broadcast to /tf
                self.br.sendTransform(t)

                self.get_logger().info(f"Published TF for marker_{marker_id}")
                self.get_logger().info(f"Publishing marker_{marker_id} relative to camera_frame")

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'camera_frame'
                pose_msg.pose.position.x = tvec[0]
                pose_msg.pose.position.y = tvec[1]
                pose_msg.pose.position.z = tvec[2]
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]

                self.pose_pub.publish(pose_msg)

        else:
            print("[INFO] No markers detected in this frame.")

        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quitting display window...")
            rclpy.shutdown()  # Optional: shut down ROS cleanly if 'q' is pressed

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    # print(script_dir)
    print("[INFO] ArUco TF node started.")
    rclpy.init(args=args)
    node = ArucoTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""