# !/usr/bin/env python3
# IP webcam rtsp capture with reduced latency
# aruco marker pose estimation
# publishes tf for each detected marker

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# ArUco config
ARUCO_DICT = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
}
aruco_dictionary_name = "DICT_5X5_50"
aruco_marker_side_length = 0.105
RTSP_STREAM = "https://10.120.83.146:8080/video"  # Changed from https to rtsp

class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.br = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.visible_ids_pub = self.create_publisher(Int32MultiArray, '/visible_marker_ids', 10)

        # Camera calibration parameters
        self.mtx = np.array(
                [[1.09531983e+03, 0.00000000e+00, 4.16403376e+02],
                [0.00000000e+00, 1.07790562e+03, 3.53302770e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                )

        self.dst = np.array(
            [-6.67526108e-01, 2.12827140e+00, 2.15226318e-03, -2.13055254e-03, -3.08210370e+00]
            )

        # Set up ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Enable corner refinement for better accuracy
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Start the video stream with FFMPEG backend explicitly
        self.init_video_capture()

        self.get_logger().info("Aruco TF publisher node started.")
        self.timer = self.create_timer(0.2, self.timer_callback)  # ~5 FPS

    def init_video_capture(self):
        # Initialize video capture with FFMPEG backend and minimal buffering
        self.get_logger().info(f"Connecting to RTSP stream: {RTSP_STREAM}")
        self.cap = cv2.VideoCapture(RTSP_STREAM, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Additional settings to reduce latency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open RTSP stream")
            return False
        return True

    def get_latest_frame(self):
        #Get the latest frame by flushing the buffer
        if not self.cap.isOpened():
            if not self.init_video_capture():
                return False, None
                
        # Flush the buffer by grabbing frames without retrieving
        for _ in range(5):  # Grab several frames to flush buffer
            self.cap.grab()
            
        # Now retrieve only the latest frame
        ret, frame = self.cap.retrieve()
        
        if not ret or frame is None:
            self.get_logger().warn("Failed to retrieve frame, attempting to reconnect...")
            self.cap.release()
            if self.init_video_capture():
                ret, frame = self.cap.read()  # Try a standard read after reconnection
            
        return ret, frame

    def timer_callback(self):
        start_time = time.time()

        # Get the latest frame with reduced latency
        ret, frame = self.get_latest_frame()
        
        if not ret or frame is None:
            self.get_logger().warn("Failed to grab frame from video stream.")
            return
        else:
            self.get_logger().info("Frame captured successfully.")

        latency_ms = (time.time() - start_time) * 1000
        self.get_logger().info(f"Estimated frame-to-detection latency: {latency_ms:.2f} ms")

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
                
                # Draw coordinate axes
                cv2.drawFrameAxes(frame, self.mtx, self.dst, rvec, tvec, 0.05)

                # Convert rotation vector to matrix and then to quaternion
                rot_matrix, _ = cv2.Rodrigues(rvec)
                quat = self.rotation_matrix_to_quaternion(rot_matrix)

                # Create and publish the INVERSE transform (marker_X → camera_frame)
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = f'marker_{marker_id}'  # Now marker is the parent
                t.child_frame_id = 'camera_frame'          # Camera is the child

                # Invert the transformation
                # For rotation: transpose the rotation matrix (or negate the rotation vector)
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
                pose_msg.header.stamp = self.get_clock().now().to_msg()
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

        # Display the frame
        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quitting display window...")
            rclpy.shutdown()
            
        # Log total processing time
        total_time = (time.time() - start_time) * 1000
        self.get_logger().info(f"Total processing time: {total_time:.2f} ms")

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        # Convert a rotation matrix to quaternion.
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
        self.cap.release()
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
-------------------WORKING COPY THAT PUBLISHES ACCURATE /CAMERA_FRAME-------------------
------------------------------------MINIMAL LATENCY-------------------------------------
-------------------------------------DO NOT DELETE--------------------------------------

!/usr/bin/env python3
# IP webcam rtsp capture with reduced latency
# aruco marker pose estimation
# publishes tf for each detected marker

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# ArUco config
ARUCO_DICT = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
}
aruco_dictionary_name = "DICT_5X5_50"
aruco_marker_side_length = 0.105
RTSP_STREAM = "https://192.168.1.93:8080/video"  # Changed from https to rtsp

class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__('aruco_tf_publisher')
        self.br = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_marker_pose', 10)
        self.visible_ids_pub = self.create_publisher(Int32MultiArray, '/visible_marker_ids', 10)

        # Camera calibration parameters
        self.mtx = np.array(
                [[1.09531983e+03, 0.00000000e+00, 4.16403376e+02],
                [0.00000000e+00, 1.07790562e+03, 3.53302770e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                )

        self.dst = np.array(
            [-6.67526108e-01, 2.12827140e+00, 2.15226318e-03, -2.13055254e-03, -3.08210370e+00]
            )

        # Set up ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Enable corner refinement for better accuracy
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # Start the video stream with FFMPEG backend explicitly
        self.init_video_capture()

        self.get_logger().info("Aruco TF publisher node started.")
        self.timer = self.create_timer(0.2, self.timer_callback)  # ~5 FPS

    def init_video_capture(self):
        # Initialize video capture with FFMPEG backend and minimal buffering
        self.get_logger().info(f"Connecting to RTSP stream: {RTSP_STREAM}")
        self.cap = cv2.VideoCapture(RTSP_STREAM, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Additional settings to reduce latency
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open RTSP stream")
            return False
        return True

    def get_latest_frame(self):
        #Get the latest frame by flushing the buffer
        if not self.cap.isOpened():
            if not self.init_video_capture():
                return False, None
                
        # Flush the buffer by grabbing frames without retrieving
        for _ in range(5):  # Grab several frames to flush buffer
            self.cap.grab()
            
        # Now retrieve only the latest frame
        ret, frame = self.cap.retrieve()
        
        if not ret or frame is None:
            self.get_logger().warn("Failed to retrieve frame, attempting to reconnect...")
            self.cap.release()
            if self.init_video_capture():
                ret, frame = self.cap.read()  # Try a standard read after reconnection
            
        return ret, frame

    def timer_callback(self):
        start_time = time.time()

        # Get the latest frame with reduced latency
        ret, frame = self.get_latest_frame()
        
        if not ret or frame is None:
            self.get_logger().warn("Failed to grab frame from video stream.")
            return
        else:
            self.get_logger().info("Frame captured successfully.")

        latency_ms = (time.time() - start_time) * 1000
        self.get_logger().info(f"Estimated frame-to-detection latency: {latency_ms:.2f} ms")

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
                
                # Draw coordinate axes
                cv2.drawFrameAxes(frame, self.mtx, self.dst, rvec, tvec, 0.05)

                # Convert rotation vector to matrix and then to quaternion
                rot_matrix, _ = cv2.Rodrigues(rvec)
                quat = self.rotation_matrix_to_quaternion(rot_matrix)

                # Create and publish the INVERSE transform (marker_X → camera_frame)
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = f'marker_{marker_id}'  # Now marker is the parent
                t.child_frame_id = 'camera_frame'          # Camera is the child

                # Invert the transformation
                # For rotation: transpose the rotation matrix (or negate the rotation vector)
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
                pose_msg.header.stamp = self.get_clock().now().to_msg()
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

        # Display the frame
        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quitting display window...")
            rclpy.shutdown()
            
        # Log total processing time
        total_time = (time.time() - start_time) * 1000
        self.get_logger().info(f"Total processing time: {total_time:.2f} ms")

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        # Convert a rotation matrix to quaternion.
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
        self.cap.release()
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