import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import transforms3d.quaternions as quat
import transforms3d.affines as affines
import json
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os

pkg_path = get_package_share_directory('wayp')
json_path = os.path.join(pkg_path, "resource", 'tag_map.json')

class MarkerTFPublisher(Node):
    def __init__(self):
        super().__init__('wayp')
        self.broadcaster = StaticTransformBroadcaster(self)
        self.timer = self.create_timer(0.5, self.publish_all_markers)

    def publish_all_markers(self):
        with open(json_path, 'r') as f:
            marker_data = json.load(f)

        transforms = []

        for marker_id, data in marker_data.items():
            try:
                position = data['position']
                rotation_matrix = np.array(data['rotation'])

                # Compose world → marker transform
                trans, rot_mat, _, _ = affines.decompose(
                    affines.compose(position, rotation_matrix, [1, 1, 1])
                )
                quaternion = quat.mat2quat(rot_mat)

                t = TransformStamped()
                t.header.stamp = rclpy.time.Time().to_msg()
                t.header.frame_id = 'world'                    # Parent frame
                t.child_frame_id = f'marker_{marker_id}'       # Marker child

                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = trans[2]
                t.transform.rotation.x = quaternion[1]
                t.transform.rotation.y = quaternion[2]
                t.transform.rotation.z = quaternion[3]
                t.transform.rotation.w = quaternion[0]

                transforms.append(t)
                self.get_logger().info(f"[TF] Publishing world → marker_{marker_id}")

            except Exception as e:
                self.get_logger().error(f"[TF ERROR] Failed for marker_{marker_id}: {repr(e)}")

        self.broadcaster.sendTransform(transforms)
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = MarkerTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
