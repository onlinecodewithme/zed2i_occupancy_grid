#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class StaticTFPublisher(Node):
    """
    Node to publish static transforms needed for the occupancy grid to work.
    This is needed when no SLAM or localization is running to provide the transforms.
    """

    def __init__(self):
        super().__init__('static_tf_publisher')
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish necessary static transforms
        tf_transforms = []
        
        # IMPORTANT: We need to ensure a proper TF tree for occupancy grid mapping
        self.get_logger().info('Setting up TF transforms for occupancy grid')
        
        # Map to odom transform - critical connection for mapping
        map_to_odom = self.create_transform('map', 'odom', 0.0, 0.0, 0.0)
        tf_transforms.append(map_to_odom)
        
        # Add a ground frame for visualization
        map_to_ground = self.create_transform('map', 'ground', 0.0, 0.0, -0.3)
        tf_transforms.append(map_to_ground)
        
        # Add fallback transforms that might be missing
        # We need to ensure the connection from odom to camera frames exists
        # Only used if ZED node doesn't publish these
        odom_to_base = self.create_transform('odom', 'zed_camera_link', 0.0, 0.0, 0.0)
        tf_transforms.append(odom_to_base)
        
        # Camera base to left camera frame transform
        # This matches the ZED camera's internal calibration
        base_to_left = self.create_transform('zed_camera_link', 'zed_left_camera_frame', 0.01, -0.06, -0.015)
        tf_transforms.append(base_to_left)
        
        # Publish all transforms
        self.static_broadcaster.sendTransform(tf_transforms)
        self.get_logger().info('Published static transforms for occupancy grid')

    def create_transform(self, parent_frame, child_frame, x, y, z, rx=0.0, ry=0.0, rz=0.0, rw=1.0):
        """Create a transform between parent and child frames"""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = parent_frame
        tf.child_frame_id = child_frame
        
        # Set translation
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        
        # Set rotation (quaternion)
        tf.transform.rotation.x = rx
        tf.transform.rotation.y = ry
        tf.transform.rotation.z = rz
        tf.transform.rotation.w = rw
        
        return tf


def main(args=None):
    rclpy.init(args=args)
    node = StaticTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
