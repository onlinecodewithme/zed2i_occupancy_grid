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
        
        # IMPORTANT: Let the ZED node handle most of the frame transforms
        # Only add the map->odom transform which the ZED node needs but doesn't provide
        self.get_logger().info('Setting up minimal TF transforms to avoid conflicts with ZED node')
        
        # Map to odom transform is the only one we need to provide
        map_to_odom = self.create_transform('map', 'odom', 0.0, 0.0, 0.0)
        tf_transforms.append(map_to_odom)
        
        # Add a ground frame for visualization
        map_to_ground = self.create_transform('map', 'ground', 0.0, 0.0, -0.3)
        tf_transforms.append(map_to_ground)
        
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
