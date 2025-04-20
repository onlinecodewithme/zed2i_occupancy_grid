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
        
        # Map to odom transform
        map_to_odom = self.create_transform('map', 'odom', 0.0, 0.0, 0.0)
        tf_transforms.append(map_to_odom)
        
        # Direct map to camera transforms to ensure occupancy grid works
        map_to_zed_center = self.create_transform('map', 'zed_camera_center', 0.0, 0.0, 0.2)
        tf_transforms.append(map_to_zed_center)
        
        map_to_zed_left = self.create_transform('map', 'zed_left_camera_frame', 0.0, -0.06, 0.2)
        tf_transforms.append(map_to_zed_left)
        
        map_to_zed2i_center = self.create_transform('map', 'zed2i_camera_center', 0.0, 0.0, 0.2)
        tf_transforms.append(map_to_zed2i_center)
        
        map_to_zed2i_left = self.create_transform('map', 'zed2i_left_camera_frame', 0.0, -0.06, 0.2)
        tf_transforms.append(map_to_zed2i_left)
        
        # Directly connect zed2i frames to odom if needed
        # Create both directions of transforms to ensure connectivity
        odom_to_zed2i_center = self.create_transform('odom', 'zed2i_camera_center', 0.0, 0.0, 0.2)
        tf_transforms.append(odom_to_zed2i_center)
        
        zed2i_center_to_zed2i_left = self.create_transform('zed2i_camera_center', 'zed2i_left_camera_frame', 0.0, -0.06, 0.0)
        tf_transforms.append(zed2i_center_to_zed2i_left)
        
        # Connect standard ZED frames
        odom_to_zed_center = self.create_transform('odom', 'zed_camera_center', 0.0, 0.0, 0.2)
        tf_transforms.append(odom_to_zed_center)
        
        zed_center_to_zed_left = self.create_transform('zed_camera_center', 'zed_left_camera_frame', 0.0, -0.06, 0.0)
        tf_transforms.append(zed_center_to_zed_left)
        
        # Add remaining bridge transforms between zed2i and zed frames
        bridge_tf_left = self.create_transform('zed2i_left_camera_frame', 'zed_left_camera_frame', 0.0, 0.0, 0.0)
        tf_transforms.append(bridge_tf_left)
        
        bridge_tf_center = self.create_transform('zed2i_camera_center', 'zed_camera_center', 0.0, 0.0, 0.0)
        tf_transforms.append(bridge_tf_center)
        
        # Add right camera frames
        zed_center_to_zed_right = self.create_transform('zed_camera_center', 'zed_right_camera_frame', 0.0, 0.06, 0.0)
        tf_transforms.append(zed_center_to_zed_right)
        
        zed2i_center_to_zed2i_right = self.create_transform('zed2i_camera_center', 'zed2i_right_camera_frame', 0.0, 0.06, 0.0)
        tf_transforms.append(zed2i_center_to_zed2i_right)
        
        bridge_tf_right = self.create_transform('zed2i_right_camera_frame', 'zed_right_camera_frame', 0.0, 0.0, 0.0)
        tf_transforms.append(bridge_tf_right)
        
        # Add optical frames
        self.add_optical_frames(tf_transforms)
        
        # Publish all transforms
        self.static_broadcaster.sendTransform(tf_transforms)
        self.get_logger().info('Published static transforms for occupancy grid')
    
    def add_optical_frames(self, transforms_list):
        """Add optical frame transforms (rotated 90 degrees around X)"""
        # Left optical frames
        left_optical = self.create_transform(
            'zed_left_camera_frame', 
            'zed_left_camera_optical_frame', 
            0.0, 0.0, 0.0,
            # Rotate 90 degrees around X axis for optical frame
            -0.5, 0.5, -0.5, 0.5
        )
        transforms_list.append(left_optical)
        
        left_optical_2i = self.create_transform(
            'zed2i_left_camera_frame', 
            'zed2i_left_camera_optical_frame', 
            0.0, 0.0, 0.0,
            -0.5, 0.5, -0.5, 0.5
        )
        transforms_list.append(left_optical_2i)
        
        # Right optical frames
        right_optical = self.create_transform(
            'zed_right_camera_frame', 
            'zed_right_camera_optical_frame', 
            0.0, 0.0, 0.0,
            -0.5, 0.5, -0.5, 0.5
        )
        transforms_list.append(right_optical)
        
        right_optical_2i = self.create_transform(
            'zed2i_right_camera_frame', 
            'zed2i_right_camera_optical_frame', 
            0.0, 0.0, 0.0,
            -0.5, 0.5, -0.5, 0.5
        )
        transforms_list.append(right_optical_2i)
        
        # Bridge between optical frames
        bridge_left_optical = self.create_transform(
            'zed2i_left_camera_optical_frame', 
            'zed_left_camera_optical_frame',
            0.0, 0.0, 0.0
        )
        transforms_list.append(bridge_left_optical)
        
        bridge_right_optical = self.create_transform(
            'zed2i_right_camera_optical_frame', 
            'zed_right_camera_optical_frame',
            0.0, 0.0, 0.0
        )
        transforms_list.append(bridge_right_optical)

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
