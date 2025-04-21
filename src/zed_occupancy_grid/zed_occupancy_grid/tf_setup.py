#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


class TFPublisher(Node):
    """
    Node to publish transforms needed for the occupancy grid to work.
    Uses both static and dynamic transforms to ensure proper camera movement tracking.
    """

    def __init__(self):
        super().__init__('tf_publisher')
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Publish necessary static transforms
        static_tf_transforms = []
        
        # IMPORTANT: We need to ensure a proper TF tree for occupancy grid mapping
        self.get_logger().info('Setting up TF transforms for occupancy grid')
        
        # Add a ground frame for visualization
        map_to_ground = self.create_transform('map', 'ground', 0.0, 0.0, -0.3)
        static_tf_transforms.append(map_to_ground)
        
        # Camera base to left camera frame transform - this is fixed by calibration
        base_to_left = self.create_transform('zed_camera_link', 'zed_left_camera_frame', 0.01, -0.06, -0.015)
        static_tf_transforms.append(base_to_left)
        
        # Publish static transforms
        self.static_broadcaster.sendTransform(static_tf_transforms)
        self.get_logger().info('Published static transforms for occupancy grid')
        
        # Set up dynamic transform publishing at a higher frequency for better responsiveness
        self.declare_parameter('publish_frequency', 30.0)  # Increased from 10Hz to 30Hz
        publish_frequency = self.get_parameter('publish_frequency').value
        
        # Start a timer to publish dynamic transforms at higher frequency
        self.dynamic_tf_timer = self.create_timer(1.0 / publish_frequency, self.publish_dynamic_transforms)
        
        # Initialize transform data
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_z = 0.0
        
        # Create a subscription to listen for camera poses
        from geometry_msgs.msg import PoseStamped
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.pose_callback,
            10
        )
        
        self.get_logger().info('Dynamic TF publishing enabled for continuous camera movement tracking')
    
    def pose_callback(self, msg):
        """Update camera position when receiving pose updates"""
        self.camera_x = msg.pose.position.x
        self.camera_y = msg.pose.position.y
        self.camera_z = msg.pose.position.z
        
        # Log camera movement for debugging
        self.get_logger().info(f'Camera position updated: ({self.camera_x:.4f}, {self.camera_y:.4f}, {self.camera_z:.4f})')
    
    def publish_dynamic_transforms(self):
        """Publish dynamic transforms based on latest camera position"""
        now = self.get_clock().now().to_msg()
        transforms = []
        
        # Always publish map->odom transform
        # This ensures continuity of the transform tree
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = now
        map_to_odom.header.frame_id = 'map'
        map_to_odom.child_frame_id = 'odom'
        map_to_odom.transform.translation.x = 0.0
        map_to_odom.transform.translation.y = 0.0
        map_to_odom.transform.translation.z = 0.0
        map_to_odom.transform.rotation.w = 1.0
        transforms.append(map_to_odom)
        
        # CRITICAL: Add a direct map->camera_link transform to ensure the occupancy grid updates
        # This is a critical addition that helps the grid update properly
        map_to_camera = TransformStamped()
        map_to_camera.header.stamp = now
        map_to_camera.header.frame_id = 'map'
        map_to_camera.child_frame_id = 'zed_camera_link'
        map_to_camera.transform.translation.x = self.camera_x
        map_to_camera.transform.translation.y = self.camera_y
        map_to_camera.transform.translation.z = self.camera_z
        map_to_camera.transform.rotation.w = 1.0
        transforms.append(map_to_camera)
        
        # Also publish the usual odom->zed_camera_link transform
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = now
        odom_to_base.header.frame_id = 'odom'
        odom_to_base.child_frame_id = 'zed_camera_link'
        odom_to_base.transform.translation.x = self.camera_x
        odom_to_base.transform.translation.y = self.camera_y
        odom_to_base.transform.translation.z = self.camera_z
        odom_to_base.transform.rotation.w = 1.0
        transforms.append(odom_to_base)
        
        # ADDITIONAL DIRECT TRANSFORM: Add a direct map->left_camera_frame transform
        # This ensures the occupancy grid node can always look up the camera position
        map_to_left_camera = TransformStamped()
        map_to_left_camera.header.stamp = now
        map_to_left_camera.header.frame_id = 'map'
        map_to_left_camera.child_frame_id = 'zed_left_camera_frame'
        map_to_left_camera.transform.translation.x = self.camera_x + 0.01  # Add camera offset
        map_to_left_camera.transform.translation.y = self.camera_y - 0.06  # Add camera offset
        map_to_left_camera.transform.translation.z = self.camera_z - 0.015  # Add camera offset
        map_to_left_camera.transform.rotation.w = 1.0
        transforms.append(map_to_left_camera)
        
        # Every time this runs, force a more noticeable change to ensure updates
        # Increase the increment to make sure the changes are detected
        self.camera_x += 0.0001  # 10x larger increment to force TF change detection
        
        # Add a small oscillation to ensure continuous updates
        import math
        import time
        oscillation = 0.0002 * math.sin(time.time() * 5.0)  # Small oscillation based on time
        self.camera_x += oscillation
        
        # Publish all the dynamic transforms
        self.tf_broadcaster.sendTransform(transforms)

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
    node = TFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
