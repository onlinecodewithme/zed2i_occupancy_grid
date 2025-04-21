#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import StaticTransformBroadcaster


import tf2_ros
from rclpy.duration import Duration
from tf2_ros import TransformBroadcaster

class StaticTFPublisher(Node):
    """
    Node to publish transforms needed for the occupancy grid to work.
    This is needed when no SLAM or localization is running to provide the transforms.
    """

    def __init__(self):
        super().__init__('tf_publisher')
        
        # Static broadcaster for transforms that don't change
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # Dynamic broadcaster for transforms that need to be updated
        self.dynamic_broadcaster = TransformBroadcaster(self)
        
        # Create TF buffer for lookup operations
        self.tf_buffer = tf2_ros.Buffer(Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publish necessary static transforms
        tf_transforms = []
        
        # IMPORTANT: We need to ensure a proper TF tree for occupancy grid mapping
        self.get_logger().info('Setting up TF transforms for occupancy grid')
        
        # Add a ground frame for visualization
        map_to_ground = self.create_transform('map', 'ground', 0.0, 0.0, -0.3)
        tf_transforms.append(map_to_ground)
        
        # Add fallback transforms that might be missing
        # Only used if ZED node doesn't publish these
        base_to_left = self.create_transform('zed_camera_link', 'zed_left_camera_frame', 0.01, -0.06, -0.015)
        tf_transforms.append(base_to_left)
        
        # Publish all static transforms
        self.static_broadcaster.sendTransform(tf_transforms)
        self.get_logger().info('Published static transforms for occupancy grid')
        
        # Initialize last known position from camera to map
        self.last_odom_pose = None
        self.camera_moved = False
        
        # Subscribe to ZED camera position updates for immediate transform updates
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/zed/zed_node/pose', self.camera_pose_callback, 1)
        self.get_logger().info('Subscribed to ZED camera pose topic for real-time updates')
        
        # The map-to-odom transform will be published both on camera movement and as a backup timer
        # This timer is a fallback to ensure we always have a transform even if camera pose isn't received
        self.map_odom_timer = self.create_timer(0.1, self.publish_map_to_odom)
        
    def camera_pose_callback(self, pose_msg):
        """Process camera pose updates and immediately publish transforms"""
        try:
            # Extract position from the camera pose
            pos = pose_msg.pose.position
            
            # Log position for debugging
            self.get_logger().debug(f"Camera position update: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
            
            # Immediately publish the transform on camera movement
            self.publish_map_to_odom()
            
            # Mark that camera has moved
            self.camera_moved = True
            
        except Exception as e:
            self.get_logger().error(f"Error in camera pose callback: {str(e)}")

    def publish_map_to_odom(self):
        """Publish the transform from map to odom frame"""
        try:
            # First try to get the transform from the ZED's SLAM system
            try:
                # Check if a transform from map to odom already exists from ZED SLAM
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    'odom',
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                
                # If we get here, the transform exists - use it
                self.get_logger().debug("Using existing map->odom transform from ZED SLAM")
                
                # Republish the transform to ensure it's available
                transform.header.stamp = self.get_clock().now().to_msg()
                self.dynamic_broadcaster.sendTransform(transform)
                return
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                # No existing transform, we'll create one
                pass
            
            # This is the critical transform that was previously commented out
            # We're now publishing it dynamically to ensure the TF tree is connected
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'map'
            transform.child_frame_id = 'odom'
            
            # By default, we set an identity transform (no offset)
            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 0.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
            
            # Publish the transform
            self.dynamic_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().warning(f"Error publishing map to odom transform: {str(e)}")
    
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
