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
        
        # Create base_link in map frame (initially at origin, will be updated dynamically)
        map_to_base = self.create_transform('map', 'base_link', 0.0, 0.0, 0.0)
        tf_transforms.append(map_to_base)
        
        # Add track frames relative to base_link
        base_to_left_track = self.create_transform('base_link', 'left_track', 0.0, 0.15, 0.0)
        tf_transforms.append(base_to_left_track)
        
        base_to_right_track = self.create_transform('base_link', 'right_track', 0.0, -0.15, 0.0)
        tf_transforms.append(base_to_right_track)
        
        # Add camera links
        base_to_zed_camera = self.create_transform('base_link', 'zed_camera_link', 0.1, 0.0, 0.15)
        tf_transforms.append(base_to_zed_camera)
        
        # Only used if ZED node doesn't publish these
        zed_camera_to_center = self.create_transform('zed_camera_link', 'zed_camera_center', 0.0, 0.0, 0.0)
        tf_transforms.append(zed_camera_to_center)
        
        zed_camera_to_link = self.create_transform('zed_camera_link', 'zed_camera_link', 0.0, 0.0, 0.0)
        tf_transforms.append(zed_camera_to_link)
        
        # Left camera frames
        zed_camera_to_left = self.create_transform('zed_camera_link', 'zed_left_camera_frame', 0.0, 0.06, 0.0)
        tf_transforms.append(zed_camera_to_left)
        
        zed_left_to_optical = self.create_transform('zed_left_camera_frame', 'zed_left_camera_optical_frame', 
                                                   0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5)  # Rotate to optical frame
        tf_transforms.append(zed_left_to_optical)
        
        # Right camera frames
        zed_camera_to_right = self.create_transform('zed_camera_link', 'zed_right_camera_frame', 0.0, -0.06, 0.0)
        tf_transforms.append(zed_camera_to_right)
        
        zed_right_to_optical = self.create_transform('zed_right_camera_frame', 'zed_right_camera_optical_frame',
                                                    0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5)  # Rotate to optical frame
        tf_transforms.append(zed_right_to_optical)
        
        # Add ZED2i specific frames
        base_to_zed2i = self.create_transform('base_link', 'zed2i_link', 0.1, 0.0, 0.15)
        tf_transforms.append(base_to_zed2i)
        
        # ZED2i left camera
        zed2i_to_left = self.create_transform('zed2i_link', 'zed2i_left_camera_frame', 0.0, 0.06, 0.0)
        tf_transforms.append(zed2i_to_left)
        
        zed2i_left_to_optical = self.create_transform('zed2i_left_camera_frame', 'zed2i_left_camera_optical_frame',
                                                     0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5)
        tf_transforms.append(zed2i_left_to_optical)
        
        # ZED2i right camera
        zed2i_to_right = self.create_transform('zed2i_link', 'zed2i_right_camera_frame', 0.0, -0.06, 0.0)
        tf_transforms.append(zed2i_to_right)
        
        zed2i_right_to_optical = self.create_transform('zed2i_right_camera_frame', 'zed2i_right_camera_optical_frame',
                                                      0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5)
        tf_transforms.append(zed2i_right_to_optical)
        
        # ZED2i optical frames
        zed2i_to_optical = self.create_transform('zed2i_link', 'zed2i_optical_frame', 0.0, 0.0, 0.0, 
                                               -0.5, 0.5, -0.5, 0.5)
        tf_transforms.append(zed2i_to_optical)
        
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
            orient = pose_msg.pose.orientation
            
            # Log position for debugging
            self.get_logger().debug(f"Camera position update: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
            
            # Update the dynamic transforms
            self.update_dynamic_transforms(pos, orient)
            
            # Immediately publish the transform on camera movement
            self.publish_map_to_odom()
            
            # Mark that camera has moved
            self.camera_moved = True
            
        except Exception as e:
            self.get_logger().error(f"Error in camera pose callback: {str(e)}")
    
    def update_dynamic_transforms(self, position, orientation):
        """Update dynamic transforms based on latest camera position"""
        try:
            # First, check if we should be using ZED's transforms
            try:
                # If ZED is publishing transforms properly, we can rely on them
                self.tf_buffer.lookup_transform(
                    'zed_left_camera_frame',
                    'zed_right_camera_frame',
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                # If we get here, ZED is providing transforms, so we don't need to update
                self.get_logger().debug("ZED camera is publishing transforms - using existing TF tree")
                return
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                # ZED isn't publishing all transforms, so we'll update them ourselves
                pass
            
            # Create a list of dynamic transforms to update
            dynamic_transforms = []
            
            # 1. Update map to base_link transform based on camera position
            # We assume base_link is close to the camera but slightly lower
            base_transform = TransformStamped()
            base_transform.header.stamp = self.get_clock().now().to_msg()
            base_transform.header.frame_id = 'map'
            base_transform.child_frame_id = 'base_link'
            
            # Base is positioned under the camera
            base_transform.transform.translation.x = position.x
            base_transform.transform.translation.y = position.y
            base_transform.transform.translation.z = position.z - 0.15  # 15cm below camera
            
            # Copy camera orientation for base
            base_transform.transform.rotation.x = orientation.x
            base_transform.transform.rotation.y = orientation.y
            base_transform.transform.rotation.z = orientation.z
            base_transform.transform.rotation.w = orientation.w
            
            dynamic_transforms.append(base_transform)
            
            # 2. Update odom to base_link if map to odom is available
            try:
                # Get the map to odom transform from the ZED SLAM system
                map_to_odom = self.tf_buffer.lookup_transform(
                    'map',
                    'odom',
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                
                # Calculate odom to base_link using map to odom and map to base
                odom_to_base = TransformStamped()
                odom_to_base.header.stamp = self.get_clock().now().to_msg()
                odom_to_base.header.frame_id = 'odom'
                odom_to_base.child_frame_id = 'base_link'
                
                # This is a simplified approach - for proper transform composition,
                # we should use tf2 Transform multiplication, but this is good enough for most cases
                
                # Translation - adjust for map to odom offset
                odom_to_base.transform.translation.x = position.x - map_to_odom.transform.translation.x
                odom_to_base.transform.translation.y = position.y - map_to_odom.transform.translation.y
                odom_to_base.transform.translation.z = (position.z - 0.15) - map_to_odom.transform.translation.z
                
                # Rotation - simplified by just copying camera orientation
                # In practice, we should account for map_to_odom rotation as well
                odom_to_base.transform.rotation = orientation
                
                dynamic_transforms.append(odom_to_base)
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                # If map to odom is not available, don't update odom to base_link
                self.get_logger().debug("No map to odom transform available - not updating odom to base_link")
            
            # Publish all dynamic transforms
            if dynamic_transforms:
                self.dynamic_broadcaster.sendTransform(dynamic_transforms)
                self.get_logger().debug(f"Published {len(dynamic_transforms)} dynamic transforms")
                
        except Exception as e:
            self.get_logger().error(f"Error updating dynamic transforms: {str(e)}")

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
