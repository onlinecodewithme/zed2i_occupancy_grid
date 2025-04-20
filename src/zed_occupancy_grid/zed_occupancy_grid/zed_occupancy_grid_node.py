#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros import TransformBroadcaster
import tf2_ros
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ZedOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('zed_occupancy_grid_node')

        # Declare parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'zed_left_camera_frame')
        self.declare_parameter('resolution', 0.05)  # meters per cell
        self.declare_parameter('grid_width', 10.0)  # meters
        self.declare_parameter('grid_height', 10.0)  # meters
        self.declare_parameter('min_depth', 0.5)    # min depth in meters
        self.declare_parameter('max_depth', 20.0)   # max depth in meters
        self.declare_parameter('depth_topic', '/zed2i/zed_node/depth/depth_registered')

        # Get parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.resolution = self.get_parameter('resolution').value
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.depth_topic = self.get_parameter('depth_topic').value

        # Initialize grid properties
        self.grid_cols = int(self.grid_width / self.resolution)
        self.grid_rows = int(self.grid_height / self.resolution)
        self.grid_origin_x = -self.grid_width / 2.0
        self.grid_origin_y = -self.grid_height / 2.0

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Set up QoS profile for reliable depth image subscription
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Publishers and subscribers
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', 10)
        
        self.depth_subscriber = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile)

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('ZED Occupancy Grid Node initialized')
        self.get_logger().info(f'Listening to depth topic: {self.depth_topic}')
        self.get_logger().info(f'Grid size: {self.grid_width}x{self.grid_height} meters, resolution: {self.resolution} m/cell')

    def depth_callback(self, depth_msg):
        try:
            # Log that we received a depth image
            self.get_logger().info(f'Received depth image with timestamp: {depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec}')
            
            # Convert depth image to OpenCV format (float32 in meters)
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            # Get depth image stats for debugging
            valid_depths = depth_image[np.isfinite(depth_image)]
            if valid_depths.size > 0:
                min_val = np.min(valid_depths)
                max_val = np.max(valid_depths)
                avg_val = np.mean(valid_depths)
                self.get_logger().info(f'Depth image stats - min: {min_val:.2f}m, max: {max_val:.2f}m, avg: {avg_val:.2f}m')
            else:
                self.get_logger().warning('No valid depth values in image')
                return
            
            # Create occupancy grid message
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = self.get_clock().now().to_msg()
            grid_msg.header.frame_id = self.map_frame
            
            # Set grid metadata
            grid_msg.info.resolution = self.resolution
            grid_msg.info.width = self.grid_cols
            grid_msg.info.height = self.grid_rows
            
            # Set grid origin
            grid_msg.info.origin.position.x = self.grid_origin_x
            grid_msg.info.origin.position.y = self.grid_origin_y
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0  # No rotation

                # Initialize grid data with all FREE cells (0) instead of unknown for better visibility
            grid_msg.data = [0] * (self.grid_cols * self.grid_rows)
            
            # Force some test cells to be occupied to verify grid is working
            # Create a large visible pattern - a cross through the middle
            center_x = self.grid_cols // 2
            center_y = self.grid_rows // 2
            pattern_size = min(self.grid_cols, self.grid_rows) // 3
            
            # Draw horizontal line
            for dx in range(-pattern_size, pattern_size + 1):
                x = center_x + dx
                y = center_y
                if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                    index = y * self.grid_cols + x
                    grid_msg.data[index] = 100  # Mark as occupied
            
            # Draw vertical line
            for dy in range(-pattern_size, pattern_size + 1):
                x = center_x
                y = center_y + dy
                if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                    index = y * self.grid_cols + x
                    grid_msg.data[index] = 100  # Mark as occupied
            
            # Draw a box around the perimeter as a test pattern
            border = 10  # cells from edge
            for x in range(border, self.grid_cols - border):
                for y in [border, self.grid_rows - border - 1]:
                    if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                        index = y * self.grid_cols + x
                        grid_msg.data[index] = 100
                        
            for y in range(border, self.grid_rows - border):
                for x in [border, self.grid_cols - border - 1]:
                    if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                        index = y * self.grid_cols + x
                        grid_msg.data[index] = 100
                        
            # Publish the grid with the test pattern
            self.get_logger().info("Publishing test occupancy grid pattern with frame_id: " + grid_msg.header.frame_id)
            self.occupancy_grid_pub.publish(grid_msg)
            
            # Try to get transform from camera to map
            # First print available frames for debugging
            try:
                # Get all frames for debugging
                frames = self.tf_buffer.all_frames_as_string()
                self.get_logger().debug(f'Available TF frames:\n{frames}')
                
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1.0)
                )
                
                self.get_logger().debug(f'Got transform from {self.camera_frame} to {self.map_frame}')
                self.get_logger().debug(f'Camera position: x={transform.transform.translation.x:.2f}, '
                                       f'y={transform.transform.translation.y:.2f}, '
                                       f'z={transform.transform.translation.z:.2f}')
                
                # Process depth image to update occupancy grid
                cells_updated = self.update_grid_from_depth(depth_image, grid_msg, transform)
                
                if cells_updated > 0:
                    self.get_logger().info(f'Updated {cells_updated} cells in occupancy grid')
                    # Publish the grid
                    self.occupancy_grid_pub.publish(grid_msg)
                else:
                    self.get_logger().warning('No cells updated in occupancy grid')
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f'TF error: {str(e)}')
                # Try to get the transform directly from odom to camera as fallback
                try:
                    self.get_logger().info(f'Trying fallback: lookup transform from odom to {self.camera_frame}')
                    transform = self.tf_buffer.lookup_transform(
                        'odom',
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=1.0)
                    )
                    # We got odom to camera, use it with map_frame as fallback
                    self.get_logger().info(f'Got transform from odom to {self.camera_frame}, using as fallback')
                    # Publish the grid with odom frame instead
                    grid_msg.header.frame_id = 'odom'
                    cells_updated = self.update_grid_from_depth(depth_image, grid_msg, transform)
                    if cells_updated > 0:
                        self.occupancy_grid_pub.publish(grid_msg)
                except Exception as e2:
                    self.get_logger().error(f'Fallback transform failed: {str(e2)}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def update_grid_from_depth(self, depth_image, grid_msg, transform):
        """
        Update the occupancy grid using the depth image and camera transform.
        Returns the number of cells updated.
        """
        height, width = depth_image.shape
        cells_updated = 0
        
        # Camera parameters (example values, should be set based on ZED camera)
        # ZED2i has roughly 110° horizontal FOV and 70° vertical FOV
        fov_horizontal = 110.0 * np.pi / 180.0  # radians
        fov_vertical = 70.0 * np.pi / 180.0    # radians
        
        # Camera position and orientation from transform
        camera_x = transform.transform.translation.x
        camera_y = transform.transform.translation.y
        camera_z = transform.transform.translation.z
        
        # Log camera position
        self.get_logger().debug(f'Processing depth image with camera at ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})')
        
        # For simplicity, assuming camera is looking forward along X
        # In a complete implementation, use quaternion to get the full orientation
        
        # Process each pixel in the depth image (could be optimized)
        # Use a smaller stride for better coverage (5 instead of 10)
        for y in range(0, height, 5):  # Stride for efficiency
            for x in range(0, width, 5):  # Stride for efficiency
                depth = depth_image[y, x]
                
                # Skip invalid or out-of-range depths
                if not np.isfinite(depth) or depth < self.min_depth or depth > self.max_depth:
                    continue
                
                # Calculate 3D position in camera frame
                angle_h = (x / width - 0.5) * fov_horizontal
                angle_v = (y / height - 0.5) * fov_vertical
                
                # Convert to 3D point in camera frame
                point_x = depth * np.cos(angle_v) * np.cos(angle_h)
                point_y = depth * np.cos(angle_v) * np.sin(angle_h)
                point_z = depth * np.sin(angle_v)
                
                # Use proper quaternion transformation
                # Extract quaternion components from the transform
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                
                # Apply quaternion rotation to the point
                # Formula: p' = q * p * q^-1 (quaternion rotation)
                # For simplicity, we're using the quaternion rotation matrix form
                
                # First, compute the rotation matrix from quaternion
                # This is a simplified form of the quaternion-to-rotation-matrix conversion
                xx = qx * qx
                xy = qx * qy
                xz = qx * qz
                xw = qx * qw
                yy = qy * qy
                yz = qy * qz
                yw = qy * qw
                zz = qz * qz
                zw = qz * qw
                
                r00 = 1 - 2 * (yy + zz)
                r01 = 2 * (xy - zw)
                r02 = 2 * (xz + yw)
                r10 = 2 * (xy + zw)
                r11 = 1 - 2 * (xx + zz)
                r12 = 2 * (yz - xw)
                r20 = 2 * (xz - yw)
                r21 = 2 * (yz + xw)
                r22 = 1 - 2 * (xx + yy)
                
                # Apply rotation matrix to point
                rotated_x = r00 * point_x + r01 * point_y + r02 * point_z
                rotated_y = r10 * point_x + r11 * point_y + r12 * point_z
                rotated_z = r20 * point_x + r21 * point_y + r22 * point_z
                
                # Apply translation
                map_x = rotated_x + camera_x
                map_y = rotated_y + camera_y
                map_z = rotated_z + camera_z  # We'll need this for 3D mapping if needed
                
                # Convert to grid cell
                grid_x = int((map_x - self.grid_origin_x) / self.resolution)
                grid_y = int((map_y - self.grid_origin_y) / self.resolution)
                
                # Check if inside grid bounds
                if 0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows:
                    # Mark cell as occupied
                    grid_index = grid_y * self.grid_cols + grid_x
                    grid_msg.data[grid_index] = 100  # 100 means occupied
                    cells_updated += 1
                    
                    # Simple ray casting - mark cells between camera and point as free
                    free_cells = self.mark_ray_as_free(camera_x, camera_y, map_x, map_y, grid_msg)
                    cells_updated += free_cells
        
        return cells_updated

    def mark_ray_as_free(self, start_x, start_y, end_x, end_y, grid_msg):
        """
        Marks cells along a ray from start to end as free (Bresenham's line algorithm).
        Returns the number of cells marked as free.
        """
        # Convert to grid coordinates
        start_grid_x = int((start_x - self.grid_origin_x) / self.resolution)
        start_grid_y = int((start_y - self.grid_origin_y) / self.resolution)
        end_grid_x = int((end_x - self.grid_origin_x) / self.resolution)
        end_grid_y = int((end_y - self.grid_origin_y) / self.resolution)
        
        # Bresenham's line algorithm
        dx = abs(end_grid_x - start_grid_x)
        dy = abs(end_grid_y - start_grid_y)
        sx = 1 if start_grid_x < end_grid_x else -1
        sy = 1 if start_grid_y < end_grid_y else -1
        err = dx - dy
        
        x, y = start_grid_x, start_grid_y
        cells_marked = 0
        
        while x != end_grid_x or y != end_grid_y:
            if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                grid_index = y * self.grid_cols + x
                # Only mark as free if not already marked as occupied
                if grid_msg.data[grid_index] != 100:
                    if grid_msg.data[grid_index] != 0:  # Don't count cells already marked as free
                        cells_marked += 1
                    grid_msg.data[grid_index] = 0  # 0 means free
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells_marked


def main(args=None):
    rclpy.init(args=args)
    node = ZedOccupancyGridNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
