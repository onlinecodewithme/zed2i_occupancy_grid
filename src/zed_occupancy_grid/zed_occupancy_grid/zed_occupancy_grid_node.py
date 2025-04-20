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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import time
import threading
import math

class ZedOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('zed_occupancy_grid_node')
        
        # Set ROS logging to be less verbose
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Occupancy probabilities - using log-odds for numerical stability
        # P(occupied) = 1 - 1/(1 + exp(l))
        self.FREE_THRESHOLD = -2.0  # log-odds threshold for considering a cell free
        self.OCCUPIED_THRESHOLD = 2.0  # log-odds threshold for considering a cell occupied
        self.LOG_ODDS_PRIOR = 0.0  # log-odds of prior probability (0.5)
        self.LOG_ODDS_FREE = -0.4  # log-odds update for free cells
        self.LOG_ODDS_OCCUPIED = 0.8  # log-odds update for occupied cells
        self.LOG_ODDS_MIN = -5.0  # minimum log-odds value
        self.LOG_ODDS_MAX = 5.0  # maximum log-odds value

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
        
        # Log-odds grid for proper probabilistic updates
        self.log_odds_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Grid data is published as int8, but we keep a float version for probabilistic updates
        self.cell_height_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Counter to track number of observations for each cell
        self.observation_count_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        
        # Filtering parameters
        self.temporal_filtering = True        # Enable temporal filtering
        self.spatial_filtering = True         # Enable spatial filtering
        self.min_observations = 2             # Reduced minimum observations for faster map updates
        self.max_ray_length = 5.0             # Maximum ray length for ray casting
        
        # Parameters for map updates with camera motion
        self.last_camera_position = None      # Track camera position to detect movement
        self.position_change_threshold = 0.05  # Min position change to trigger map update (meters)
        self.rotation_change_threshold = 0.05  # Min rotation change to trigger update (radians)
        self.reset_cells_on_movement = False   # Whether to reset cells that become out of view
        self.camera_motion_detected = False    # Flag to indicate if camera has moved
        self.last_camera_quaternion = None     # Track camera rotation
        
        # Camera movement adaptation settings
        self.static_alpha = 0.9               # Higher temporal filtering when static
        self.moving_alpha = 0.5               # Lower temporal filtering when moving
        self.current_alpha = self.static_alpha  # Current alpha value based on motion

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Set up QoS profile for reliable depth image subscription
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20  # Increased to handle more messages
        )
        
        # Set up QoS profile for reliable occupancy grid publication
        # Make it compatible with RViz's requirements
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Critical for RViz compatibility
            depth=20
        )
        
        # QoS profile for point cloud subscription - to match ZED node settings
        qos_profile_point_cloud = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Match ZED's publisher durability
            depth=10
        )

        # Publishers and subscribers
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', qos_profile_pub)
        
        # Make sure subscription QoS matches the publisher's QoS
        self.depth_subscriber = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sub)
            
        # Add rate limiting with separate update/publish cycles
        self.last_publish_time = 0.0
        self.publish_period = 0.2  # Faster publishing (5Hz) for better responsiveness
        self.last_update_time = 0.0
        self.update_period = 0.1  # Process depth at 10Hz for better map updates with motion
        
        # Add a mutex for thread safety
        self.grid_lock = threading.Lock()
        
        # Cache the last grid message for reuse
        self.last_grid_msg = None
        
        # Create a timer for regular map updates (even with no new data)
        self.map_timer = self.create_timer(0.5, self.publish_map_timer_callback)  # 2Hz timer

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('ZED Occupancy Grid Node initialized')
        self.get_logger().info(f'Listening to depth topic: {self.depth_topic}')
        self.get_logger().info(f'Grid size: {self.grid_width}x{self.grid_height} meters, resolution: {self.resolution} m/cell')
        self.get_logger().info(f'Motion-adaptive temporal filtering enabled')

    def depth_callback(self, depth_msg):
        try:
            # Rate limiting to prevent overwhelming RViz
            current_time = time.time()
            if current_time - self.last_update_time < self.update_period:
                return  # Skip this message to limit rate
            
            self.last_update_time = current_time
            
            # Convert depth image to OpenCV format (float32 in meters)
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            # Get depth image stats for debugging (reduce verbosity)
            valid_depths = depth_image[np.isfinite(depth_image)]
            if valid_depths.size > 0:
                min_val = np.min(valid_depths)
                max_val = np.max(valid_depths)
                avg_val = np.mean(valid_depths)
                self.get_logger().debug(f'Depth image stats - min: {min_val:.2f}m, max: {max_val:.2f}m, avg: {avg_val:.2f}m')
            else:
                self.get_logger().warning('No valid depth values in image')
                return
                
            # Check if any TF frames are missing and log info to help debugging
            try:
                frames = self.tf_buffer.all_frames_as_string()
                if "map" not in frames or "zed_left_camera_frame" not in frames:
                    missing = []
                    if "map" not in frames:
                        missing.append("map")
                    if "zed_left_camera_frame" not in frames:
                        missing.append("zed_left_camera_frame")
                    self.get_logger().warn(f"Missing TF frames: {', '.join(missing)}. Available frames: {frames}")
            except Exception as e:
                self.get_logger().warn(f"Error checking TF frames: {e}")
            
            # Try to get transform from camera to map
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1.0)
                )
                
                # Update the occupancy grid using depth image and transform
                self.update_grid_from_depth(depth_image, transform)
                
                # Publish at the specified rate
                if current_time - self.last_publish_time >= self.publish_period:
                    self.last_publish_time = current_time
                    try:
                        self.publish_occupancy_grid()
                    except Exception as e:
                        self.get_logger().error(f"Error publishing occupancy grid: {e}")
                
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
                    # We got odom to camera, use it with odom as frame
                    self.get_logger().info(f'Got transform from odom to {self.camera_frame}, using as fallback')
                    self.update_grid_from_depth(depth_image, transform)
                    
                    if current_time - self.last_publish_time >= self.publish_period:
                        self.last_publish_time = current_time
                        # Override map frame with odom for this publish
                        self.publish_occupancy_grid(override_frame='odom')
                except Exception as e2:
                    self.get_logger().error(f'Fallback transform failed: {str(e2)}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def publish_map_timer_callback(self):
        """Regular timer callback to publish the latest map, even if no new depth data"""
        with self.grid_lock:
            # Only publish if we have a map
            if np.any(self.observation_count_grid > 0):
                self.publish_occupancy_grid()
    
    def publish_occupancy_grid(self, override_frame=None):
        """Publish the occupancy grid with the latest data"""
        with self.grid_lock:
            # Create occupancy grid message
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = self.get_clock().now().to_msg()
            grid_msg.header.frame_id = override_frame if override_frame else self.map_frame
            
            # Set grid metadata
            grid_msg.info.resolution = self.resolution
            grid_msg.info.width = self.grid_cols
            grid_msg.info.height = self.grid_rows
            
            # Set grid origin
            grid_msg.info.origin.position.x = self.grid_origin_x
            grid_msg.info.origin.position.y = self.grid_origin_y
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0  # No rotation

            # Convert log-odds to occupancy probabilities [0,100]
            # Apply spatial filtering if enabled
            grid_msg.data = self.create_grid_msg_from_log_odds()
                            
            # Publish the grid
            self.occupancy_grid_pub.publish(grid_msg)
            
            # Cache for reuse
            self.last_grid_msg = grid_msg
            
            # Log the publish with current camera state
            state = "moving" if self.camera_motion_detected else "static"
            self.get_logger().debug(f'Published occupancy grid with frame_id: {grid_msg.header.frame_id} (camera {state})')

    def create_grid_msg_from_log_odds(self):
        """Convert log-odds grid to occupancy grid message data"""
        grid_data = []
        
        # First apply median filter if spatial filtering is enabled
        if self.spatial_filtering:
            # Apply 3x3 median filter to reduce noise
            from scipy.ndimage import median_filter
            filtered_log_odds = median_filter(self.log_odds_grid, size=3)
        else:
            filtered_log_odds = self.log_odds_grid
            
        # Convert log-odds to probabilities
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                log_odds = filtered_log_odds[y, x]
                obs_count = self.observation_count_grid[y, x]
                
                if obs_count < self.min_observations:
                    # If not enough observations, mark as unknown (-1)
                    grid_data.append(-1)
                else:
                    if log_odds > self.OCCUPIED_THRESHOLD:
                        # Occupied cell
                        grid_data.append(100)
                    elif log_odds < self.FREE_THRESHOLD:
                        # Free cell
                        grid_data.append(0)
                    else:
                        # Convert log-odds to probability [0-100]
                        prob = 1.0 - (1.0 / (1.0 + math.exp(log_odds)))
                        grid_data.append(int(prob * 100))
        
        return grid_data
    
    def detect_camera_motion(self, transform):
        """
        Detect if the camera has moved significantly since the last update
        Returns: True if camera moved, False if static
        """
        camera_x = transform.transform.translation.x
        camera_y = transform.transform.translation.y
        camera_z = transform.transform.translation.z
        
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Calculate position change
        position_changed = False
        rotation_changed = False
        
        if self.last_camera_position is not None:
            dx = camera_x - self.last_camera_position[0]
            dy = camera_y - self.last_camera_position[1]
            dz = camera_z - self.last_camera_position[2]
            position_change = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if position_change > self.position_change_threshold:
                position_changed = True
                
        # Calculate rotation change using quaternion dot product
        if self.last_camera_quaternion is not None:
            # Quaternion dot product gives cos(theta/2) where theta is the angle between orientations
            dot_product = (qw * self.last_camera_quaternion[0] + 
                          qx * self.last_camera_quaternion[1] + 
                          qy * self.last_camera_quaternion[2] + 
                          qz * self.last_camera_quaternion[3])
            
            # Ensure dot product is in valid range [-1, 1]
            dot_product = max(min(dot_product, 1.0), -1.0)
            
            # Convert to angle
            angle_diff = 2.0 * math.acos(abs(dot_product))
            
            if angle_diff > self.rotation_change_threshold:
                rotation_changed = True
        
        # Update last known position and rotation
        self.last_camera_position = (camera_x, camera_y, camera_z)
        self.last_camera_quaternion = (qw, qx, qy, qz)
        
        # Camera has moved if either position or rotation changed significantly
        camera_moved = position_changed or rotation_changed
        
        # Log camera motion if state changed
        if camera_moved != self.camera_motion_detected:
            if camera_moved:
                self.get_logger().info("Camera movement detected - adapting filter settings")
                self.current_alpha = self.moving_alpha
            else:
                self.get_logger().info("Camera is static - reverting to stable filter settings")
                self.current_alpha = self.static_alpha
        
        self.camera_motion_detected = camera_moved
        return camera_moved
        
    def update_grid_from_depth(self, depth_image, transform):
        """
        Update the log-odds grid using the depth image and camera transform.
        Uses probabilistic updates for better map quality.
        """
        height, width = depth_image.shape
        
        # Extract camera position
        camera_x = transform.transform.translation.x
        camera_y = transform.transform.translation.y
        camera_z = transform.transform.translation.z
        
        # Camera parameters - ZED2i camera
        fov_horizontal = 110.0 * np.pi / 180.0  # radians
        fov_vertical = 70.0 * np.pi / 180.0     # radians
        
        # Extract quaternion from transform for rotation
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Compute the rotation matrix once
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
        
        # Detect if camera has moved
        is_moving = self.detect_camera_motion(transform)
        
        # Choose the right sampling grid size based on camera motion
        # When moving, we use a coarser grid to process faster
        stride_y = 10 if not is_moving else 20
        stride_x = 10 if not is_moving else 20
        
        # Lock for thread safety
        with self.grid_lock:
            # Create a temporary grid for this update
            current_update = np.full((self.grid_rows, self.grid_cols), self.LOG_ODDS_PRIOR, dtype=np.float32)
            
            # Process each pixel in the depth image with adaptive stride
            for y in range(0, height, stride_y):
                for x in range(0, width, stride_x):
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
                    
                    # Apply rotation matrix to point
                    rotated_x = r00 * point_x + r01 * point_y + r02 * point_z
                    rotated_y = r10 * point_x + r11 * point_y + r12 * point_z
                    rotated_z = r20 * point_x + r21 * point_y + r22 * point_z
                    
                    # Apply translation
                    map_x = rotated_x + camera_x
                    map_y = rotated_y + camera_y
                    map_z = rotated_z + camera_z  # We'll need this for 3D mapping if needed
                    
                    # Convert to grid cell coordinates
                    grid_x = int((map_x - self.grid_origin_x) / self.resolution)
                    grid_y = int((map_y - self.grid_origin_y) / self.resolution)
                    
                    # Check if inside grid bounds
                    if 0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows:
                        # Increase observation count for the cell
                        self.observation_count_grid[grid_y, grid_x] += 1
                        
                        # Update the occupied cell with positive log-odds
                        current_update[grid_y, grid_x] = self.LOG_ODDS_OCCUPIED
                        
                        # Also store the height for potential 3D mapping
                        if self.cell_height_grid[grid_y, grid_x] == 0:
                            self.cell_height_grid[grid_y, grid_x] = map_z
                        else:
                            # Exponential weighted average for height
                            alpha = 0.8
                            self.cell_height_grid[grid_y, grid_x] = alpha * self.cell_height_grid[grid_y, grid_x] + (1-alpha) * map_z
                    
                        # Mark cells along the ray as free using Bresenham's line algorithm
                        self.mark_ray_as_free_3d(camera_x, camera_y, map_x, map_y, current_update)
            
            # Integrate the current update with the main grid
            # Use temporal filtering to reduce noise if enabled
            if self.temporal_filtering:
                # Apply weighted average based on camera motion
                # Moving camera = less temporal filtering (lower alpha) for faster updates
                # Static camera = more temporal filtering (higher alpha) for stability
                mask = (current_update != self.LOG_ODDS_PRIOR)  # Only update cells we observed
                self.log_odds_grid[mask] = self.current_alpha * self.log_odds_grid[mask] + (1-self.current_alpha) * current_update[mask]
            else:
                # Faster but less stable updates - direct addition of log-odds
                self.log_odds_grid += current_update * (current_update != self.LOG_ODDS_PRIOR)
            
            # Clamp log-odds values to prevent numerical issues
            self.log_odds_grid = np.clip(self.log_odds_grid, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX)

    def mark_ray_as_free_3d(self, start_x, start_y, end_x, end_y, current_update):
        """
        Marks cells along a ray from start to end as free using 3D Bresenham's algorithm.
        Updates the current_update array with negative log-odds for free cells.
        """
        # Limit the ray length to avoid marking too much as free
        dx = end_x - start_x
        dy = end_y - start_y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > self.max_ray_length:
            # Scale the end point to max_ray_length
            factor = self.max_ray_length / dist
            end_x = start_x + dx * factor
            end_y = start_y + dy * factor
        
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
        
        # For each point along the line except the endpoint (which might be occupied)
        while (x != end_grid_x or y != end_grid_y):
            if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                # Mark as free using negative log-odds
                current_update[y, x] = self.LOG_ODDS_FREE
                # Also increment observation count
                self.observation_count_grid[y, x] += 1
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy


def main(args=None):
    rclpy.init(args=args)
    node = ZedOccupancyGridNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
