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
        
        # Set ROS logging to INFO to reduce overhead from excessive logging
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Occupancy probabilities - using log-odds for numerical stability
        # P(occupied) = 1 - 1/(1 + exp(l))
        self.FREE_THRESHOLD = -2.0  # log-odds threshold for considering a cell free
        self.OCCUPIED_THRESHOLD = 2.0  # log-odds threshold for considering a cell occupied
        self.LOG_ODDS_PRIOR = 0.0  # log-odds of prior probability (0.5)
        self.LOG_ODDS_FREE = -1.5  # log-odds update for free cells (MORE AGGRESSIVE)
        self.LOG_ODDS_OCCUPIED = 3.0  # log-odds update for occupied cells (MORE AGGRESSIVE)
        self.LOG_ODDS_MIN = -5.0  # minimum log-odds value
        self.LOG_ODDS_MAX = 5.0  # maximum log-odds value
        
        # Map persistence settings
        self.map_persistence_enabled = True
        self.map_file_directory = '/tmp/zed_occupancy_map/'
        self.map_file_base = 'occupancy_map'
        self.auto_save_period = 60.0  # Save map every 60 seconds
        self.last_save_time = time.time()
        
        # Create the directory if it doesn't exist
        import os
        if not os.path.exists(self.map_file_directory):
            os.makedirs(self.map_file_directory)
            self.get_logger().info(f"Created map directory: {self.map_file_directory}")
            
        # Add auto-save map timer
        self.map_save_timer = self.create_timer(self.auto_save_period, self.save_map_timer_callback)
        
        # Add a mutex for thread safety
        self.grid_lock = threading.Lock()
        
        # Load map from file if available
        self.load_map()

        # Declare parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'zed_left_camera_frame')
        self.declare_parameter('resolution', 0.05)  # meters per cell
        self.declare_parameter('grid_width', 10.0)  # meters
        self.declare_parameter('grid_height', 10.0)  # meters
        self.declare_parameter('min_depth', 0.5)    # min depth in meters
        self.declare_parameter('max_depth', 20.0)   # max depth in meters
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')

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
        
        # Filtering parameters - MORE AGGRESSIVE FOR MOVEMENT
        self.temporal_filtering = True       # Enable temporal filtering
        self.spatial_filtering = True        # Enable spatial filtering
        self.min_observations = 1            # Minimum observations before marking as occupied - reduced to 1
        self.max_ray_length = 5.0            # Maximum ray length for ray casting
        
        # Parameters for map updates with camera motion - EXTREME SENSITIVITY FOR DYNAMIC UPDATES
        self.last_camera_position = None     # Track camera position to detect movement
        self.declare_parameter('position_change_threshold', 0.0001)  # Ultra sensitive - detect almost any change 
        self.position_change_threshold = self.get_parameter('position_change_threshold').value  # Get from param
        self.declare_parameter('rotation_change_threshold', 0.0001)  # Ultra sensitive
        self.rotation_change_threshold = self.get_parameter('rotation_change_threshold').value  # Get from param
        self.reset_cells_on_movement = True   # Enable resetting cells that become out of view
        self.camera_motion_detected = False   # Flag to indicate if camera has moved
        self.last_camera_quaternion = None    # Track camera rotation
        self.get_logger().info("!!! EXTREMELY SENSITIVE MOTION DETECTION ENABLED !!!")
        
        # Camera movement adaptation settings - MORE RESPONSIVE
        self.static_alpha = 0.7              # Less temporal filtering even when static (0.7 instead of 0.9)
        self.moving_alpha = 0.3              # Much less filtering when moving (0.3 instead of 0.5)
        self.current_alpha = self.moving_alpha # Start with moving settings for faster updates

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

        # Publishers and subscribers
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', qos_profile_pub)
        
        # Make sure subscription QoS matches the publisher's QoS
        # ADDITIONAL DEBUG TOPICS - Subscribe to pose directly to monitor camera movement
        self.depth_subscriber = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sub)
            
        # Add MULTIPLE special subscribers to force-track camera position
        from geometry_msgs.msg import PoseStamped, TransformStamped
        
        # Subscribe to multiple pose topics for redundancy
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/zed/zed_node/pose', self.pose_callback, 1)  # Higher priority (QoS=1)
            
        # Track TF directly for additional movement detection
        self.tf_subscriber = self.create_timer(0.1, self.check_tf_callback)  # 10Hz TF checks
        
        # Add high-frequency forced updates
        self.force_update_timer = self.create_timer(1.0, self.force_update_callback)  # Force update every second
            
        # Create a publisher for debug info
        from std_msgs.msg import String
        self.debug_pub = self.create_publisher(String, '/zed_grid_debug', 10)
            
        # Add rate limiting with separate update/publish cycles - FASTER UPDATES
        self.last_publish_time = 0.0
        self.publish_period = 0.1            # 10Hz publishing for better responsiveness
        self.last_update_time = 0.0
        self.update_period = 0.05            # 20Hz processing for immediate updates
        
        # Add a mutex for thread safety
        self.grid_lock = threading.Lock()
        
        # Cache the last grid message for reuse
        self.last_grid_msg = None
        
        # Create a timer for regular map updates (even with no new data)
        self.map_timer = self.create_timer(0.2, self.publish_map_timer_callback)  # 5Hz timer
        
        # Create a special timer that always logs camera position regardless of movement
        self.camera_monitor_timer = self.create_timer(0.5, self.camera_monitor_callback)  # 2Hz timer

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=10.0))  # Larger buffer
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('ZED Occupancy Grid Node initialized with MOVEMENT-OPTIMIZED settings')
        self.get_logger().info(f'Listening to depth topic: {self.depth_topic}')
        self.get_logger().info(f'Grid size: {self.grid_width}x{self.grid_height} meters, resolution: {self.resolution} m/cell')
        self.get_logger().info(f'ENHANCED motion-adaptive temporal filtering enabled')

    def depth_callback(self, depth_msg):
        try:
            # ALWAYS PROCESS ALL DEPTH FRAMES - force processing
            current_time = time.time()
            
            # Force update every time
            self.last_update_time = current_time
            self.get_logger().warn("DEPTH CALLBACK PROCESSING FRAME")
            
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
            
            # IMPORTANT: ALWAYS mark as moving camera to ensure continuous updates
            self.camera_motion_detected = True
            self.current_alpha = self.moving_alpha  # Always use the more responsive alpha value
                
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
                
                # Always publish when camera is moving, or at regular intervals when static
                camera_pos = transform.transform.translation
                if self.camera_motion_detected or current_time - self.last_publish_time >= self.publish_period:
                    self.last_publish_time = current_time
                    try:
                        self.publish_occupancy_grid()
                        self.get_logger().info(f"Published grid, camera pos: ({camera_pos.x:.2f}, {camera_pos.y:.2f}, {camera_pos.z:.2f})")
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
            # Return to using the map frame
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
            state = "MOVING" if self.camera_motion_detected else "static"
            self.get_logger().debug(f'Published occupancy grid with frame_id: {grid_msg.header.frame_id} (camera {state})')

    def create_grid_msg_from_log_odds(self):
        """Convert log-odds grid to occupancy probabilities [0,100] - using vectorized operations for speed"""
        # First apply median filter if spatial filtering is enabled
        if self.spatial_filtering:
            # Apply 3x3 median filter to reduce noise
            from scipy.ndimage import median_filter
            filtered_log_odds = median_filter(self.log_odds_grid, size=3)
        else:
            filtered_log_odds = self.log_odds_grid
            
        # Create a copy of the grid to work with
        grid_data = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int8)
        
        # Use numpy vectorized operations for much faster processing
        # Mark cells with insufficient observations as unknown (-1)
        insufficient_obs_mask = self.observation_count_grid < self.min_observations
        grid_data[insufficient_obs_mask] = -1
        
        # Only process cells with sufficient observations
        process_mask = ~insufficient_obs_mask
        
        # Mark occupied cells (100)
        occupied_mask = (filtered_log_odds > self.OCCUPIED_THRESHOLD) & process_mask
        grid_data[occupied_mask] = 100
        
        # Mark free cells (0)
        free_mask = (filtered_log_odds < self.FREE_THRESHOLD) & process_mask
        grid_data[free_mask] = 0
        
        # Convert remaining cells from log-odds to probabilities [0-100]
        # Only process cells that are neither occupied nor free but have sufficient observations
        remaining_mask = process_mask & ~occupied_mask & ~free_mask
        if np.any(remaining_mask):
            # Vectorized conversion from log-odds to probability
            probs = 1.0 - (1.0 / (1.0 + np.exp(filtered_log_odds[remaining_mask])))
            grid_data[remaining_mask] = (probs * 100).astype(np.int8)
        
        # Flatten the grid for the message
        return grid_data.flatten().tolist()
    
    def detect_camera_motion(self, transform):
        """
        Detect if the camera has moved significantly since the last update
        Returns: (is_moving, position_change) tuple - where position_change is the distance moved
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
        position_change = 0.0  # Default if no last position
        
        if self.last_camera_position is not None:
            dx = camera_x - self.last_camera_position[0]
            dy = camera_y - self.last_camera_position[1]
            dz = camera_z - self.last_camera_position[2]
            position_change = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if position_change > self.position_change_threshold:
                position_changed = True
                self.get_logger().info(f"Position changed by {position_change:.4f}m ({dx:.3f}, {dy:.3f}, {dz:.3f})")
                
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
                self.get_logger().info(f"Rotation changed by {angle_diff:.4f} radians")
        
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
        return (camera_moved, position_change)
        
    def save_map_timer_callback(self):
        """Timer callback to automatically save the map at regular intervals"""
        if self.map_persistence_enabled and np.any(self.observation_count_grid > 0):
            current_time = time.time()
            if current_time - self.last_save_time >= self.auto_save_period:
                self.save_map()
                self.last_save_time = current_time
    
    def save_map(self, filename=None):
        """Save the current map to disk"""
        import os
        import numpy as np
        
        with self.grid_lock:
            if not np.any(self.observation_count_grid > 0):
                self.get_logger().info("No map data to save")
                return False
            
            if filename is None:
                # Generate a timestamp-based filename if none provided
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{self.map_file_base}_{timestamp}.npz"
            
            # Create full path
            full_path = os.path.join(self.map_file_directory, filename)
            
            try:
                # Save all grid data to a compressed numpy file
                np.savez_compressed(
                    full_path,
                    log_odds_grid=self.log_odds_grid,
                    cell_height_grid=self.cell_height_grid,
                    observation_count_grid=self.observation_count_grid,
                    resolution=self.resolution,
                    grid_origin_x=self.grid_origin_x,
                    grid_origin_y=self.grid_origin_y,
                    map_frame=self.map_frame
                )
                self.get_logger().info(f"Map saved to {full_path}")
                
                # Also save a latest version
                latest_path = os.path.join(self.map_file_directory, f"{self.map_file_base}_latest.npz")
                np.savez_compressed(
                    latest_path,
                    log_odds_grid=self.log_odds_grid,
                    cell_height_grid=self.cell_height_grid,
                    observation_count_grid=self.observation_count_grid,
                    resolution=self.resolution,
                    grid_origin_x=self.grid_origin_x,
                    grid_origin_y=self.grid_origin_y,
                    map_frame=self.map_frame
                )
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to save map: {e}")
                return False
    
    def load_map(self, filename=None):
        """Load a previously saved map from disk"""
        import os
        import numpy as np
        
        if filename is None:
            # Try to load the latest map by default
            filename = f"{self.map_file_base}_latest.npz"
        
        # Create full path
        full_path = os.path.join(self.map_file_directory, filename)
        
        if not os.path.exists(full_path):
            self.get_logger().info(f"No previous map found at {full_path}")
            return False
        
        try:
            with self.grid_lock:
                # Load the map data
                data = np.load(full_path)
                
                # Check if the grid dimensions match
                loaded_log_odds = data['log_odds_grid']
                if loaded_log_odds.shape != (self.grid_rows, self.grid_cols):
                    self.get_logger().warn(
                        f"Loaded map dimensions ({loaded_log_odds.shape}) don't match current grid " +
                        f"({self.grid_rows}, {self.grid_cols}). Resizing..."
                    )
                    
                    # Resize the loaded map to fit current grid dimensions
                    from scipy.ndimage import zoom
                    
                    # Calculate zoom factors
                    zoom_y = self.grid_rows / loaded_log_odds.shape[0]
                    zoom_x = self.grid_cols / loaded_log_odds.shape[1]
                    
                    # Resize each grid
                    self.log_odds_grid = zoom(loaded_log_odds, (zoom_y, zoom_x), order=1)
                    self.cell_height_grid = zoom(data['cell_height_grid'], (zoom_y, zoom_x), order=1)
                    self.observation_count_grid = zoom(data['observation_count_grid'], (zoom_y, zoom_x), order=0).astype(np.int32)
                else:
                    # Direct assignment if dimensions match
                    self.log_odds_grid = loaded_log_odds
                    self.cell_height_grid = data['cell_height_grid']
                    self.observation_count_grid = data['observation_count_grid']
                
                self.get_logger().info(f"Successfully loaded map from {full_path}")
                self.get_logger().info(f"Map contains {np.sum(self.log_odds_grid > self.OCCUPIED_THRESHOLD)} occupied cells")
                
                # Force an immediate publish of the loaded map
                self.publish_occupancy_grid()
                return True
                
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {e}")
            return False
    
    def pose_callback(self, pose_msg):
        """
        Direct callback for camera pose messages
        This ensures we detect camera movement even when transform detection fails
        """
        from std_msgs.msg import String
        
        # Extract position from the pose message
        pos = pose_msg.pose.position
        
        # Log the position to both ROS log and a debug topic
        position_str = f"POSE_MONITOR: Camera at ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})"
        self.get_logger().info(position_str)
        
        # Publish to debug topic for external monitoring
        debug_msg = String()
        debug_msg.data = position_str
        self.debug_pub.publish(debug_msg)
        
        # Calculate position change if we have a previous position
        if self.last_camera_position is not None:
            dx = pos.x - self.last_camera_position[0]
            dy = pos.y - self.last_camera_position[1]
            dz = pos.z - self.last_camera_position[2]
            position_change = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Just detect movement, but don't reset the map for persistence
            if position_change > 0.001:  # Use a slightly higher threshold for pose
                self.camera_motion_detected = True
                self.get_logger().info(f"POSE_CALLBACK: Movement detected: {position_change:.6f}m")
                # Save map on significant movement
                if position_change > 0.1:  # Save map on larger movements
                    self.save_map()
                
        # Update last position
        self.last_camera_position = (pos.x, pos.y, pos.z)
        
    def update_grid_from_depth(self, depth_image, transform):
        """
        Update the log-odds grid using the depth image and camera transform.
        Uses probabilistic updates for better map quality.
        """
        # Get motion information first so we can use position_change
        is_moving, position_change = self.detect_camera_motion(transform)
        
        # For persistent mapping, we want to PRESERVE the grid and only update incrementally
        # Never reset, even on significant motion
        if position_change > self.position_change_threshold * 10:  # Significant movement
            self.get_logger().info(f"*** SIGNIFICANT MOVEMENT DETECTED - PRESERVING GRID ***")
            # Don't decay values, just continue accumulating observations
            # This is critical for map persistence during exploration
        else:
            self.get_logger().debug("Preserving existing grid and applying updates")
        
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
        
        # ALWAYS force the camera to be considered in motion to ensure continuous updates
        self.camera_motion_detected = True
        # Use extremely aggressive filtering for immediate updates
        self.current_alpha = 0.05  # Super aggressive value for immediate feedback
        
        # Force debug output
        self.get_logger().error(f"!!!! UPDATING GRID - CAMERA AT ({camera_x:.4f}, {camera_y:.4f}, {camera_z:.4f}) !!!!")
        
        # This is just for debug info
        is_moving, position_change = self.detect_camera_motion(transform)
        
        # Debug - explicitly log current camera position every time we process data
        self.get_logger().info(f"CAMERA MONITOR - Current position: ({camera_x:.4f}, {camera_y:.4f}, {camera_z:.4f})")
        
        # Choose sampling density - use more aggressive sampling for faster updates
        # Increase stride to process fewer pixels but maintain reasonable detail
        stride_y = 10  # Coarser sampling for faster processing
        stride_x = 10  # Coarser sampling for faster processing
        
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
                
            # When camera is moving, give extra weight to new observations and reduce weight of old observations
            # We're always considering the camera to be moving now, so this code will always run
            # Debug logs to track exact motion values
            self.get_logger().info(f"MOTION DEBUG: position_change={position_change:.6f}, threshold={self.position_change_threshold}")
            self.get_logger().info(f"MOTION DEBUG: camera at ({camera_x:.3f}, {camera_y:.3f}, {camera_z:.3f})")
            
            # Always update regardless of movement magnitude
            # CRITICAL: We need a new approach to make the map update when the camera moves
            # The issue is that the occupancy grid isn't seeing the camera position changes
            
            # Completely replace existing logic with a more aggressive approach
            
            # STEP 1: Always update *all* grid cells that we see in this frame
            occupied_mask = (current_update == self.LOG_ODDS_OCCUPIED)
            if np.any(occupied_mask):
                # HARD RESET occupied cells - don't blend, just set them
                self.log_odds_grid[occupied_mask] = self.LOG_ODDS_MAX
                num_occupied = np.sum(occupied_mask)
                self.get_logger().error(f"RESET {num_occupied} occupied cells to MAX")
            
            # STEP 2: Much more aggressive free cell handling
            free_mask = (current_update == self.LOG_ODDS_FREE)
            if np.any(free_mask):
                # Just set all free cells to minimum - no blending
                self.log_odds_grid[free_mask] = self.LOG_ODDS_MIN
                num_free = np.sum(free_mask)
                self.get_logger().error(f"RESET {num_free} free cells to MIN")
            
            # STEP 3: Increment a frame counter for unique maps
            if not hasattr(self, 'frame_counter'):
                self.frame_counter = 0
            self.frame_counter += 1
            
            # STEP 4: Force an immediate publish with unique timestamp to ensure update
            current_time = self.get_clock().now().to_msg()
            
            # STEP 5: Always republish even if nothing changed
            try:
                self.publish_occupancy_grid()
                self.get_logger().error(f"FORCED MAP UPDATE - Frame {self.frame_counter}")
            except Exception as ex:
                self.get_logger().error(f"Failed to publish: {ex}")
                
            # Save map periodically
            if self.frame_counter % 10 == 0:  # Every 10 frames
                self.save_map()
            
            # Clamp log-odds values to prevent numerical issues
            self.log_odds_grid = np.clip(self.log_odds_grid, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX)

    def check_tf_callback(self):
        """
        Regularly check TF to detect camera movement independently
        This ensures we catch all camera movements even if pose callbacks are missing
        """
        try:
            # Try to get the current camera position from TF
            transform = self.tf_buffer.lookup_transform(
                'map',  # Try map first
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Force camera to be considered moving for immediate updates
            self.camera_motion_detected = True
            
            # Get the position for logging
            camera_x = transform.transform.translation.x
            camera_y = transform.transform.translation.y
            camera_z = transform.transform.translation.z
            
            # Log position for debugging
            self.get_logger().warn(f"TF CHECK - Camera at ({camera_x:.4f}, {camera_y:.4f}, {camera_z:.4f})")
            
            # Process a depth image if we have one
            self.force_depth_update()
            
        except Exception as e:
            # Try odom frame as fallback
            try:
                transform = self.tf_buffer.lookup_transform(
                    'odom',
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                
                # Still force camera to be moving
                self.camera_motion_detected = True
                
                # Get position for logging
                camera_x = transform.transform.translation.x
                camera_y = transform.transform.translation.y
                camera_z = transform.transform.translation.z
                
                # Log as warning for visibility
                self.get_logger().warn(f"TF CHECK (odom) - Camera at ({camera_x:.4f}, {camera_y:.4f}, {camera_z:.4f})")
                
                # Try to process a depth image
                self.force_depth_update()
                
            except Exception as e2:
                # Just log the error
                pass
    
    def force_update_callback(self):
        """
        Force periodic grid updates regardless of camera movement
        Ensures the map is constantly updated even when the camera appears static
        """
        # Force motion flag to ensure grid updates
        self.camera_motion_detected = True
        
        # Log this forced update
        self.get_logger().warn("FORCED UPDATE - Ensuring continuous map building")
        
        # Force publishing the grid
        if np.any(self.observation_count_grid > 0):
            self.publish_occupancy_grid()
            
        # Try to force a depth update
        self.force_depth_update()
    
    def force_depth_update(self):
        """
        Attempt to trigger a depth update using the last known transform
        This helps ensure continuous map updates
        """
        try:
            # Try to get a transform to use for updating
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Only process if we got a valid transform
            if hasattr(transform, 'transform'):
                # Log that we're forcing an update
                camera_x = transform.transform.translation.x
                camera_y = transform.transform.translation.y
                camera_z = transform.transform.translation.z
                
                self.get_logger().warn(f"FORCING DEPTH UPDATE - Camera at ({camera_x:.4f}, {camera_y:.4f}, {camera_z:.4f})")
                
                # Force camera to be considered moving
                self.camera_motion_detected = True
                
                # Update any previous position to trigger movement detection on next update
                # This ensures that even small movements will be detected
                if self.last_camera_position is not None:
                    # Add a slight offset to ensure the next position will be seen as changed
                    offset = 0.002  # 2mm offset to force detection above threshold
                    self.last_camera_position = (
                        self.last_camera_position[0] + offset,
                        self.last_camera_position[1] + offset,
                        self.last_camera_position[2]
                    )
                    self.get_logger().info(f"Adjusted position reference to force movement detection")
                
                # Force a map update
                self.publish_occupancy_grid()
        except Exception as e:
            # Just log at debug level since this is a background operation
            self.get_logger().debug(f"Could not force depth update: {e}")
            
    def camera_monitor_callback(self):
        """Timer callback that regularly checks and logs camera position regardless of movement"""
        try:
            # Try to get transform from camera to map or odom
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)  # Short timeout
                )
                # Use map frame
                frame = self.map_frame
            except Exception:
                # Fall back to odom if map isn't available
                transform = self.tf_buffer.lookup_transform(
                    'odom',
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                frame = 'odom'
            
            # Extract camera position
            pos = transform.transform.translation
            
            # Always log camera position, even when not moving
            from std_msgs.msg import String
            position_str = f"TIMER_MONITOR: Camera at ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}) in {frame} frame"
            self.get_logger().info(position_str)
            
            # Publish to debug topic for external monitoring
            debug_msg = String()
            debug_msg.data = position_str
            self.debug_pub.publish(debug_msg)
            
            # Don't reset grid from the timer - just log and monitor
            # Force a publish only if we have data
            if np.any(self.observation_count_grid > 0):
                self.publish_occupancy_grid(override_frame=frame)
            
        except Exception as e:
            self.get_logger().error(f"Error in camera monitor timer: {e}")
    
    def mark_cells_in_view(self, start_grid_x, start_grid_y, end_grid_x, end_grid_y, in_view_mask):
        """
        Marks cells along a ray as being in view using Bresenham's line algorithm.
        Similar to mark_ray_as_free_3d but just marks cells as visible without changing log-odds.
        Used to track which cells are currently in the camera's field of view.
        """
        # Bresenham's line algorithm
        dx = abs(end_grid_x - start_grid_x)
        dy = abs(end_grid_y - start_grid_y)
        sx = 1 if start_grid_x < end_grid_x else -1
        sy = 1 if start_grid_y < end_grid_y else -1
        err = dx - dy
        
        x, y = start_grid_x, start_grid_y
        
        # For each point along the line including the endpoint
        while True:
            if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                # Mark as in view
                in_view_mask[y, x] = True
            
            # Check if we've reached the endpoint
            if x == end_grid_x and y == end_grid_y:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def mark_ray_as_free_3d(self, start_x, start_y, end_x, end_y, current_update):
        """
        Optimized version that marks cells along a ray as free using a faster Bresenham implementation.
        Uses ray skipping and batch updates for better performance.
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
        
        # Skip ray casting for very short rays (optimization)
        if abs(end_grid_x - start_grid_x) <= 1 and abs(end_grid_y - start_grid_y) <= 1:
            return
            
        # Use ray skipping: mark every few cells instead of every cell
        # This significantly speeds up ray casting while maintaining a good map
        ray_skip = 2  # Mark every 2nd cell
        
        # Bresenham's line algorithm - optimized version
        dx = abs(end_grid_x - start_grid_x)
        dy = abs(end_grid_y - start_grid_y)
        sx = 1 if start_grid_x < end_grid_x else -1
        sy = 1 if start_grid_y < end_grid_y else -1
        err = dx - dy
        
        x, y = start_grid_x, start_grid_y
        steps = 0
        
        # Collect cells to mark as free in batches for better performance
        cells_to_mark = []
        
        # For each point along the line except the endpoint (which might be occupied)
        while (x != end_grid_x or y != end_grid_y):
            # Only mark cells at specified skip intervals (optimization)
            if steps % ray_skip == 0:
                if 0 <= x < self.grid_cols and 0 <= y < self.grid_rows:
                    cells_to_mark.append((y, x))
            
            steps += 1
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        # Batch update the cells for better performance
        if cells_to_mark:
            # Extract y and x coordinates
            y_coords, x_coords = zip(*cells_to_mark)
            
            # Update cells in batch operations
            current_update[y_coords, x_coords] = self.LOG_ODDS_FREE
            self.observation_count_grid[y_coords, x_coords] += 1


def main(args=None):
    rclpy.init(args=args)
    node = ZedOccupancyGridNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
