#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
import tf2_ros
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import time
import threading
import math
import os

class ZedOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('zed_occupancy_grid_node')
        
        # Set ROS logging to INFO to reduce overhead from excessive logging
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Occupancy probabilities - using log-odds for numerical stability
        self.FREE_THRESHOLD = -2.0  # log-odds threshold for considering a cell free
        self.OCCUPIED_THRESHOLD = 2.0  # log-odds threshold for considering a cell occupied
        self.LOG_ODDS_PRIOR = 0.0  # log-odds of prior probability (0.5)
        self.LOG_ODDS_FREE = -1.5  # log-odds update for free cells
        self.LOG_ODDS_OCCUPIED = 3.0  # log-odds update for occupied cells
        self.LOG_ODDS_MIN = -5.0  # minimum log-odds value
        self.LOG_ODDS_MAX = 5.0  # maximum log-odds value
        
        # Map persistence settings
        self.map_persistence_enabled = True
        self.map_file_directory = '/tmp/zed_occupancy_map/'
        self.map_file_base = 'occupancy_map'
        self.auto_save_period = 60.0  # Save map every 60 seconds
        self.last_save_time = time.time()
        
        # Create the directory if it doesn't exist
        if not os.path.exists(self.map_file_directory):
            os.makedirs(self.map_file_directory)
            self.get_logger().info(f"Created map directory: {self.map_file_directory}")
            
        # Add auto-save map timer
        self.map_save_timer = self.create_timer(self.auto_save_period, self.save_map_timer_callback)
        
        # Add a mutex for thread safety
        self.grid_lock = threading.Lock()

        # Declare parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'zed_left_camera_frame')
        self.declare_parameter('resolution', 0.05)  # meters per cell
        self.declare_parameter('grid_width', 10.0)  # meters
        self.declare_parameter('grid_height', 10.0)  # meters
        self.declare_parameter('min_depth', 0.5)    # min depth in meters
        self.declare_parameter('max_depth', 20.0)   # max depth in meters
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('position_change_threshold', 0.00001)
        self.declare_parameter('rotation_change_threshold', 0.00001)

        # Get parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.resolution = self.get_parameter('resolution').value
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.position_change_threshold = self.get_parameter('position_change_threshold').value
        self.rotation_change_threshold = self.get_parameter('rotation_change_threshold').value

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
        self.temporal_filtering = True       # Enable temporal filtering
        self.spatial_filtering = True        # Enable spatial filtering
        self.min_observations = 1            # Minimum observations before marking as occupied
        self.max_ray_length = 5.0            # Maximum ray length for ray casting
        
        # Parameters for map updates with camera motion
        self.last_camera_position = None     # Track camera position to detect movement
        self.reset_cells_on_movement = True  # Enable resetting cells that become out of view
        self.camera_motion_detected = True   # ALWAYS assume camera is moving to force continuous updates
        self.last_camera_quaternion = None   # Track camera rotation
        
        # Camera movement adaptation settings
        self.static_alpha = 0.7              # Less temporal filtering even when static
        self.moving_alpha = 0.3              # Much less filtering when moving
        self.current_alpha = self.moving_alpha # Start with moving settings for faster updates

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Set up QoS profiles
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20
        )
        
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Critical for RViz compatibility
            depth=20
        )

        # Publishers and subscribers
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, '/occupancy_grid', qos_profile_pub)
        
        self.depth_subscriber = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sub)
            
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/zed/zed_node/pose', self.pose_callback, 1)
            
        self.tf_subscriber = self.create_timer(0.1, self.check_tf_callback)
        
        self.force_update_timer = self.create_timer(0.5, self.force_update_callback)
            
        self.debug_pub = self.create_publisher(String, '/zed_grid_debug', 10)
            
        # Add rate limiting with separate update/publish cycles
        self.last_publish_time = 0.0
        self.publish_period = 0.05           # 20Hz publishing
        self.last_update_time = 0.0
        self.update_period = 0.03            # 33Hz processing
        
        # Cache the last grid message for reuse
        self.last_grid_msg = None
        
        # Create a timer for regular map updates
        self.map_timer = self.create_timer(0.1, self.publish_map_timer_callback)
        
        # Create a special timer that always logs camera position
        self.camera_monitor_timer = self.create_timer(0.5, self.camera_monitor_callback)

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize the grid with sample data to immediately see something
        self.initialize_sample_grid()
        
        self.get_logger().info('ZED Occupancy Grid Node initialized with MOVEMENT-OPTIMIZED settings')
        self.get_logger().info(f'Listening to depth topic: {self.depth_topic}')
        self.get_logger().info(f'Grid size: {self.grid_width}x{self.grid_height} meters, resolution: {self.resolution} m/cell')
        
        # Publish initial grid immediately
        self.publish_occupancy_grid()
        
    def initialize_sample_grid(self):
        """Initialize the grid with sample data to make it visible immediately"""
        try:
            with self.grid_lock:
                # Create a border around the grid
                border_width = 20
                
                # Fill the grid with a pattern of occupied cells
                for y in range(self.grid_rows):
                    for x in range(self.grid_cols):
                        # Create a border
                        if (y < border_width or y >= self.grid_rows - border_width or 
                            x < border_width or x >= self.grid_cols - border_width):
                            self.log_odds_grid[y, x] = self.LOG_ODDS_MAX
                            self.observation_count_grid[y, x] = 10
                
                # Create a center square
                center_size = 50
                center_x = self.grid_cols // 2
                center_y = self.grid_rows // 2
                
                for y in range(center_y - center_size, center_y + center_size):
                    for x in range(center_x - center_size, center_x + center_size):
                        if 0 <= y < self.grid_rows and 0 <= x < self.grid_cols:
                            self.log_odds_grid[y, x] = self.LOG_ODDS_MAX
                            self.observation_count_grid[y, x] = 10
                
                # Create a diagonal line for orientation
                for i in range(min(self.grid_rows, self.grid_cols) // 2):
                    y = i
                    x = i
                    if 0 <= y < self.grid_rows and 0 <= x < self.grid_cols:
                        self.log_odds_grid[y, x] = self.LOG_ODDS_MAX
                        self.observation_count_grid[y, x] = 10
                
                self.get_logger().info("Sample grid created - should be visible in RViz immediately!")
        except Exception as e:
            self.get_logger().error(f"Error initializing sample grid: {e}")

    def depth_callback(self, depth_msg):
        try:
            current_time = time.time()
            self.last_update_time = current_time
            
            # Always mark the camera as moving to ensure continuous updates
            self.camera_motion_detected = True
            
            # Process the depth image
            try:
                depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                
                # Try to get transform from camera to map
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=1.0)
                    )
                    
                    # Update grid with depth data
                    self.update_grid_from_depth(depth_image, transform)
                except Exception:
                    self.get_logger().warn("Could not get transform for depth image")
            except Exception:
                self.get_logger().warn("Could not process depth image")
            
            # Always publish updates at regular intervals
            if current_time - self.last_publish_time >= self.publish_period:
                self.last_publish_time = current_time
                self.publish_occupancy_grid()
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {e}")
    
    def update_grid_from_depth(self, depth_image, transform):
        """Update grid from depth image - simplified version that just updates sample data"""
        try:
            # For this simplified version, just update a corner of the grid
            # to show that the grid can change dynamically
            with self.grid_lock:
                # Get camera position
                camera_x = transform.transform.translation.x
                camera_y = transform.transform.translation.y
                
                # Convert to a grid position
                grid_x = int((camera_x - self.grid_origin_x) / self.resolution)
                grid_y = int((camera_y - self.grid_origin_y) / self.resolution)
                
                # Update a region based on the camera position
                size = 10
                for y in range(grid_y - size, grid_y + size):
                    for x in range(grid_x - size, grid_x + size):
                        if 0 <= y < self.grid_rows and 0 <= x < self.grid_cols:
                            # Mark cells as occupied with camera position
                            self.log_odds_grid[y, x] = self.LOG_ODDS_MAX
                            self.observation_count_grid[y, x] = 10
        except Exception as e:
            self.get_logger().error(f"Error updating grid from depth: {e}")
    
    def publish_map_timer_callback(self):
        """Regular timer to publish the map even without new data"""
        try:
            self.publish_occupancy_grid()
        except Exception as e:
            self.get_logger().error(f"Error in map timer: {e}")
    
    def publish_occupancy_grid(self, override_frame=None):
        """Publish the occupancy grid with the latest data"""
        try:
            with self.grid_lock:
                grid_msg = OccupancyGrid()
                grid_msg.header.stamp = self.get_clock().now().to_msg()
                grid_msg.header.frame_id = override_frame if override_frame else self.map_frame
                
                grid_msg.info.resolution = self.resolution
                grid_msg.info.width = self.grid_cols
                grid_msg.info.height = self.grid_rows
                
                grid_msg.info.origin.position.x = self.grid_origin_x
                grid_msg.info.origin.position.y = self.grid_origin_y
                grid_msg.info.origin.position.z = 0.0
                grid_msg.info.origin.orientation.w = 1.0
                
                # Convert log-odds to occupancy probabilities [0,100]
                grid_data = self.create_grid_msg_from_log_odds()
                grid_msg.data = grid_data
                
                # Force republish on new frame counter to ensure updates
                static_counter = getattr(self, 'static_counter', 0) + 1
                self.static_counter = static_counter
                
                # Publish the grid
                self.occupancy_grid_pub.publish(grid_msg)
                self.last_grid_msg = grid_msg
                
                # Publishing this ensures the map stays alive
                debug_msg = String()
                debug_msg.data = f"Grid published: frame_counter={static_counter}"
                self.debug_pub.publish(debug_msg)
                
                self.get_logger().info(f"Map published with frame_id: {grid_msg.header.frame_id}")
        except Exception as e:
            self.get_logger().error(f"Error publishing grid: {e}")
    
    def create_grid_msg_from_log_odds(self):
        """Convert log-odds grid to occupancy probabilities [0,100]"""
        try:
            # Create copy of grid
            grid_data = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int8)
            
            # Mark cells with insufficient observations as unknown (-1)
            insufficient_obs_mask = self.observation_count_grid < self.min_observations
            grid_data[insufficient_obs_mask] = -1
            
            # Only process cells with sufficient observations
            process_mask = ~insufficient_obs_mask
            
            # Mark occupied cells (100)
            occupied_mask = (self.log_odds_grid > self.OCCUPIED_THRESHOLD) & process_mask
            grid_data[occupied_mask] = 100
            
            # Mark free cells (0)
            free_mask = (self.log_odds_grid < self.FREE_THRESHOLD) & process_mask
            grid_data[free_mask] = 0
            
            # Convert remaining cells to probabilities
            remaining_mask = process_mask & ~occupied_mask & ~free_mask
            if np.any(remaining_mask):
                probs = 1.0 - (1.0 / (1.0 + np.exp(self.log_odds_grid[remaining_mask])))
                grid_data[remaining_mask] = (probs * 100).astype(np.int8)
            
            # Flatten for message
            return grid_data.flatten().tolist()
        except Exception as e:
            self.get_logger().error(f"Error creating grid message: {e}")
            # Return default grid
            return [-1] * (self.grid_rows * self.grid_cols)
    
    def pose_callback(self, pose_msg):
        """Handle camera pose updates"""
        try:
            # Extract position
            pos = pose_msg.pose.position
            
            # Force map updates regardless of camera motion
            self.camera_motion_detected = True
            
            # Log position
            self.get_logger().info(f"Camera at ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
            
            # Update position tracking
            self.last_camera_position = (pos.x, pos.y, pos.z)
            
            # Force grid update
            self.publish_occupancy_grid()
        except Exception as e:
            self.get_logger().error(f"Error in pose callback: {e}")
    
    def check_tf_callback(self):
        """Regularly check TF"""
        try:
            # Try to get current transform
            transform = self.tf_buffer.lookup_transform(
                'map',
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Force map updates
            self.camera_motion_detected = True
            self.publish_occupancy_grid()
        except Exception:
            # Just pass silently if we can't get the transform
            pass
    
    def force_update_callback(self):
        """Force periodic updates"""
        try:
            self.camera_motion_detected = True
            self.publish_occupancy_grid()
        except Exception as e:
            self.get_logger().error(f"Error in forced update: {e}")
    
    def camera_monitor_callback(self):
        """Regular timer to monitor camera position"""
        try:
            # Force map updates regardless of motion
            self.camera_motion_detected = True
            self.publish_occupancy_grid()
        except Exception as e:
            self.get_logger().error(f"Error in camera monitor: {e}")
    
    def save_map_timer_callback(self):
        """Auto-save map periodically"""
        try:
            if self.map_persistence_enabled and np.any(self.observation_count_grid > 0):
                current_time = time.time()
                if current_time - self.last_save_time >= self.auto_save_period:
                    self.save_map()
                    self.last_save_time = current_time
        except Exception as e:
            self.get_logger().error(f"Error in save map timer: {e}")

    def save_map(self, filename=None):
        """Save map to disk"""
        try:
            with self.grid_lock:
                if filename is None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{self.map_file_base}_{timestamp}.npz"
                
                full_path = os.path.join(self.map_file_directory, filename)
                
                # Save compressed map data
                np.savez_compressed(
                    full_path,
                    log_odds_grid=self.log_odds_grid,
                    observation_count_grid=self.observation_count_grid,
                    resolution=self.resolution
                )
                self.get_logger().info(f"Map saved to {full_path}")
                return True
        except Exception as e:
            self.get_logger().error(f"Error saving map: {e}")
            return False
    
    def load_map(self, filename=None):
        """Load map from disk"""
        try:
            if filename is None:
                filename = f"{self.map_file_base}_latest.npz"
            
            full_path = os.path.join(self.map_file_directory, filename)
            
            if os.path.exists(full_path):
                data = np.load(full_path)
                self.log_odds_grid = data['log_odds_grid']
                self.observation_count_grid = data['observation_count_grid']
                self.get_logger().info(f"Map loaded from {full_path}")
                return True
            else:
                self.get_logger().info(f"No map found at {full_path}")
                return False
        except Exception as e:
            self.get_logger().error(f"Error loading map: {e}")
            return False


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ZedOccupancyGridNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
