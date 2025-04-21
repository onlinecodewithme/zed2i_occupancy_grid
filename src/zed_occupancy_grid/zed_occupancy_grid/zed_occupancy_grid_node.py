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

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'

# CUDA and GPU acceleration imports
try:
    from . import cuda_acceleration
    CUDA_AVAILABLE = cuda_acceleration.CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        print(f"\n{GREEN}{BOLD}========================================{END}")
        print(f"{GREEN}{BOLD}✓ CUDA ACCELERATION ENABLED AND ACTIVE ✓{END}")
        print(f"{GREEN}{BOLD}========================================{END}")
        print(f"{GREEN}✓ CUDA acceleration module imported successfully!{END}")
    else:
        print(f"\n{YELLOW}{BOLD}=========================================={END}")
        print(f"{YELLOW}{BOLD}⚠ CUDA DISABLED - USING CPU PROCESSING ⚠{END}")
        print(f"{YELLOW}{BOLD}=========================================={END}")
        print(f"{YELLOW}⚠ CUDA module imported but GPU not available!{END}")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"\n{RED}{BOLD}=========================================={END}")
    print(f"{RED}{BOLD}❌ CUDA UNAVAILABLE - USING CPU FALLBACK ❌{END}")
    print(f"{RED}{BOLD}=========================================={END}")
    print(f"{RED}❌ Error: {str(e)}{END}")
    print(f"{RED}❌ Falling back to CPU processing (slower){END}")

class ZedOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('zed_occupancy_grid_node')

        # Check CUDA availability for GPU acceleration
        if CUDA_AVAILABLE:
            self.get_logger().info("CUDA GPU acceleration is ENABLED - Using JETSON ORIN NX Acceleration!")
            # Initialize CUDA context and compile kernels
            self.initialize_cuda_functions()
        else:
            self.get_logger().warn("CUDA is not available - using CPU fallback")
    
    def initialize_cuda_functions(self):
        """Initialize CUDA accelerator for GPU-accelerated processing"""
        global CUDA_AVAILABLE
        
        if not CUDA_AVAILABLE:
            return
        
        try:
            # Initialize our optimized CUDA accelerator
            from . import cuda_acceleration
            
            # Create the accelerator instance with our logger
            self.cuda_accelerator = cuda_acceleration.CudaAccelerator(self.get_logger())
            
            # Check if CUDA is really available
            if self.cuda_accelerator.is_available():
                self.get_logger().info("CUDA GPU acceleration initialized successfully with optimal settings")
                
                # Add CUDA-specific parameters
                self.declare_parameter('use_cuda', True)
                self.declare_parameter('cuda_step', 4)  # Step size for CUDA processing (higher = faster but less detail)
                self.declare_parameter('cuda_ray_step', 2)  # Step size for ray tracing on GPU
                
                # Get parameters
                self.use_cuda = self.get_parameter('use_cuda').value
                self.cuda_step = self.get_parameter('cuda_step').value
                self.cuda_ray_step = self.get_parameter('cuda_ray_step').value
                
                self.get_logger().info(f"CUDA Parameters: use_cuda={self.use_cuda}, cuda_step={self.cuda_step}, cuda_ray_step={self.cuda_ray_step}")
            else:
                self.get_logger().error("CUDA accelerator initialized but reports CUDA is not available")
                CUDA_AVAILABLE = False
            
        except Exception as e:
            self.get_logger().error(f"Error initializing CUDA functions: {e}")
            self.get_logger().warn("Falling back to CPU-based processing")
            CUDA_AVAILABLE = False
        
        # Set ROS logging to DEBUG to get all messages
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        # WALL-BASED MAP: Optimized parameters for solid walls and navigation
        # P(occupied) = 1 - 1/(1 + exp(l))
        self.FREE_THRESHOLD = -1.5     # Higher threshold to better distinguish free space
        self.OCCUPIED_THRESHOLD = 1.0   # Higher threshold for more confident obstacle marking
        self.LOG_ODDS_PRIOR = 0.0       # log-odds of prior probability (0.5)
        self.LOG_ODDS_FREE = -0.8       # Stronger free space marking to create clear corridors
        self.LOG_ODDS_OCCUPIED = 3.5    # Stronger obstacle marking for solid walls
        self.LOG_ODDS_MIN = -5.0        # Standard minimum
        self.LOG_ODDS_MAX = 5.0         # Standard maximum
        
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
        self.position_change_threshold = 0.0001  # MODIFIED: Small but non-zero threshold for stable detection
        self.declare_parameter('rotation_change_threshold', 0.0001)  # Ultra sensitive
        self.rotation_change_threshold = 0.0001  # MODIFIED: Small but non-zero threshold for stable detection
        self.reset_cells_on_movement = True   # Enable resetting cells that become out of view
        self.camera_motion_detected = True    # MODIFIED: Always consider camera in motion to ensure updates
        self.last_camera_quaternion = None    # Track camera rotation
        self.get_logger().info("!!! ALWAYS UPDATING OCCUPANCY GRID - CAMERA ALWAYS CONSIDERED MOVING !!!")
        
        # Camera movement adaptation settings - SUPER RESPONSIVE
        self.static_alpha = 0.3              # MODIFIED: Less temporal filtering when static for faster updates
        self.moving_alpha = 0.1              # MODIFIED: Almost no temporal filtering when moving for immediate updates
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
        from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
        from sensor_msgs.msg import Imu
        from nav_msgs.msg import Odometry
        
        # !!! EMERGENCY FIX: Try ALL possible movement-related topics at highest QoS
        # Subscribe with ABSOLUTE HIGHEST PRIORITY to ALL possible movement topics
        self.pose_subscriber = self.create_subscription(
            PoseStamped, '/zed/zed_node/pose', self.pose_callback, 1)
            
        self.pose_subscriber2 = self.create_subscription(
            PoseStamped, '/zed/pose', self.pose_callback, 1)
            
        self.pose_subscriber3 = self.create_subscription(
            PoseStamped, '/zed2i/zed_node/pose', self.pose_callback, 1)
            
        # Add odometry subscriptions which may be more reliable
        self.odom_subscriber = self.create_subscription(
            Odometry, '/zed/zed_node/odom', self.odom_callback, 1)
            
        self.odom_subscriber2 = self.create_subscription(
            Odometry, '/zed/odom', self.odom_callback, 1)
            
        # Add IMU subscriptions to detect movement from acceleration data
        self.imu_subscriber = self.create_subscription(
            Imu, '/zed/zed_node/imu/data', self.imu_callback, 1)
            
        # Add velocity subscriptions to detect commanded movement
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/cmd_vel', self.twist_callback, 1)
            
        # Try one more subscriber with raw camera name
        self.pose_subscriber4 = self.create_subscription(
            PoseStamped, '/zed_node/pose', self.pose_callback, 1)
            
        # Create a FORCED camera-moving timer that always assumes camera is moving
        # This ensures we get updates even if movement detection fails
        self.forced_motion_timer = self.create_timer(0.25, self.forced_movement_callback)
            
        # Add debug logging for each subscription
        self.get_logger().error("EMERGENCY FIX: Subscribed to ALL possible movement detection topics")
            
        # Track TF directly for additional movement detection
        self.tf_subscriber = self.create_timer(0.1, self.check_tf_callback)  # 10Hz TF checks
        
        # Add high-frequency forced updates
        self.force_update_timer = self.create_timer(0.5, self.force_update_callback)  # MODIFIED: Force update every 0.5 seconds
            
        # Create a publisher for debug info
        from std_msgs.msg import String
        self.debug_pub = self.create_publisher(String, '/zed_grid_debug', 10)
            
        # Add rate limiting with separate update/publish cycles - MAXIMUM SPEED UPDATES
        self.last_publish_time = 0.0
        self.publish_period = 0.05           # MODIFIED: 20Hz publishing for instant responsiveness
        self.last_update_time = 0.0
        self.update_period = 0.01            # MODIFIED: 100Hz processing for immediate updates
        
        # Add a mutex for thread safety
        self.grid_lock = threading.Lock()
        
        # Cache the last grid message for reuse
        self.last_grid_msg = None
        
        # Maximum speed settings
        self.spatial_filtering = False  # DISABLED spatial filtering for maximum speed 
        
        # Create a timer for ULTRA-HIGH FREQUENCY map updates (even with no new data)
        self.map_timer = self.create_timer(0.02, self.publish_map_timer_callback)  # MODIFIED: 50Hz timer for real-time updates
        
        # Create a special timer that always logs camera position regardless of movement
        self.camera_monitor_timer = self.create_timer(0.2, self.camera_monitor_callback)  # MODIFIED: 5Hz timer for more frequent checks

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
            
            # CRITICAL: Always consider camera in motion to ensure updates
            self.camera_motion_detected = True
            
            # Convert depth image to OpenCV format (float32 in meters)
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            # Store the depth image for use in pose callbacks (allows immediate grid updates)
            self.latest_depth_image = depth_image
            
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
                
            # Additional check for available TF frames to help with debugging
            try:
                frames = self.tf_buffer.all_frames_as_string()
                self.get_logger().debug(f"Available TF frames: {frames}")
            except Exception as e:
                self.get_logger().warning(f"Error getting TF frames: {e}")
                
            # Process depth data and update occupancy grid
            self.process_depth_data(depth_image, current_time)
                
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
            
    def process_depth_data(self, depth_image, current_time):
        """Process depth image and update occupancy grid"""
        try:
            # First attempt to get camera-to-map transform directly
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            
            # Log success - this is important for debugging
            self.get_logger().info(f"SUCCESS: Found direct transform from {self.map_frame} to {self.camera_frame}")
            
            # Update the occupancy grid using depth image and transform
            self.update_grid_from_depth(depth_image, transform)
            
            # Always publish when camera is moving, or at regular intervals when static
            camera_pos = transform.transform.translation
            # MODIFIED: Always publish on every depth frame for instant updates
            self.last_publish_time = current_time
            try:
                self.publish_occupancy_grid()
                self.get_logger().info(f"Published grid, camera pos: ({camera_pos.x:.2f}, {camera_pos.y:.2f}, {camera_pos.z:.2f})")
            except Exception as e:
                self.get_logger().error(f"Error publishing occupancy grid: {e}")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f'Direct TF lookup failed: {str(e)}')
            
            # First fallback: Try to construct the transform chain manually
            try:
                # Try to get transform from odom to camera
                self.get_logger().info(f'Attempting to manually construct transform chain')
                
                # Get odom to camera transform
                odom_to_camera = self.tf_buffer.lookup_transform(
                    'odom',
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5)
                )
                
                # Get map to odom transform
                map_to_odom = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    'odom',
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5)
                )
                
                # Combine the transforms (simplified approach)
                transform = odom_to_camera
                transform.header.frame_id = self.map_frame
                
                # Add the translations - this is a simplified approach
                transform.transform.translation.x += map_to_odom.transform.translation.x
                transform.transform.translation.y += map_to_odom.transform.translation.y
                transform.transform.translation.z += map_to_odom.transform.translation.z
                
                # We should properly combine the rotations using quaternion multiplication
                # But for simplicity, we'll just use the original camera rotation for now
                
                self.get_logger().info(f'Successfully constructed combined transform chain')
                self.update_grid_from_depth(depth_image, transform)
                self.last_publish_time = current_time
                self.publish_occupancy_grid()
                
            except Exception as e1:
                self.get_logger().warning(f'Manual transform chain failed: {str(e1)}')
                
                # Second fallback: Try using just odom frame
                try:
                    self.get_logger().info(f'Trying fallback: lookup transform from odom to {self.camera_frame}')
                    transform = self.tf_buffer.lookup_transform(
                        'odom',
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.5)
                    )
                    # We got odom to camera, use it with odom as frame
                    self.get_logger().info(f'Got transform from odom to {self.camera_frame}, using as fallback')
                    self.update_grid_from_depth(depth_image, transform)
                    
                    # MODIFIED: Always publish when we get a new depth frame
                    self.last_publish_time = current_time
                    # Override map frame with odom for this publish
                    self.publish_occupancy_grid(override_frame='odom')
                except Exception as e2:
                    self.get_logger().error(f'All transform fallbacks failed: {str(e2)}')
                
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
        """Convert log-odds grid to occupancy grid message data - VECTORIZED for maximum performance"""
        # Use NumPy vectorized operations for massive speedup
        import numpy as np
        from scipy.ndimage import median_filter

        # Start with a grid of unknown cells (-1)
        grid_data = np.full(self.grid_rows * self.grid_cols, -1, dtype=np.int8)
        
        # Get mask of cells with enough observations
        valid_mask = self.observation_count_grid >= self.min_observations
        
        # Get filtered log-odds only for valid cells (avoid filtering the entire grid)
        if self.spatial_filtering:
            # Only filter regions with observations for better performance
            regions_with_obs = np.where(valid_mask)
            min_y, max_y = max(0, np.min(regions_with_obs[0]) - 2), min(self.grid_rows, np.max(regions_with_obs[0]) + 3)
            min_x, max_x = max(0, np.min(regions_with_obs[1]) - 2), min(self.grid_cols, np.max(regions_with_obs[1]) + 3)
            
            # Only filter the active region
            active_region = self.log_odds_grid[min_y:max_y, min_x:max_x]
            filtered_region = median_filter(active_region, size=3)
            
            # Create a copy of log_odds to apply filtered values
            filtered_log_odds = np.copy(self.log_odds_grid)
            filtered_log_odds[min_y:max_y, min_x:max_x] = filtered_region
        else:
            filtered_log_odds = self.log_odds_grid
        
        # Flatten for vectorized operations
        valid_indices = np.flatnonzero(valid_mask)
        valid_log_odds = filtered_log_odds.flat[valid_indices]
        
        # Vector classifications
        occupied_mask = valid_log_odds > self.OCCUPIED_THRESHOLD
        free_mask = valid_log_odds < self.FREE_THRESHOLD
        uncertain_mask = ~(occupied_mask | free_mask)
        
        # Mark occupied cells (100)
        grid_data[valid_indices[occupied_mask]] = 100
        
        # Mark free cells (0)
        grid_data[valid_indices[free_mask]] = 0
        
        # Process uncertain cells (convert log-odds to probabilities)
        if np.any(uncertain_mask):
            uncertain_indices = valid_indices[uncertain_mask]
            uncertain_log_odds = valid_log_odds[uncertain_mask]
            
            # Vectorized calculation of probabilities
            probs = 1.0 - (1.0 / (1.0 + np.exp(uncertain_log_odds)))
            grid_data[uncertain_indices] = (probs * 100).astype(np.int8)
        
        return grid_data.tolist()  # Convert to list for ROS message
    
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
        camera_moved = True  # MODIFIED: Always consider the camera as moving
        
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
        ENHANCED: Immediately updates and publishes grid on camera movement
        """
        from std_msgs.msg import String
        
        # Extract position from the pose message
        pos = pose_msg.pose.position
        
        # Calculate position change if we have a previous position
        position_change = 0.0
        if self.last_camera_position is not None:
            dx = pos.x - self.last_camera_position[0]
            dy = pos.y - self.last_camera_position[1]
            dz = pos.z - self.last_camera_position[2]
            position_change = math.sqrt(dx*dx + dy*dy + dz*dz)
            
        # Log the position to both ROS log and a debug topic
        position_str = f"POSE_MONITOR: Camera at ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})"
        self.get_logger().info(position_str)
        debug_msg = String()
        debug_msg.data = position_str
        self.debug_pub.publish(debug_msg)
        
        # Update camera state and mark as moving
        self.camera_motion_detected = True
        self.get_logger().info(f"POSE_CALLBACK: Movement detected: {position_change:.6f}m - IMMEDIATE UPDATE")
        
        # Immediately attempt to update the occupancy grid based on this new position
        try:
            # Get transform from map to camera
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)  # Short timeout for real-time response
            )
            
            # IMMEDIATE UPDATE: Process existing depth data with new transform
            # This allows grid updates even without new depth frames
            with self.grid_lock:
                # For significant movements (> 1cm), force full update of the grid
                if position_change > 0.01:
                    # Get most recent depth image if we have it
                    if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                        self.get_logger().warn("IMMEDIATE grid update triggered by camera movement")
                        self.update_grid_from_depth(self.latest_depth_image, transform)
                        self.publish_occupancy_grid()
            
            # Save map on significant movement
            if position_change > 0.1:  # Save map on larger movements
                self.save_map()
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f"Cannot update grid immediately: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error in immediate grid update: {str(e)}")
        
        # Update last position
        self.last_camera_position = (pos.x, pos.y, pos.z)
        
    def update_grid_from_depth(self, depth_image, transform):
        """
        Update the log-odds grid using the depth image and camera transform.
        Uses probabilistic updates for better map quality.
        JETSON ORIN GPU-accelerated when CUDA is available.
        """
        # DEBUG: Add logging to track function entry
        self.get_logger().info("Entering update_grid_from_depth")
        
        # MODIFIED: Always consider camera moving to ensure updates
        self.camera_motion_detected = True
        
        # Get motion information for debugging purposes
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
        camera_pos = np.array([camera_x, camera_y, camera_z], dtype=np.float32)
        
        # Camera parameters - ZED2i camera
        fov_horizontal = 110.0 * np.pi / 180.0  # radians
        fov_vertical = 70.0 * np.pi / 180.0     # radians
        
        # Extract quaternion from transform for rotation
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Convert quaternion to rotation matrix
        # Formula from: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        rotation_matrix = np.zeros((3, 3), dtype=np.float32)
        
        # First row
        rotation_matrix[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        rotation_matrix[0, 1] = 2.0 * (qx * qy - qz * qw)
        rotation_matrix[0, 2] = 2.0 * (qx * qz + qy * qw)
        
        # Second row
        rotation_matrix[1, 0] = 2.0 * (qx * qy + qz * qw)
        rotation_matrix[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
        rotation_matrix[1, 2] = 2.0 * (qy * qz - qx * qw)
        
        # Third row
        rotation_matrix[2, 0] = 2.0 * (qx * qz - qy * qw)
        rotation_matrix[2, 1] = 2.0 * (qy * qz + qx * qw)
        rotation_matrix[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
        
        # EXTREME PERFORMANCE OPTIMIZATION for real-time updates
        # Use an incredibly aggressive fixed step size for maximum performance
        step = 32  # MODIFIED: Process only 1/32 of pixels - ultra-focused on speed over detail
        
        # Precompute camera position in grid coordinates
        camera_grid_x = int((camera_x - self.grid_origin_x) / self.resolution)
        camera_grid_y = int((camera_y - self.grid_origin_y) / self.resolution)
        
        # Precompute constants to avoid repeated calculations
        tan_fov_h = np.tan(fov_horizontal / 2.0)
        tan_fov_v = np.tan(fov_vertical / 2.0)
        
        # Use a simpler direct approach to ensure reliability
        with self.grid_lock:  # Thread safety
            # Track ray count for debugging
            ray_count = 0
            
            # CUDA Acceleration path using our optimized accelerator
            if CUDA_AVAILABLE and hasattr(self, 'cuda_accelerator') and hasattr(self, 'use_cuda') and self.use_cuda:
                try:
                    self.get_logger().info("Using optimized CUDA acceleration module for grid update")
                    
                    # Create transform data dictionary for the CUDA accelerator
                    transform_data = {
                        'rotation_matrix': rotation_matrix,
                        'camera_pos': camera_pos,
                        'camera_grid_x': camera_grid_x,
                        'camera_grid_y': camera_grid_y
                    }
                    
                    # Create grid data dictionary for the CUDA accelerator
                    grid_data = {
                        'log_odds_grid': self.log_odds_grid,
                        'cell_height_grid': self.cell_height_grid,
                        'observation_count_grid': self.observation_count_grid
                    }
                    
                    # Set up parameters for the CUDA accelerator
                    params = {
                        'step': self.cuda_step,  # Use CUDA-specific step size parameter
                        'min_depth': self.min_depth,
                        'max_depth': self.max_depth,
                        'log_odds_free': self.LOG_ODDS_FREE,
                        'log_odds_occupied': self.LOG_ODDS_OCCUPIED,
                        'log_odds_min': self.LOG_ODDS_MIN,
                        'log_odds_max': self.LOG_ODDS_MAX,
                        'grid_origin_x': self.grid_origin_x,
                        'grid_origin_y': self.grid_origin_y,
                        'resolution': self.resolution,
                        'max_ray_length': self.max_ray_length,
                        'current_alpha': self.current_alpha,
                        'temporal_filtering': self.temporal_filtering,
                        'tan_fov_h': tan_fov_h,
                        'tan_fov_v': tan_fov_v,
                        'ray_step': self.cuda_ray_step  # Use CUDA-specific ray step size parameter
                    }
                    
                    # Call the CUDA accelerator to update the grid
                    result = self.cuda_accelerator.update_grid(depth_image, transform_data, grid_data, params)
                    
                    if result:
                        # Update the grid with the results
                        self.log_odds_grid = result['log_odds_grid']
                        self.cell_height_grid = result['cell_height_grid']
                        self.observation_count_grid = result['observation_count_grid']
                        
                        # Log statistics
                        ray_count = result['stats']['valid_points']
                        total_time = result['stats']['total_time']
                        
                        self.get_logger().info(f"CUDA GPU accelerated: Processed {ray_count} rays in {total_time:.4f} seconds")
                    else:
                        self.get_logger().error("CUDA acceleration failed, falling back to CPU")
                        ray_count = self._update_grid_cpu(
                            depth_image, camera_grid_x, camera_grid_y, 
                            height, width, step, rotation_matrix,
                            camera_x, camera_y, camera_z, tan_fov_h, tan_fov_v
                        )
                except Exception as e:
                    self.get_logger().error(f"Error in CUDA acceleration: {e}")
                    self.get_logger().warn("Falling back to CPU implementation")
                    
                    # Fall back to CPU implementation
                    ray_count = self._update_grid_cpu(
                        depth_image, camera_grid_x, camera_grid_y, 
                        height, width, step, rotation_matrix,
                        camera_x, camera_y, camera_z, tan_fov_h, tan_fov_v
                    )
            else:
                # CPU implementation - use the standard algorithm
                ray_count = self._update_grid_cpu(
                    depth_image, camera_grid_x, camera_grid_y, 
                    height, width, step, rotation_matrix,
                    camera_x, camera_y, camera_z, tan_fov_h, tan_fov_v
                )
            
            # DEBUG: Log how many rays were processed
            if ray_count > 0:
                self.get_logger().info(f"Processed {ray_count} rays in update_grid_from_depth")
            else:
                self.get_logger().warn("No valid rays processed! Check depth data and transforms.")
                
    def _update_grid_cpu(self, depth_image, camera_grid_x, camera_grid_y, 
                          height, width, step, rotation_matrix,
                          camera_x, camera_y, camera_z, tan_fov_h, tan_fov_v):
        """CPU implementation of the grid update (used as fallback when CUDA fails)"""
        ray_count = 0
        
        # Process image in a simple loop approach
        for v in range(0, height, step):
            for u in range(0, width, step):
                # Get depth value
                depth = depth_image[v, u]
                
                # Skip invalid depth values
                if not np.isfinite(depth) or depth < self.min_depth or depth > self.max_depth:
                    continue
                
                # Calculate normalized image coordinates
                normalized_u = (2.0 * u / width - 1.0)
                normalized_v = (2.0 * v / height - 1.0)
                
                # Calculate 3D vector from camera using field of view
                ray_x = normalized_u * tan_fov_h * depth
                ray_y = normalized_v * tan_fov_v * depth
                ray_z = depth
                
                # Transform ray from camera to world coordinates
                world_x = (rotation_matrix[0, 0] * ray_x +
                          rotation_matrix[0, 1] * ray_y +
                          rotation_matrix[0, 2] * ray_z) + camera_x
                
                world_y = (rotation_matrix[1, 0] * ray_x +
                          rotation_matrix[1, 1] * ray_y +
                          rotation_matrix[1, 2] * ray_z) + camera_y
                
                world_z = (rotation_matrix[2, 0] * ray_x +
                          rotation_matrix[2, 1] * ray_y +
                          rotation_matrix[2, 2] * ray_z) + camera_z
                
                # Skip points that are too far from camera
                if np.sqrt((world_x - camera_x)**2 + (world_y - camera_y)**2) > self.max_ray_length:
                    continue
                
                # Convert world coordinates to grid cell coordinates
                grid_x = int((world_x - self.grid_origin_x) / self.resolution)
                grid_y = int((world_y - self.grid_origin_y) / self.resolution)
                
                # Skip if out of grid bounds
                if (grid_x < 0 or grid_x >= self.grid_cols or
                    grid_y < 0 or grid_y >= self.grid_rows):
                    continue
                
                # Bresenham ray-tracing from camera to point
                # This marks cells along the ray as free, and the endpoint as occupied
                self.raytrace_bresenham(
                    camera_grid_x, camera_grid_y,
                    grid_x, grid_y, world_z)
                
                ray_count += 1
                
        return ray_count
    
    def raytrace_bresenham(self, x0, y0, x1, y1, point_height):
        """
        Bresenham's line algorithm for ray tracing through the grid
        Marks cells as free along the ray, and the endpoint as occupied
        """
        # Ensure coordinates are within grid bounds
        x0 = max(0, min(x0, self.grid_cols - 1))
        y0 = max(0, min(y0, self.grid_rows - 1))
        x1 = max(0, min(x1, self.grid_cols - 1))
        y1 = max(0, min(y1, self.grid_rows - 1))
        
        # Calculate differences and steps
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        # Variables to track progress along the ray
        traveled = 0
        total_distance = np.sqrt(dx**2 + dy**2)
        
        # EXTREME OPTIMIZATION: Ultra-fast ray tracing 
        # Skip many points along the ray for maximum performance
        ray_sampling = 8  # MODIFIED: Process only every 8th point along the ray for extreme performance
        count = 0
        
        # Ray tracing
        x, y = x0, y0
        while x != x1 or y != y1:
            # Skip updating the cell where the camera is located
            if (x != x0 or y != y0) and (count % ray_sampling == 0):
                # Update log-odds for free cell (cells along the ray)
                # We use temporal filtering to slowly update log-odds
                if self.temporal_filtering:
                    alpha = self.current_alpha
                    self.log_odds_grid[y, x] = (1 - alpha) * self.log_odds_grid[y, x] + alpha * self.LOG_ODDS_FREE
                else:
                    self.log_odds_grid[y, x] += self.LOG_ODDS_FREE
                
                # Ensure log-odds value stays within bounds
                self.log_odds_grid[y, x] = max(self.LOG_ODDS_MIN, min(self.LOG_ODDS_MAX, self.log_odds_grid[y, x]))
                
                # Increment observation count
                self.observation_count_grid[y, x] += 1
            
            # Bresenham algorithm step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
            # Break if we've gone too far
            traveled += 1
            count += 1
            if traveled > total_distance:
                break
        
        # Mark the endpoint as occupied if it's valid and not the camera position
        if (x1 != x0 or y1 != y0) and x1 >= 0 and x1 < self.grid_cols and y1 >= 0 and y1 < self.grid_rows:
            # Update log-odds for occupied cell (endpoint)
            if self.temporal_filtering:
                alpha = self.current_alpha
                self.log_odds_grid[y1, x1] = (1 - alpha) * self.log_odds_grid[y1, x1] + alpha * self.LOG_ODDS_OCCUPIED
            else:
                self.log_odds_grid[y1, x1] += self.LOG_ODDS_OCCUPIED
            
            # Ensure log-odds value stays within bounds
            self.log_odds_grid[y1, x1] = max(self.LOG_ODDS_MIN, min(self.LOG_ODDS_MAX, self.log_odds_grid[y1, x1]))
            
            # Update height value for the cell (for visualization)
            if self.cell_height_grid[y1, x1] == 0:
                self.cell_height_grid[y1, x1] = point_height
            else:
                # Average with previous height
                self.cell_height_grid[y1, x1] = 0.7 * self.cell_height_grid[y1, x1] + 0.3 * point_height
            
            # Increment observation count
            self.observation_count_grid[y1, x1] += 1
    
    def check_tf_callback(self):
        """
        Check transforms periodically to detect camera movement
        MODIFIED: Far more aggressive detection, forced processing
        """
        try:
            # Try ALL possible transform lookups for maximum robustness
            
            # Attempt 1: Direct lookup (map to camera)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                
                # Log PROMINENTLY for debugging
                camera_pos = transform.transform.translation
                self.get_logger().warn(f"TF DIRECT: Map->Camera position: ({camera_pos.x:.4f}, {camera_pos.y:.4f}, {camera_pos.z:.4f})")
                
                # MANUAL/FORCED MOVEMENT DETECTION
                # Even if no movement is reported, treat minor differences as movement
                if self.last_camera_position is not None:
                    dx = camera_pos.x - self.last_camera_position[0]
                    dy = camera_pos.y - self.last_camera_position[1]
                    dz = camera_pos.z - self.last_camera_position[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # ULTRA SENSITIVE detection - treat ANY non-zero change as movement
                    if abs(dx) > 0.00001 or abs(dy) > 0.00001 or abs(dz) > 0.00001:
                        self.camera_motion_detected = True
                        self.get_logger().error(f"CRITICAL: DETECTED MINISCULE MOVEMENT: {dist:.9f}m ({dx:.9f}, {dy:.9f}, {dz:.9f})")
                        
                        # Force grid update immediately if we have depth data
                        if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                            self.get_logger().error("TRIGGERING FORCED UPDATE FROM TF CALLBACK")
                            self.update_grid_from_depth(self.latest_depth_image, transform)
                            self.publish_occupancy_grid()
                
                self.last_camera_position = (camera_pos.x, camera_pos.y, camera_pos.z)
                return  # Success, no need to try other methods
            except Exception as e1:
                self.get_logger().info(f"Direct transform lookup attempt failed: {str(e1)}")
                # Fall through to next attempt
                
            # Attempt 2: Via odom frame
            try:
                # Get odom->camera transform
                transform = self.tf_buffer.lookup_transform(
                    'odom',
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.05)
                )
                
                # Log position for debugging
                camera_pos = transform.transform.translation
                self.get_logger().warn(f"TF ODOM: Odom->Camera position: ({camera_pos.x:.4f}, {camera_pos.y:.4f}, {camera_pos.z:.4f})")
                
                # Always treat as movement when using odom frame
                self.camera_motion_detected = True
                
                # Treat ANY detection as significant
                if self.last_camera_position is None:
                    self.last_camera_position = (camera_pos.x, camera_pos.y, camera_pos.z)
                else:
                    # Calculate "delta" from last position
                    dx = camera_pos.x - self.last_camera_position[0] 
                    dy = camera_pos.y - self.last_camera_position[1]
                    dz = camera_pos.z - self.last_camera_position[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Report any change
                    if dist > 0.00001:
                        self.get_logger().error(f"CRITICAL: ODOM MOVEMENT: {dist:.9f}m ({dx:.9f}, {dy:.9f}, {dz:.9f})")
                    
                    # Update last position 
                    self.last_camera_position = (camera_pos.x, camera_pos.y, camera_pos.z)
                
                # Always force a grid update with odom frame
                if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                    self.get_logger().warn("TRIGGERING FORCED UPDATE FROM ODOM TF LOOKUP")
                    self.update_grid_from_depth(self.latest_depth_image, transform)
                    self.publish_occupancy_grid(override_frame='odom')
                
                return  # Success with odom frame
            except Exception as e2:
                self.get_logger().info(f"Odom transform lookup attempt failed: {str(e2)}")
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF Monitor: Error getting any transform: {str(e)}')
    
    def force_update_callback(self):
        """Force grid updates periodically even without new depth data"""
        # Only force updates if we've already built a map
        if np.any(self.observation_count_grid > 0):
            self.get_logger().debug("Forcing grid update")
            
            # Try to get current camera position to check for movement
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                
                # If we got a valid transform, check if camera has moved
                camera_pos = transform.transform.translation
                if self.last_camera_position is not None:
                    dx = camera_pos.x - self.last_camera_position[0]
                    dy = camera_pos.y - self.last_camera_position[1]
                    dz = camera_pos.z - self.last_camera_position[2]
                    position_change = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Log position change for debugging
                    if position_change > 0.001:  # Very small threshold to detect minor movements
                        self.get_logger().info(f"FORCED UPDATE: Camera moved by {position_change:.6f}m")
                        self.camera_motion_detected = True
                        
                        # Update last known position
                        self.last_camera_position = (camera_pos.x, camera_pos.y, camera_pos.z)
                
                # Publish the grid regardless of movement
                self.publish_occupancy_grid()
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # If transform lookup fails, still try to publish the grid
                self.get_logger().debug(f"Force update TF error: {str(e)}")
                self.publish_occupancy_grid()
            except Exception as e:
                self.get_logger().warning(f"Unexpected error in force update: {str(e)}")
                # Still try to publish
                self.publish_occupancy_grid()
    
    def odom_callback(self, odom_msg):
        """Process odometry messages to detect camera movement"""
        # Extract position from odometry message
        pos = odom_msg.pose.pose.position
        
        # Log the odometry position
        self.get_logger().error(f"!!! ODOM DATA RECEIVED !!! Position: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
        
        # Always force an update with odometry data
        try:
            # Mark as moving - critical for grid updates
            self.camera_motion_detected = True
            
            # Force process depth data if available
            if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                # Try to get map->camera transform
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Force a grid update with latest transform
                    self.get_logger().error("*** IMMEDIATE UPDATE FROM ODOM CALLBACK ***")
                    self.update_grid_from_depth(self.latest_depth_image, transform)
                    self.publish_occupancy_grid()
                except Exception as e:
                    self.get_logger().error(f"Error getting transform in odom callback: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error in odom callback: {str(e)}")
    
    def imu_callback(self, imu_msg):
        """Process IMU data to detect camera movement from acceleration"""
        # Extract acceleration data
        accel = imu_msg.linear_acceleration
        
        # Skip if acceleration is too small (noise)
        accel_magnitude = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        if accel_magnitude > 0.05:  # Any non-zero acceleration is likely movement
            self.get_logger().error(f"!!! IMU MOVEMENT DETECTED !!! Acceleration: {accel_magnitude:.4f} m/s²")
            
            # Mark as moving
            self.camera_motion_detected = True
            
            # Try to force update grid with latest transform and depth image
            try:
                if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Force update
                    self.get_logger().error("*** IMMEDIATE UPDATE FROM IMU CALLBACK ***")
                    self.update_grid_from_depth(self.latest_depth_image, transform)
                    self.publish_occupancy_grid()
            except Exception as e:
                self.get_logger().error(f"Error updating grid from IMU: {str(e)}")
    
    def twist_callback(self, twist_msg):
        """Process velocity commands to preemptively detect camera movement"""
        # Extract linear and angular velocities
        linear = twist_msg.linear
        angular = twist_msg.angular
        
        # Check if there's significant movement commanded
        linear_mag = math.sqrt(linear.x**2 + linear.y**2 + linear.z**2)
        angular_mag = math.sqrt(angular.x**2 + angular.y**2 + angular.z**2)
        
        if linear_mag > 0.01 or angular_mag > 0.01:  # Very small threshold
            self.get_logger().error(f"!!! VELOCITY COMMAND DETECTED !!! Linear: {linear_mag:.4f}, Angular: {angular_mag:.4f}")
            
            # Mark as moving
            self.camera_motion_detected = True
            
            # Try to trigger grid update
            try:
                if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Force update
                    self.update_grid_from_depth(self.latest_depth_image, transform)
                    self.publish_occupancy_grid()
            except Exception as e:
                self.get_logger().error(f"Error updating grid from velocity: {str(e)}")
    
    def forced_movement_callback(self):
        """
        CRITICAL: Timer-based forced update regardless of camera movement
        This ensures continuous grid updates even when camera movement detection fails
        """
        try:
            # Force camera-moving flag to true - this is CRITICAL
            self.camera_motion_detected = True
            self.get_logger().error("*** FORCED MOVEMENT FLAG SET - EMERGENCY BYPASS ***")
            
            # Get latest transform and force grid update
            if hasattr(self, 'latest_depth_image') and self.latest_depth_image is not None:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        self.camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Log transform for debugging
                    camera_pos = transform.transform.translation
                    self.get_logger().error(f"*** FORCED UPDATE WITH POSITION: ({camera_pos.x:.4f}, {camera_pos.y:.4f}, {camera_pos.z:.4f}) ***")
                    
                    # Force grid update and publish
                    self.update_grid_from_depth(self.latest_depth_image, transform)
                    self.publish_occupancy_grid()
                except Exception as e:
                    self.get_logger().error(f"Error in forced update: {str(e)}")
                    
                    # Even if transform lookup fails, try with odom frame as fallback
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            'odom',
                            self.camera_frame,
                            rclpy.time.Time(),
                            rclpy.duration.Duration(seconds=0.1)
                        )
                        
                        # Force update with odom frame
                        self.update_grid_from_depth(self.latest_depth_image, transform)
                        self.publish_occupancy_grid(override_frame='odom')
                    except Exception as e2:
                        self.get_logger().error(f"Fallback forced update also failed: {str(e2)}")
        except Exception as e:
            self.get_logger().error(f"Error in forced movement callback: {str(e)}")
            
    def camera_monitor_callback(self):
        """Log camera position and status periodically"""
        from std_msgs.msg import String
        
        # Check if we have position info
        if self.last_camera_position is not None:
            status = "MOVING" if self.camera_motion_detected else "STATIC"
            position_str = (f"Camera {status}: ({self.last_camera_position[0]:.3f}, "
                           f"{self.last_camera_position[1]:.3f}, {self.last_camera_position[2]:.3f})")
            
            # Publish debug info
            debug_msg = String()
            debug_msg.data = position_str
            self.debug_pub.publish(debug_msg)


def main(args=None):
    """Main entry point for the ZED Occupancy Grid Node"""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create the node
    node = ZedOccupancyGridNode()
    
    # Set up multi-threaded executor for better performance with multiple callbacks
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # Spin the node
        node.get_logger().info('ZED Occupancy Grid Node is spinning...')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        # Clean shutdown - save map before exit
        try:
            node.get_logger().info('Saving final map before shutdown...')
            node.save_map()
        except Exception as e:
            node.get_logger().error(f'Error saving final map: {str(e)}')
            
        # Destroy node and shut down ROS
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
