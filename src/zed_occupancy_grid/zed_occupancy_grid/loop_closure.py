#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
import threading
import math
import time
import cv2
import tf2_ros
from geometry_msgs.msg import PoseStamped, Transform, TransformStamped, Pose
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Float32, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from scipy.spatial import KDTree
from scipy import optimize

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'

class LoopClosureDetector:
    """Class handling loop closure detection and correction for occupancy grid maps"""
    
    def __init__(self, logger, tf_buffer):
        """Initialize the loop closure detector"""
        self.logger = logger
        self.tf_buffer = tf_buffer
        
        # Parameters for loop closure detection
        self.distance_threshold = 0.5  # Minimum distance for considering a potential loop closure
        self.min_revisit_distance = 0.2  # Minimum distance to track as a distinct position
        self.min_loop_length = 20  # Minimum number of poses to form a loop
        self.scan_consistency_threshold = 0.75  # Required similarity for positive match
        self.min_time_between_loops = 10.0  # Minimum time (seconds) between loop closure events
        
        # Camera trajectory history
        self.trajectory_poses = []  # List of (pose, timestamp) tuples
        self.trajectory_depth_fingerprints = []  # Depth image fingerprints at key poses
        self.trajectory_grid_patches = []  # Grid patches at key poses
        self.trajectory_kdtree = None  # KDTree for fast pose lookups
        
        # Loop closure status
        self.last_loop_closure_time = 0.0
        self.loop_closures_detected = 0
        self.loop_closures_accepted = 0
        self.last_loop_correction = None  # Last loop correction transform
        
        # Grid update information
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_resolution = 0.05
        self.log_odds_grid = None
        self.observation_count_grid = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info(f"{GREEN}{BOLD}Loop Closure Module Initialized{END}")
        
    def set_grid_info(self, log_odds_grid, observation_count_grid, origin_x, origin_y, resolution):
        """Set grid information for loop closure operations"""
        with self.lock:
            self.log_odds_grid = log_odds_grid
            self.observation_count_grid = observation_count_grid
            self.map_origin_x = origin_x
            self.map_origin_y = origin_y
            self.map_resolution = resolution
            
            # Rebuild KDTree if trajectory exists
            if len(self.trajectory_poses) > 10:
                positions = np.array([(p[0].position.x, p[0].position.y) for p in self.trajectory_poses])
                self.trajectory_kdtree = KDTree(positions)
                
    def update_camera_position(self, camera_pose, depth_image=None, grid_data=None, timestamp=None):
        """
        Update the camera trajectory with a new pose
        Optionally provide depth image and grid data for feature storage
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            # Skip if we have no previous positions
            if not self.trajectory_poses:
                # Store the first pose
                self.trajectory_poses.append((camera_pose, timestamp))
                
                # Store depth fingerprint if available
                if depth_image is not None:
                    self.trajectory_depth_fingerprints.append(self._create_depth_fingerprint(depth_image))
                else:
                    self.trajectory_depth_fingerprints.append(None)
                    
                # Store grid patch if available
                if grid_data is not None and self.log_odds_grid is not None:
                    self.trajectory_grid_patches.append(self._extract_grid_patch(camera_pose))
                else:
                    self.trajectory_grid_patches.append(None)
                    
                self.logger.debug("Added first camera pose to trajectory")
                return False
            
            # Check if we've moved significantly from the last position
            last_pose = self.trajectory_poses[-1][0]
            dist = self._pose_distance(camera_pose, last_pose)
            
            if dist < self.min_revisit_distance:
                # Not significant movement, skip
                return False
                
            # Add this position to our trajectory
            self.trajectory_poses.append((camera_pose, timestamp))
            
            # Store depth fingerprint if available
            if depth_image is not None:
                self.trajectory_depth_fingerprints.append(self._create_depth_fingerprint(depth_image))
            else:
                self.trajectory_depth_fingerprints.append(None)
                
            # Store grid patch if available
            if grid_data is not None and self.log_odds_grid is not None:
                self.trajectory_grid_patches.append(self._extract_grid_patch(camera_pose))
            else:
                self.trajectory_grid_patches.append(None)
            
            # Rebuild the KDTree with the new position
            if len(self.trajectory_poses) > 10:
                positions = np.array([(p[0].position.x, p[0].position.y) for p in self.trajectory_poses])
                self.trajectory_kdtree = KDTree(positions)
                
            # Only check for loop closure after accumulating enough poses
            if len(self.trajectory_poses) < self.min_loop_length:
                return False
                
            # Check for potential loop closures if we've moved enough
            return self._check_for_loop_closure(camera_pose, depth_image, grid_data, timestamp)
            
    def _check_for_loop_closure(self, current_pose, depth_image, grid_data, timestamp):
        """
        Check if current pose creates a loop closure with any previous poses
        Returns: (bool) True if loop closure detected and accepted
        """
        # Skip if not enough time has passed since last loop closure
        if timestamp - self.last_loop_closure_time < self.min_time_between_loops:
            return False
            
        # Skip if we don't have a KDTree yet
        if self.trajectory_kdtree is None:
            return False
            
        # Only proceed if we have enough poses
        if len(self.trajectory_poses) < self.min_loop_length:
            return False
            
        # Query KDTree for nearby poses that are not recent
        current_pos = np.array([[current_pose.position.x, current_pose.position.y]])
        
        # Get distances and indices of potential loop closures
        distances, indices = self.trajectory_kdtree.query(current_pos, k=10)
        distances = distances[0]
        indices = indices[0]
        
        # Filter by distance threshold and loop length
        close_indices = []
        for i, idx in enumerate(indices):
            if distances[i] <= self.distance_threshold:
                # Only consider poses that are not recent (forming a loop)
                if len(self.trajectory_poses) - idx > self.min_loop_length:
                    close_indices.append(idx)
        
        if not close_indices:
            return False
            
        self.logger.info(f"{YELLOW}Found {len(close_indices)} potential loop closure candidates{END}")
        
        # Create fingerprint of current depth image
        current_fingerprint = None
        if depth_image is not None:
            current_fingerprint = self._create_depth_fingerprint(depth_image)
            
        # Create patch of current grid section
        current_grid_patch = None
        if grid_data is not None and self.log_odds_grid is not None:
            current_grid_patch = self._extract_grid_patch(current_pose)
            
        # Evaluate each candidate
        best_match_score = 0
        best_match_index = -1
        
        for idx in close_indices:
            # Skip if we don't have fingerprints stored
            if self.trajectory_depth_fingerprints[idx] is None or current_fingerprint is None:
                continue
                
            # Compare depth fingerprints
            fp_similarity = self._compare_fingerprints(
                current_fingerprint, 
                self.trajectory_depth_fingerprints[idx]
            )
            
            # Compare grid patches
            grid_similarity = 0
            if self.trajectory_grid_patches[idx] is not None and current_grid_patch is not None:
                grid_similarity = self._compare_grid_patches(
                    current_grid_patch,
                    self.trajectory_grid_patches[idx]
                )
                
            # Combined similarity score (weighted)
            combined_score = 0.6 * fp_similarity + 0.4 * grid_similarity
            
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_index = idx
        
        # If we have a good match, process the loop closure
        if best_match_score > self.scan_consistency_threshold and best_match_index >= 0:
            self.loop_closures_detected += 1
            self.logger.info(f"{GREEN}{BOLD}Loop closure detected with score {best_match_score:.4f}{END}")
            
            # Get the matched historical pose
            matched_pose = self.trajectory_poses[best_match_index][0]
            
            # Calculate the correction transform
            correction = self._calculate_loop_closure_correction(current_pose, matched_pose)
            if correction is not None:
                # Apply the correction
                self.last_loop_correction = correction
                self.last_loop_closure_time = timestamp
                self.loop_closures_accepted += 1
                
                self.logger.info(f"{GREEN}{BOLD}Loop closure #{self.loop_closures_accepted} applied!{END}")
                self.logger.info(f"{GREEN}Correction: dx={correction[0]:.4f}, dy={correction[1]:.4f}, dtheta={correction[2]:.4f}{END}")
                
                return True
        
        return False
        
    def _create_depth_fingerprint(self, depth_image):
        """Create a fingerprint from a depth image for loop closure detection"""
        try:
            # Fast depth image fingerprinting - resize to small size
            if depth_image is None:
                return None
                
            # Create a valid depth mask
            valid_mask = np.isfinite(depth_image)
            
            # Skip if not enough valid depth points
            if np.sum(valid_mask) < 100:
                return None
                
            # Normalize and resize the depth image for fingerprinting
            normalized = np.zeros_like(depth_image)
            normalized[valid_mask] = depth_image[valid_mask]
            normalized[~valid_mask] = 0
            
            # Resize to smaller representation for faster comparison
            small_depth = cv2.resize(normalized, (32, 24))
            
            # Apply a Gaussian blur to smooth the fingerprint
            small_depth = cv2.GaussianBlur(small_depth, (3, 3), 0)
            
            # Convert to bytes for storage efficiency
            return small_depth
            
        except Exception as e:
            self.logger.error(f"Error creating depth fingerprint: {e}")
            return None
            
    def _compare_fingerprints(self, fp1, fp2):
        """Compare two depth fingerprints and return similarity score [0-1]"""
        try:
            if fp1 is None or fp2 is None:
                return 0.0
                
            # Normalize fingerprints if not already normalized
            fp1_norm = fp1 / (np.max(fp1) + 1e-10)
            fp2_norm = fp2 / (np.max(fp2) + 1e-10)
            
            # Calculate correlation
            correlation = np.corrcoef(fp1_norm.flatten(), fp2_norm.flatten())[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                return 0.0
                
            # Convert to a similarity score [0-1]
            similarity = (correlation + 1) / 2.0
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error comparing fingerprints: {e}")
            return 0.0
            
    def _extract_grid_patch(self, pose):
        """Extract a small patch of the occupancy grid around the pose"""
        try:
            if self.log_odds_grid is None:
                return None
                
            # Convert pose to grid coordinates
            grid_x = int((pose.position.x - self.map_origin_x) / self.map_resolution)
            grid_y = int((pose.position.y - self.map_origin_y) / self.map_resolution)
            
            # Define patch size
            patch_size = 20  # 20x20 cells centered on position
            half_size = patch_size // 2
            
            # Check bounds
            rows, cols = self.log_odds_grid.shape
            if (grid_x - half_size < 0 or grid_x + half_size >= cols or
                grid_y - half_size < 0 or grid_y + half_size >= rows):
                return None
                
            # Extract patch
            patch = self.log_odds_grid[
                grid_y - half_size:grid_y + half_size,
                grid_x - half_size:grid_x + half_size
            ].copy()
            
            return patch
            
        except Exception as e:
            self.logger.error(f"Error extracting grid patch: {e}")
            return None
            
    def _compare_grid_patches(self, patch1, patch2):
        """Compare two grid patches and return similarity score [0-1]"""
        try:
            if patch1 is None or patch2 is None:
                return 0.0
                
            # Ensure they're the same size
            if patch1.shape != patch2.shape:
                return 0.0
                
            # Create binary occupied/free representation
            # LogOdds > 0 means likely occupied, < 0 means likely free
            binary1 = patch1 > 0
            binary2 = patch2 > 0
            
            # Calculate Intersection over Union
            intersection = np.logical_and(binary1, binary2)
            union = np.logical_or(binary1, binary2)
            
            # Avoid division by zero
            if np.sum(union) == 0:
                return 0.0
                
            iou = np.sum(intersection) / np.sum(union)
            
            return iou
            
        except Exception as e:
            self.logger.error(f"Error comparing grid patches: {e}")
            return 0.0
            
    def _pose_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)
        
    def _calculate_loop_closure_correction(self, current_pose, matched_pose):
        """
        Calculate correction transform for loop closure
        Returns (dx, dy, dtheta) correction
        """
        try:
            # Extract positions
            cur_x = current_pose.position.x
            cur_y = current_pose.position.y
            match_x = matched_pose.position.x
            match_y = matched_pose.position.y
            
            # Extract orientations (assuming quaternions)
            # Convert quaternions to Euler angles (yaw only for 2D)
            cur_quat = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ]
            match_quat = [
                matched_pose.orientation.x,
                matched_pose.orientation.y,
                matched_pose.orientation.z,
                matched_pose.orientation.w
            ]
            
            # Calculate yaw angles
            cur_yaw = self._quaternion_to_yaw(cur_quat)
            match_yaw = self._quaternion_to_yaw(match_quat)
            
            # Calculate correction
            dx = match_x - cur_x
            dy = match_y - cur_y
            dtheta = match_yaw - cur_yaw
            
            # Normalize angle to [-pi, pi]
            while dtheta > math.pi:
                dtheta -= 2 * math.pi
            while dtheta < -math.pi:
                dtheta += 2 * math.pi
                
            return (dx, dy, dtheta)
            
        except Exception as e:
            self.logger.error(f"Error calculating loop closure correction: {e}")
            return None
            
    def _quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle (for 2D)"""
        # Extract quaternion components
        x, y, z, w = quat
        
        # Calculate yaw (rotation around z axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw
        
    def apply_loop_closure_correction(self, grid_to_correct=None):
        """
        Apply the latest loop closure correction to the grid
        Returns: corrected grid
        """
        if self.last_loop_correction is None:
            return grid_to_correct
            
        with self.lock:
            dx, dy, dtheta = self.last_loop_correction
            
            # If we're not provided a grid to correct, use the internal one
            if grid_to_correct is None:
                if self.log_odds_grid is None:
                    return None
                grid_to_correct = self.log_odds_grid
                
            rows, cols = grid_to_correct.shape
            
            # Create transformation matrix
            cos_theta = math.cos(dtheta)
            sin_theta = math.sin(dtheta)
            
            # Apply correction with interpolation
            corrected_grid = np.zeros_like(grid_to_correct)
            
            # Calculate grid cell adjustments
            dx_cells = dx / self.map_resolution
            dy_cells = dy / self.map_resolution
            
            # Process the grid cells
            y_indices, x_indices = np.meshgrid(
                np.arange(rows),
                np.arange(cols),
                indexing='ij'
            )
            
            # Calculate source coordinates with rotation and translation
            center_y, center_x = rows/2, cols/2
            y_centered = y_indices - center_y
            x_centered = x_indices - center_x
            
            # Apply rotation (counter-rotation)
            x_rotated = x_centered * cos_theta - y_centered * sin_theta
            y_rotated = x_centered * sin_theta + y_centered * cos_theta
            
            # Apply translation and shift back
            src_x = x_rotated + center_x - dx_cells
            src_y = y_rotated + center_y - dy_cells
            
            # Interpolate using OpenCV's remap
            # Convert to float32 for remap
            map_x = src_x.astype(np.float32)
            map_y = src_y.astype(np.float32)
            
            # Use opencv remap for efficient interpolation
            corrected_grid = cv2.remap(
                grid_to_correct.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )
            
            return corrected_grid
            
    def get_loop_closure_stats(self):
        """Return current loop closure statistics"""
        with self.lock:
            return {
                'detected': self.loop_closures_detected,
                'accepted': self.loop_closures_accepted,
                'last_time': self.last_loop_closure_time,
                'trajectory_length': len(self.trajectory_poses)
            }
            
    def optimize_map(self, log_odds_grid=None, observation_count_grid=None):
        """
        Perform global pose graph optimization and update the map
        This should be called periodically after multiple loop closures
        """
        with self.lock:
            # If we don't have enough loop closures, don't optimize
            if self.loop_closures_accepted < 2:
                return log_odds_grid, observation_count_grid
                
            # Simple implementation: just apply the latest correction
            # A full pose graph optimization would be more complex
            if log_odds_grid is None:
                log_odds_grid = self.log_odds_grid
            
            if observation_count_grid is None:
                observation_count_grid = self.observation_count_grid
                
            # Apply loop closure correction to both grids
            corrected_log_odds = self.apply_loop_closure_correction(log_odds_grid)
            
            # Correct observation count using same transform
            corrected_obs_count = None
            if observation_count_grid is not None:
                corrected_obs_count = self.apply_loop_closure_correction(observation_count_grid)
                
            # Reset the correction after applying
            self.last_loop_correction = None
            
            self.logger.info(f"{GREEN}{BOLD}Map optimized after {self.loop_closures_accepted} loop closures{END}")
            
            return corrected_log_odds, corrected_obs_count


class LoopClosureNode(Node):
    """ROS2 node for loop closure detection and correction"""
    
    def __init__(self):
        super().__init__('loop_closure_node')
        
        # Set up TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create loop closure detector
        self.loop_closure_detector = LoopClosureDetector(self.get_logger(), self.tf_buffer)
        
        # Declare parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'zed_left_camera_frame')
        self.declare_parameter('grid_topic', '/occupancy_grid')
        self.declare_parameter('pose_topic', '/zed/zed_node/pose')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        
        # Get parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.grid_topic = self.get_parameter('grid_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        
        # Create subscribers with reliable QoS
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to occupancy grid
        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            self.grid_topic,
            self.grid_callback,
            qos_profile
        )
        
        # Subscribe to camera pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            qos_profile
        )
        
        # Subscribe to depth image
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        self.cv_bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile
        )
        
        # Publishers
        self.loop_closure_detected_pub = self.create_publisher(
            Bool,
            '/loop_closure/detected',
            10
        )
        
        self.loop_closure_info_pub = self.create_publisher(
            String,
            '/loop_closure/info',
            10
        )
        
        # Create a corrected grid publisher
        self.corrected_grid_pub = self.create_publisher(
            OccupancyGrid,
            '/occupancy_grid_corrected',
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1
            )
        )
        
        # Create timer for periodic tasks
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Variables to store the latest data
        self.latest_depth_image = None
        self.latest_grid_msg = None
        self.latest_pose = None
        self.loop_closure_detected = False
        
        # Cache log-odds grid representation
        self.latest_log_odds_grid = None
        self.latest_observation_count_grid = None
        
        # Last update time
        self.last_update_time = time.time()
        self.last_depth_time = 0.0
        self.last_pose_time = 0.0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # TF broadcaster for publishing corrected transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.get_logger().info(f"{GREEN}{BOLD}Loop Closure Node Initialized{END}")
        
    def grid_callback(self, grid_msg):
        """Process occupancy grid messages"""
        with self.lock:
            # Store the latest grid message
            self.latest_grid_msg = grid_msg
            
            # Convert grid data to log-odds form for the detector
            grid_data = np.array(grid_msg.data).reshape(
                grid_msg.info.height, grid_msg.info.width
            )
            
            # Convert occupancy probabilities [0,100] to log-odds
            # -1 (unknown) stays as is
            log_odds_grid = np.zeros_like(grid_data, dtype=np.float32)
            
            # Convert only known cells
            known_mask = grid_data != -1
            
            # Scale from [0,100] to [0,1]
            probs = grid_data[known_mask].astype(np.float32) / 100.0
            
            # Avoid log(0) and log(1) by clipping
            probs = np.clip(probs, 0.01, 0.99)
            
            # Convert to log-odds: log(p/(1-p))
            log_odds_grid[known_mask] = np.log(probs / (1.0 - probs))
            
            # Generate a simple observation count grid (1 for observed cells)
            observation_count_grid = np.zeros_like(grid_data, dtype=np.int32)
            observation_count_grid[known_mask] = 1
            
            # Store for loop closure detector
            self.latest_log_odds_grid = log_odds_grid
            self.latest_observation_count_grid = observation_count_grid
            
            # Update grid info in detector
            self.loop_closure_detector.set_grid_info(
                log_odds_grid,
                observation_count_grid,
                grid_msg.info.origin.position.x,
                grid_msg.info.origin.position.y,
                grid_msg.info.resolution
            )
            
    def pose_callback(self, pose_msg):
        """Process camera pose messages"""
        with self.lock:
            # Store the latest pose
            self.latest_pose = pose_msg.pose
            self.last_pose_time = time.time()
            
            # If we have all necessary data, process loop closure
            self._process_loop_closure()
            
    def depth_callback(self, depth_msg):
        """Process depth image messages"""
        with self.lock:
            try:
                # Convert depth image to OpenCV format
                depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
                self.latest_depth_image = depth_image
                self.last_depth_time = time.time()
                
                # If we have all necessary data, process loop closure
                self._process_loop_closure()
                
            except Exception as e:
                self.get_logger().error(f"Error processing depth image: {str(e)}")
    
    def _process_loop_closure(self):
        """Process all data for loop closure detection"""
        # Only process if we have all necessary data
        if (self.latest_pose is None or 
            self.latest_depth_image is None or 
            self.latest_log_odds_grid is None):
            return
            
        # Don't process too frequently
        current_time = time.time()
        if current_time - self.last_update_time < 0.5:  # Max 2Hz processing
            return
            
        # Update last process time
        self.last_update_time = current_time
        
        # Update trajectory and check for loop closure
        self.loop_closure_detected = self.loop_closure_detector.update_camera_position(
            self.latest_pose,
            self.latest_depth_image,
            self.latest_log_odds_grid,
            current_time
        )
        
        # If loop closure detected, apply corrections and publish updated grid
        if self.loop_closure_detected:
            self._handle_loop_closure()
            
            # Publish detection event
            detection_msg = Bool()
            detection_msg.data = True
            self.loop_closure_detected_pub.publish(detection_msg)
            
            # Publish stats
            stats = self.loop_closure_detector.get_loop_closure_stats()
            info_msg = String()
            info_msg.data = f"Loop closures: {stats['accepted']}/{stats['detected']}"
            self.loop_closure_info_pub.publish(info_msg)
            
    def _handle_loop_closure(self):
        """Handle a detected loop closure"""
        # Only proceed if we have all needed data
        if self.latest_grid_msg is None:
            self.get_logger().error("Cannot handle loop closure: missing grid data")
            return
            
        # Get the corrected grids
        corrected_log_odds, corrected_obs_count = self.loop_closure_detector.optimize_map(
            self.latest_log_odds_grid,
            self.latest_observation_count_grid
        )
        
        if corrected_log_odds is None:
            self.get_logger().error("Failed to get corrected grid")
            return
            
        # Convert log-odds back to occupancy probability [0,100]
        # p = 1 - 1/(1+exp(l))
        log_odds = corrected_log_odds.flatten()
        probs = 1.0 - (1.0 / (1.0 + np.exp(log_odds)))
        
        # Convert to int8 for occupancy grid message
        grid_data = (probs * 100).astype(np.int8).tolist()
        
        # Create and publish corrected grid message
        corrected_grid_msg = OccupancyGrid()
        corrected_grid_msg.header = self.latest_grid_msg.header
        corrected_grid_msg.header.stamp = self.get_clock().now().to_msg()
        corrected_grid_msg.info = self.latest_grid_msg.info
        corrected_grid_msg.data = grid_data
        
        self.corrected_grid_pub.publish(corrected_grid_msg)
        
        # Publish the loop closure correction as a transform
        self._publish_loop_closure_transform()
        
        self.get_logger().info(f"{GREEN}Published corrected occupancy grid after loop closure{END}")
        
    def _publish_loop_closure_transform(self):
        """Publish a transform representing the loop closure correction"""
        # Only proceed if we have a correction
        if self.loop_closure_detector.last_loop_correction is None:
            return
            
        # Get the correction
        dx, dy, dtheta = self.loop_closure_detector.last_loop_correction
        
        # Create a transform message
        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = self.map_frame
        transform_msg.child_frame_id = 'loop_closure_correction'
        
        # Set translation
        transform_msg.transform.translation.x = dx
        transform_msg.transform.translation.y = dy
        transform_msg.transform.translation.z = 0.0
        
        # Set rotation (convert theta to quaternion)
        from math import sin, cos
        transform_msg.transform.rotation.x = 0.0
        transform_msg.transform.rotation.y = 0.0
        transform_msg.transform.rotation.z = sin(dtheta / 2.0)
        transform_msg.transform.rotation.w = cos(dtheta / 2.0)
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_msg)
        
    def timer_callback(self):
        """Regular timer callback for periodic operations"""
        # Check data freshness
        current_time = time.time()
        pose_age = current_time - self.last_pose_time
        depth_age = current_time - self.last_depth_time
        
        # Log status
        stats = self.loop_closure_detector.get_loop_closure_stats()
        self.get_logger().info(
            f"Loop closure status: {stats['accepted']}/{stats['detected']} closures, "
            f"{stats['trajectory_length']} poses tracked"
        )
        
        # Check if data is stale
        if pose_age > 5.0 and self.last_pose_time > 0:
            self.get_logger().warn(f"Pose data is stale ({pose_age:.1f}s old)")
            
        if depth_age > 5.0 and self.last_depth_time > 0:
            self.get_logger().warn(f"Depth data is stale ({depth_age:.1f}s old)")
            
        # Try to optimize map periodically if we have multiple loop closures
        if stats['accepted'] >= 2:
            self.get_logger().info("Triggering periodic map optimization")
            self._handle_loop_closure()


def main(args=None):
    """Main entry point for the Loop Closure Node"""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create the node
    node = LoopClosureNode()
    
    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
