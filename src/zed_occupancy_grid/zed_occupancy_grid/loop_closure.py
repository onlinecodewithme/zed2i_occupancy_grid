#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Transform, TransformStamped
import tf2_ros
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import threading
import time
import math
from std_msgs.msg import Bool, String
from sklearn.neighbors import KDTree  # For efficient spatial neighbor search
import cv2
from scipy.spatial.transform import Rotation as R

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
BOLD = '\033[1m'
END = '\033[0m'

class DepthFingerprint:
    """A compact representation of a depth image for place recognition"""
    
    def __init__(self, depth_image, position, orientation):
        """
        Create a fingerprint from a depth image
        
        Args:
            depth_image: numpy array of depth values (meters)
            position: (x, y, z) position where image was taken
            orientation: (x, y, z, w) quaternion orientation
        """
        self.position = np.array(position)
        self.orientation = np.array(orientation)  # Quaternion (x, y, z, w)
        
        # Create a compact representation
        self.features = self._compute_features(depth_image)
        
        # Store timestamp for age-based filtering
        self.timestamp = time.time()
        
    def _compute_features(self, depth_image):
        """
        Compute a feature descriptor from the depth image
        
        This creates a histogram-based representation of the depth distribution
        which is robust to small viewpoint changes
        """
        # Skip invalid values
        valid_mask = np.isfinite(depth_image)
        if np.sum(valid_mask) < 100:  # Need enough valid points
            return np.zeros(32)  # Return empty feature
            
        valid_depths = depth_image[valid_mask]
        
        # Compute distance histogram (32 bins up to 10m)
        hist, _ = np.histogram(valid_depths, bins=32, range=(0.1, 10.0), density=True)
        
        # Normalize histogram to make it scale-invariant
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        # Add additional features: depth statistics
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        
        # Compute depth gradients for structure info
        # Subsample for speed
        h, w = depth_image.shape
        subsampled = depth_image[::10, ::10]
        grad_y, grad_x = np.gradient(subsampled)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_features = np.histogram(grad_mag[np.isfinite(grad_mag)], bins=8, range=(0, 2.0))[0]
        if np.sum(grad_features) > 0:
            grad_features = grad_features / np.sum(grad_features)
        
        # Combine all features
        features = np.concatenate([
            hist,
            [mean_depth, std_depth, min_depth, max_depth],
            grad_features
        ])
        
        return features
        
    def similarity(self, other):
        """
        Compute similarity score with another fingerprint
        
        Returns:
            float: Similarity score (0-1)
        """
        # Compute feature similarity
        feature_dist = np.linalg.norm(self.features - other.features)
        feature_sim = np.exp(-feature_dist)
        
        # Weight by inverse distance (closer positions are more likely to be similar)
        pos_dist = np.linalg.norm(self.position - other.position)
        max_dist = 10.0  # Maximum distance to consider for loop closure
        
        # If too far apart, unlikely to be the same place
        if pos_dist > max_dist:
            return 0.0
            
        # Decay score with distance
        dist_factor = np.exp(-pos_dist / 5.0)  
        
        # Combine feature and distance metrics
        return feature_sim * dist_factor
        
    def is_match(self, other, threshold=0.8):
        """Check if this fingerprint matches another one"""
        return self.similarity(other) > threshold


class LoopClosureDetector:
    """Class to detect loop closures in a trajectory"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Collection of fingerprints from previous frames
        self.fingerprints = []
        
        # KD Tree for efficient nearest neighbor search
        self.kd_tree = None
        self.positions = []
        
        # Minimum distance between keyframes 
        self.min_keyframe_distance = 0.2  # meters
        
        # Minimum time between keyframes
        self.min_keyframe_time = 1.0  # seconds
        
        # Last added keyframe time
        self.last_keyframe_time = 0
        
        # Minimum distance for loop closure candidates
        self.min_loop_dist = 3.0  # meters
        
        # Time window for loop closure (don't match with very recent frames)
        self.min_loop_time = 10.0  # seconds
        
        # Flag to determine if we should be checking for loop closures
        self.enabled = True
        
        # Counter for diagnostic logging
        self.frames_processed = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def add_keyframe(self, depth_image, position, orientation):
        """
        Add a new keyframe to the database if it's sufficiently different
        
        Args:
            depth_image: Depth image
            position: (x, y, z) camera position
            orientation: (x, y, z, w) camera orientation (quaternion)
            
        Returns:
            bool: True if keyframe was added
        """
        current_time = time.time()
        
        # Skip if too soon after last keyframe
        if current_time - self.last_keyframe_time < self.min_keyframe_time:
            return False
            
        with self.lock:
            # Check if we have previous keyframes
            if len(self.fingerprints) > 0:
                # Check distance to latest keyframe
                latest_pos = self.fingerprints[-1].position
                dist_to_latest = np.linalg.norm(np.array(position) - latest_pos)
                
                # Skip if too close to previous keyframe
                if dist_to_latest < self.min_keyframe_distance:
                    return False
            
            # Create fingerprint
            fingerprint = DepthFingerprint(depth_image, position, orientation)
            
            # Add to database
            self.fingerprints.append(fingerprint)
            self.positions.append(fingerprint.position)
            
            # Rebuild KD tree periodically (every 10 frames)
            if len(self.fingerprints) % 10 == 0 or self.kd_tree is None:
                self.kd_tree = KDTree(np.array(self.positions))
                
            self.last_keyframe_time = current_time
            self.frames_processed += 1
            
            # Log info every 10 frames
            if len(self.fingerprints) % 10 == 0:
                self.logger.info(
                    f"{CYAN}Loop closure keyframe database has {len(self.fingerprints)} entries{END}"
                )
                
            return True
    
    def detect_loop_closure(self, depth_image, current_position, current_orientation):
        """
        Detect if we've returned to a previously visited location
        
        Args:
            depth_image: Current depth image
            current_position: (x, y, z) current position
            current_orientation: (x, y, z, w) current orientation (quaternion)
            
        Returns:
            tuple: (detected, match_idx, transform) - whether loop closure was detected,
                   the index of the matching fingerprint, and the transform to apply
        """
        if not self.enabled or len(self.fingerprints) < 10:
            return False, -1, None
            
        current_time = time.time()
        
        with self.lock:
            current_fingerprint = DepthFingerprint(depth_image, current_position, current_orientation)
            
            # Use KD tree to find potential spatial neighbors efficiently
            if self.kd_tree is not None:
                # Query for potential matches
                current_pos_np = np.array([current_position]).reshape(1, -1)
                
                # Find neighbors with distance > min_loop_dist to avoid matching with recent trajectory
                distances, indices = self.kd_tree.query(current_pos_np, k=5)
                distances = distances[0]
                indices = indices[0]
                
                # Filter candidates by distance - we want places that are far in 
                # trajectory distance but close in 3D distance
                candidates = []
                
                for dist, idx in zip(distances, indices):
                    candidate = self.fingerprints[idx]
                    
                    # Skip recent frames
                    time_diff = current_time - candidate.timestamp
                    if time_diff < self.min_loop_time:
                        continue
                        
                    # Skip frames that are too far away (noisy matches)
                    if dist > 5.0:
                        continue
                        
                    # Compute full similarity including feature matching
                    similarity = current_fingerprint.similarity(candidate)
                    
                    self.logger.debug(
                        f"Loop closure candidate: idx={idx}, dist={dist:.2f}m, "
                        f"similarity={similarity:.3f}, age={time_diff:.1f}s"
                    )
                    
                    # Good match if similarity is high
                    if similarity > 0.75:
                        candidates.append((similarity, idx, candidate))
                
                # If we have candidates, check for the best match
                if candidates:
                    # Sort by similarity score (descending)
                    candidates.sort(reverse=True)
                    
                    # Get best match
                    best_sim, best_idx, best_match = candidates[0]
                    
                    self.logger.info(
                        f"{GREEN}Found potential loop closure match! "
                        f"Similarity: {best_sim:.3f}, index: {best_idx}, "
                        f"position diff: {np.linalg.norm(best_match.position - current_fingerprint.position):.2f}m{END}"
                    )
                    
                    # Compute transform between matched fingerprints
                    transform = self._compute_transform(best_match, current_fingerprint)
                    
                    # Return success
                    return True, best_idx, transform
            
            # No match found
            return False, -1, None
            
    def _compute_transform(self, match_fingerprint, current_fingerprint):
        """
        Compute the transform to align the current position with the matched position
        
        This gives us the correction needed for loop closure
        
        Args:
            match_fingerprint: The matched fingerprint from history
            current_fingerprint: The current fingerprint
            
        Returns:
            Transform: The transform to apply for correction
        """
        # Get positions and orientations
        p1 = match_fingerprint.position
        q1 = match_fingerprint.orientation
        
        p2 = current_fingerprint.position
        q2 = current_fingerprint.orientation
        
        # Create the transform message
        transform = Transform()
        
        # Correction is the difference between where we think we are and where we were
        # when we first saw this place
        transform.translation.x = p1[0] - p2[0]
        transform.translation.y = p1[1] - p2[1]
        transform.translation.z = p1[2] - p2[2]
        
        # For rotation, we need to compute q1 * q2^-1
        # This gives the rotation from current orientation to matched orientation
        # Python quaternions are typically (x, y, z, w) but many libraries use (w, x, y, z)
        # Make sure to check the library you're using
        r1 = R.from_quat([q1[0], q1[1], q1[2], q1[3]])
        r2 = R.from_quat([q2[0], q2[1], q2[2], q2[3]])
        
        r_diff = r1 * r2.inv()
        quat_diff = r_diff.as_quat()
        
        transform.rotation.x = quat_diff[0]
        transform.rotation.y = quat_diff[1]
        transform.rotation.z = quat_diff[2]
        transform.rotation.w = quat_diff[3]
        
        self.logger.info(
            f"{GREEN}Loop closure correction: "
            f"translation=({transform.translation.x:.3f}, {transform.translation.y:.3f}, {transform.translation.z:.3f}), "
            f"rotation=({transform.rotation.x:.3f}, {transform.rotation.y:.3f}, {transform.rotation.z:.3f}, {transform.rotation.w:.3f}){END}"
        )
        
        return transform


class LoopClosureNode(Node):
    """ROS node for loop closure detection and correction"""
    
    def __init__(self):
        super().__init__('loop_closure_node')
        self.get_logger().info(f"{GREEN}{BOLD}Loop Closure Node starting...{END}")
        
        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Create loop closure detector
        self.detector = LoopClosureDetector(self.get_logger())
        
        # Declare parameters
        self.declare_parameter('camera_frame', 'zed_left_camera_frame')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('grid_topic', '/occupancy_grid')
        self.declare_parameter('pose_topic', '/zed/zed_node/pose')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        
        # Get parameters
        self.camera_frame = self.get_parameter('camera_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.grid_topic = self.get_parameter('grid_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        
        # Set up QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Important for map data
            depth=10
        )
        
        # Publishers
        self.loop_closure_pub = self.create_publisher(
            Bool, '/loop_closure/detected', 10
        )
        self.info_pub = self.create_publisher(
            String, '/loop_closure/info', 10
        )
        self.corrected_grid_pub = self.create_publisher(
            OccupancyGrid, '/occupancy_grid_corrected', qos_profile
        )
        
        # Subscribers
        self.pose_subscriber = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10
        )
        self.depth_subscriber = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, 10
        )
        self.grid_subscriber = self.create_subscription(
            OccupancyGrid, self.grid_topic, self.grid_callback, qos_profile
        )
        
        # TF buffer and listener for getting transforms
        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Store latest data
        self.latest_depth_image = None
        self.latest_pose = None
        self.latest_grid = None
        
        # Processing lock
        self.lock = threading.Lock()
        
        # Flag to indicate if loop closure was detected
        self.loop_closure_detected = False
        self.last_loop_closure_time = 0
        self.min_time_between_loop_closures = 5.0  # seconds
        
        # Set up timer to process data
        self.timer = self.create_timer(0.5, self.process_callback)
        
        self.get_logger().info(
            f"{GREEN}Loop closure node initialized with:{END}\n"
            f"- Camera frame: {self.camera_frame}\n"
            f"- Map frame: {self.map_frame}\n"
            f"- Grid topic: {self.grid_topic}\n"
            f"- Pose topic: {self.pose_topic}\n"
            f"- Depth topic: {self.depth_topic}"
        )
    
    def pose_callback(self, pose_msg):
        """Store the latest pose"""
        with self.lock:
            self.latest_pose = pose_msg
            
    def depth_callback(self, depth_msg):
        """Store the latest depth image"""
        try:
            # Convert depth image to OpenCV format (float32 in meters)
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            with self.lock:
                self.latest_depth_image = depth_image
                
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {str(e)}")
            
    def grid_callback(self, grid_msg):
        """Store the latest occupancy grid"""
        with self.lock:
            self.latest_grid = grid_msg
            
    def process_callback(self):
        """Main processing loop"""
        # Skip if we don't have all required data
        with self.lock:
            if (self.latest_depth_image is None or
                self.latest_pose is None or
                self.latest_grid is None):
                return
                
            # Make local copies to avoid data changing during processing
            depth_image = self.latest_depth_image.copy()
            pose = self.latest_pose
            grid = self.latest_grid
            
        # Extract current position and orientation
        pos = pose.pose.position
        orient = pose.pose.orientation
        
        current_position = (pos.x, pos.y, pos.z)
        current_orientation = (orient.x, orient.y, orient.z, orient.w)
        
        # Add keyframe to database
        self.detector.add_keyframe(depth_image, current_position, current_orientation)
        
        # Only check for loop closures if we have enough keyframes
        current_time = time.time()
        if (len(self.detector.fingerprints) > 20 and
            current_time - self.last_loop_closure_time > self.min_time_between_loop_closures):
            
            # Check for loop closure
            detected, match_idx, transform = self.detector.detect_loop_closure(
                depth_image, current_position, current_orientation
            )
            
            # If loop closure detected, apply correction
            if detected and transform is not None:
                self.get_logger().info(
                    f"{GREEN}{BOLD}Loop closure detected! "
                    f"Current position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}), "
                    f"Match index: {match_idx}{END}"
                )
                
                # Publish detection and info
                detection_msg = Bool()
                detection_msg.data = True
                self.loop_closure_pub.publish(detection_msg)
                
                info_msg = String()
                info_msg.data = (
                    f"Loop closure detected! Match idx: {match_idx}, "
                    f"Translation: ({transform.translation.x:.3f}, {transform.translation.y:.3f}, {transform.translation.z:.3f}), "
                    f"Rotation: ({transform.rotation.x:.3f}, {transform.rotation.y:.3f}, {transform.rotation.z:.3f}, {transform.rotation.w:.3f})"
                )
                self.info_pub.publish(info_msg)
                
                # Apply correction to grid (simplified for now)
                corrected_grid = self.apply_correction_to_grid(grid, transform)
                
                # Publish corrected grid
                self.corrected_grid_pub.publish(corrected_grid)
                
                # Update timestamp
                self.last_loop_closure_time = current_time
                self.loop_closure_detected = True
                
                return
                
        # If we get here, no loop closure was detected in this iteration
        if self.loop_closure_detected:
            # Reset and publish false detection
            detection_msg = Bool()
            detection_msg.data = False
            self.loop_closure_pub.publish(detection_msg)
            self.loop_closure_detected = False
            
    def apply_correction_to_grid(self, grid, transform):
        """
        Apply the loop closure correction to the occupancy grid
        
        Args:
            grid: Original occupancy grid
            transform: Transform to apply
            
        Returns:
            OccupancyGrid: Corrected grid
        """
        # Create a copy of the original grid
        corrected_grid = OccupancyGrid()
        corrected_grid.header = grid.header
        corrected_grid.info = grid.info
        
        # Get grid info
        width = grid.info.width
        height = grid.info.height
        resolution = grid.info.resolution
        
        # Get transform parameters
        tx = transform.translation.x
        ty = transform.translation.y
        
        # Convert to grid cell coordinates
        cell_tx = int(tx / resolution)
        cell_ty = int(ty / resolution)
        
        # Create a copy of the data
        corrected_data = list(grid.data)
        
        # Apply the shift to the grid data
        # This is a simplified approach - a full implementation would handle
        # rotation and interpolation more carefully
        new_data = [-1] * (width * height)
        
        for y in range(height):
            for x in range(width):
                # Original index
                orig_idx = y * width + x
                
                # Apply transform
                new_x = x + cell_tx
                new_y = y + cell_ty
                
                # Check bounds
                if 0 <= new_x < width and 0 <= new_y < height:
                    new_idx = new_y * width + new_x
                    new_data[new_idx] = corrected_data[orig_idx]
        
        corrected_grid.data = new_data
        
        # Add debug information to header timestamp
        corrected_grid.header.stamp = self.get_clock().now().to_msg()
        
        self.get_logger().info(f"{GREEN}Created corrected grid with loop closure transform{END}")
        
        return corrected_grid


def main(args=None):
    """Main entry point for the loop closure node"""
    try:
        # Initialize ROS
        rclpy.init(args=args)
        
        # Create node
        node = LoopClosureNode()
        
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean shutdown
        rclpy.shutdown()


if __name__ == '__main__':
    main()
