#!/usr/bin/env python3

"""
CUDA Acceleration Module for ZED Occupancy Grid
This module provides GPU-accelerated functions for processing depth data and updating
occupancy grids using NVIDIA CUDA.
"""

import numpy as np
import math
import time

# Utility function to find CUDA driver
def find_cuda_driver():
    import os
    
    # List of common paths for libcuda.so on different Linux distributions
    common_paths = [
        '/usr/lib/x86_64-linux-gnu/libcuda.so',
        '/usr/lib/aarch64-linux-gnu/libcuda.so',  # For ARM64 (Jetson)
        '/usr/lib/libcuda.so',
        '/usr/lib64/libcuda.so',
        '/opt/cuda/lib64/libcuda.so',
        # Add more paths as needed
    ]
    
    # Check if NUMBA_CUDA_DRIVER is set in environment, use that first
    driver_path = os.environ.get('NUMBA_CUDA_DRIVER')
    if driver_path and os.path.exists(driver_path):
        print(f"Using CUDA driver from environment variable: {driver_path}")
        return driver_path
    elif driver_path:
        print(f"Warning: NUMBA_CUDA_DRIVER is set to {driver_path} but file doesn't exist")
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found CUDA driver at: {path}")
            # Set environment variable for numba
            os.environ['NUMBA_CUDA_DRIVER'] = path
            return path
    
    print("No CUDA driver found in common locations")
    return None

# CUDA imports
try:
    # Try to find CUDA driver first
    cuda_driver = find_cuda_driver()
    if cuda_driver is None:
        print("❌ CUDA driver not found. GPU acceleration will not be available.")
        raise ImportError("CUDA driver not found")
    
    import cupy as cp
    from numba import cuda
    import numba
    
    # ANSI color codes for terminal output
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"{GREEN}{BOLD}✅ CUDA GPU ACCELERATION ENABLED! 🚀{END}")
        print(f"{GREEN}✅ CUDA libraries successfully imported in acceleration module!{END}")
        
        # Print CUDA device information
        device = cuda.get_current_device()
        print(f"\n{CYAN}{BOLD}CUDA Device Information:{END}")
        print(f"{CYAN}Device Name: {device.name}{END}")
        print(f"{CYAN}Compute Capability: {device.compute_capability}{END}")
        print(f"{CYAN}Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}{END}")
        print(f"{CYAN}Max Block Dimensions: {device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z}{END}")
        print(f"{CYAN}Max Grid Dimensions: {device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z}{END}")
        print(f"{CYAN}Warp Size: {device.WARP_SIZE}{END}")
        
        # Recommended block size based on device
        if device.compute_capability[0] >= 8:  # Ampere or newer
            RECOMMENDED_BLOCK_SIZE = (32, 32)
        else:
            RECOMMENDED_BLOCK_SIZE = (16, 16)
            
        print(f"Recommended Thread Block Size: {RECOMMENDED_BLOCK_SIZE}")
    else:
        CUDA_AVAILABLE = False
        print(f"{RED}{BOLD}❌ CUDA IS NOT AVAILABLE - USING CPU FALLBACK ❌{END}")
        print(f"{RED}GPU acceleration will not be available.{END}")
        RECOMMENDED_BLOCK_SIZE = (16, 16)  # Default value for when CUDA is not available
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"{RED}{BOLD}❌ CUDA LIBRARIES NOT FOUND - USING CPU FALLBACK ❌{END}")
    print(f"{RED}Error: {e}{END}")
    print(f"{RED}GPU acceleration will not be available.{END}")
    RECOMMENDED_BLOCK_SIZE = (16, 16)  # Default value for when CUDA is not available

# Define CUDA kernels using Numba's @cuda.jit decorator
if CUDA_AVAILABLE:
    # 1. Transform depth points to world coordinates
    @cuda.jit
    def transform_points_kernel(depth_image, rotation_matrix, camera_pos, 
                               width, height, step, tan_fov_h, tan_fov_v,
                               min_depth, max_depth, world_points):
        # Get thread indices
        tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        
        # Check if within image bounds and step size
        if tx < width and ty < height and tx % step == 0 and ty % step == 0:
            # Get depth value
            depth = depth_image[ty, tx]
            
            # Skip invalid depth values
            if not math.isfinite(depth) or depth < min_depth or depth > max_depth:
                world_points[ty, tx, 0] = -1000.0  # Mark as invalid
                return
                
            # Calculate normalized image coordinates
            normalized_u = (2.0 * tx / width - 1.0)
            normalized_v = (2.0 * ty / height - 1.0)
            
            # Calculate 3D vector from camera using field of view
            ray_x = normalized_u * tan_fov_h * depth
            ray_y = normalized_v * tan_fov_v * depth
            ray_z = depth
            
            # Transform ray from camera to world coordinates
            world_x = (rotation_matrix[0, 0] * ray_x +
                       rotation_matrix[0, 1] * ray_y +
                       rotation_matrix[0, 2] * ray_z) + camera_pos[0]
            
            world_y = (rotation_matrix[1, 0] * ray_x +
                       rotation_matrix[1, 1] * ray_y +
                       rotation_matrix[1, 2] * ray_z) + camera_pos[1]
            
            world_z = (rotation_matrix[2, 0] * ray_x +
                       rotation_matrix[2, 1] * ray_y +
                       rotation_matrix[2, 2] * ray_z) + camera_pos[2]
            
            # Store result in output array
            world_points[ty, tx, 0] = world_x
            world_points[ty, tx, 1] = world_y
            world_points[ty, tx, 2] = world_z

    # 2. Update grid cells from ray endpoints
    @cuda.jit
    def update_endpoints_kernel(world_points, grid_origin_x, grid_origin_y, 
                               resolution, grid_cols, grid_rows,
                               camera_x, camera_y, max_ray_length,
                               log_odds_grid, cell_height_grid, observation_count_grid,
                               log_odds_occupied, log_odds_min, log_odds_max,
                               current_alpha, temporal_filtering):
        # Get thread indices
        tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        
        # Check if within points array bounds
        if tx < world_points.shape[1] and ty < world_points.shape[0]:
            # Get world point
            world_x = world_points[ty, tx, 0]
            world_y = world_points[ty, tx, 1]
            world_z = world_points[ty, tx, 2]
            
            # Skip invalid points (marked with -1000)
            if world_x <= -999.0:
                return
                
            # Skip points that are too far from camera
            dist_sq = (world_x - camera_x)**2 + (world_y - camera_y)**2
            if dist_sq > max_ray_length**2:
                return
            
            # Convert world coordinates to grid cell coordinates
            grid_x = int((world_x - grid_origin_x) / resolution)
            grid_y = int((world_y - grid_origin_y) / resolution)
            
            # Skip if out of grid bounds
            if grid_x < 0 or grid_x >= grid_cols or grid_y < 0 or grid_y >= grid_rows:
                return
            
            # Update log-odds for occupied cell (endpoint)
            if temporal_filtering:
                old_val = log_odds_grid[grid_y, grid_x]
                new_val = (1.0 - current_alpha) * old_val + current_alpha * log_odds_occupied
                # Ensure log-odds value stays within bounds
                new_val = max(log_odds_min, min(log_odds_max, new_val))
                log_odds_grid[grid_y, grid_x] = new_val
            else:
                log_odds_grid[grid_y, grid_x] += log_odds_occupied
                # Ensure log-odds value stays within bounds
                log_odds_grid[grid_y, grid_x] = max(log_odds_min, min(log_odds_max, log_odds_grid[grid_y, grid_x]))
            
            # Update height value
            if cell_height_grid[grid_y, grid_x] == 0:
                cell_height_grid[grid_y, grid_x] = world_z
            else:
                # Average with previous height
                cell_height_grid[grid_y, grid_x] = 0.7 * cell_height_grid[grid_y, grid_x] + 0.3 * world_z
            
            # Increment observation count - use atomic add to prevent race conditions
            cuda.atomic.add(observation_count_grid, (grid_y, grid_x), 1)

    # 3. Ray tracing kernel to mark free space along rays
    @cuda.jit
    def raytrace_kernel(camera_grid_x, camera_grid_y, world_points,
                       grid_origin_x, grid_origin_y, resolution,
                       grid_cols, grid_rows, log_odds_grid, observation_count_grid,
                       log_odds_free, log_odds_min, log_odds_max, current_alpha,
                       temporal_filtering, ray_step):
        # Get thread indices
        tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        
        # Check if within points array bounds
        if tx < world_points.shape[1] and ty < world_points.shape[0]:
            # Get world point
            world_x = world_points[ty, tx, 0]
            world_y = world_points[ty, tx, 1]
            
            # Skip invalid points
            if world_x <= -999.0:
                return
            
            # Convert world coordinates to grid cell coordinates
            end_grid_x = int((world_x - grid_origin_x) / resolution)
            end_grid_y = int((world_y - grid_origin_y) / resolution)
            
            # Skip if endpoint is out of grid bounds
            if end_grid_x < 0 or end_grid_x >= grid_cols or end_grid_y < 0 or end_grid_y >= grid_rows:
                return
            
            # Skip if endpoint is the same as camera position
            if end_grid_x == camera_grid_x and end_grid_y == camera_grid_y:
                return
                
            # Perform Bresenham ray tracing to mark cells along the ray as free
            # Based on Bresenham's line algorithm
            x0, y0 = camera_grid_x, camera_grid_y
            x1, y1 = end_grid_x, end_grid_y
            
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            
            x, y = x0, y0
            count = 0
            
            while x != x1 or y != y1:
                if (x != x0 or y != y0) and count % ray_step == 0:
                    # Only process cells that are within the grid bounds
                    if 0 <= x < grid_cols and 0 <= y < grid_rows:
                        # Update log-odds for free cell (cells along the ray)
                        if temporal_filtering:
                            old_val = log_odds_grid[y, x]
                            new_val = (1.0 - current_alpha) * old_val + current_alpha * log_odds_free
                            # Ensure log-odds value stays within bounds
                            new_val = max(log_odds_min, min(log_odds_max, new_val))
                            log_odds_grid[y, x] = new_val
                        else:
                            log_odds_grid[y, x] += log_odds_free
                            # Ensure log-odds value stays within bounds
                            log_odds_grid[y, x] = max(log_odds_min, min(log_odds_max, log_odds_grid[y, x]))
                        
                        # Increment observation count
                        cuda.atomic.add(observation_count_grid, (y, x), 1)
                
                # Bresenham algorithm step
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
                    
                count += 1
                
                # Break if we've gone too far (safety check)
                if count > 1000:  # Arbitrary limit to prevent infinite loops
                    break

    # 4. Log-odds to occupancy probability conversion kernel
    @cuda.jit
    def log_odds_to_occupancy_kernel(log_odds_grid, observation_count_grid, min_observations,
                                    occupied_threshold, free_threshold, grid_data):
        # Get thread indices
        tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        
        # Check if within grid bounds
        if tx < log_odds_grid.shape[1] and ty < log_odds_grid.shape[0]:
            # Calculate flat grid index
            idx = ty * log_odds_grid.shape[1] + tx
            
            # Default to unknown (-1)
            grid_data[idx] = -1
            
            # Check if cell has enough observations
            if observation_count_grid[ty, tx] >= min_observations:
                log_odds = log_odds_grid[ty, tx]
                
                # Classify cell based on log-odds
                if log_odds > occupied_threshold:
                    grid_data[idx] = 100  # Occupied
                elif log_odds < free_threshold:
                    grid_data[idx] = 0    # Free
                else:
                    # Convert log-odds to probability in [0, 100]
                    # P(occupied) = 1 - 1/(1 + exp(log_odds))
                    prob = 1.0 - (1.0 / (1.0 + math.exp(log_odds)))
                    grid_data[idx] = int(prob * 100)


class CudaAccelerator:
    """CUDA-based accelerator for occupancy grid computations"""
    
    def __init__(self, logger=None):
        """Initialize CUDA accelerator"""
        self.logger = logger
        self.cuda_available = CUDA_AVAILABLE
        
        if self.cuda_available:
            self.threads_per_block = RECOMMENDED_BLOCK_SIZE
            if logger:
                logger.info(f"CUDA accelerator initialized with thread block size: {self.threads_per_block}")
        else:
            if logger:
                logger.warn("CUDA accelerator initialized but CUDA is not available - using CPU fallback")
    
    def log(self, level, message):
        """Log a message if logger is available"""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warn':
                self.logger.warn(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def is_available(self):
        """Check if CUDA acceleration is available"""
        return self.cuda_available
    
    def update_grid(self, depth_image, transform_data, grid_data, params):
        """
        Update occupancy grid using CUDA acceleration
        
        Args:
            depth_image: 2D numpy array of depth values
            transform_data: Dict containing camera transform information
            grid_data: Dict containing grid data (log_odds_grid, etc.)
            params: Dict containing algorithm parameters
            
        Returns:
            Dict with updated grid data and statistics
        """
        if not self.cuda_available:
            self.log('warn', "CUDA not available, cannot use GPU acceleration")
            return None
        
        start_time = time.time()
        
        try:
            # Extract transform data
            rotation_matrix = transform_data['rotation_matrix']
            camera_pos = transform_data['camera_pos']
            camera_grid_x = transform_data['camera_grid_x']
            camera_grid_y = transform_data['camera_grid_y']
            
            # Extract grid data
            log_odds_grid = grid_data['log_odds_grid']
            cell_height_grid = grid_data['cell_height_grid']
            observation_count_grid = grid_data['observation_count_grid']
            
            # Extract parameters
            height, width = depth_image.shape
            step = params['step']
            min_depth = params['min_depth']
            max_depth = params['max_depth']
            log_odds_free = params['log_odds_free']
            log_odds_occupied = params['log_odds_occupied']
            log_odds_min = params['log_odds_min']
            log_odds_max = params['log_odds_max']
            grid_origin_x = params['grid_origin_x']
            grid_origin_y = params['grid_origin_y']
            resolution = params['resolution']
            grid_rows, grid_cols = log_odds_grid.shape
            max_ray_length = params['max_ray_length']
            current_alpha = params['current_alpha']
            temporal_filtering = params['temporal_filtering']
            ray_step = params.get('ray_step', 4)  # Step size for ray tracing
            
            # Transfer data to GPU
            transfer_start = time.time()
            
            # Convert to float32 for better GPU performance
            if depth_image.dtype != np.float32:
                depth_image = depth_image.astype(np.float32)
                
            d_depth_image = cuda.to_device(depth_image)
            d_rotation_matrix = cuda.to_device(rotation_matrix.astype(np.float32))
            d_camera_pos = cuda.to_device(camera_pos.astype(np.float32))
            
            # Create output arrays with an initial value of -1000.0 (invalid marker)
            d_world_points = cuda.to_device(np.full((height, width, 3), -1000.0, dtype=np.float32))
            
            # Grid arrays
            d_log_odds_grid = cuda.to_device(log_odds_grid)
            d_cell_height_grid = cuda.to_device(cell_height_grid)
            d_observation_count_grid = cuda.to_device(observation_count_grid)
            
            transfer_time = time.time() - transfer_start
            self.log('debug', f"GPU data transfer time: {transfer_time:.4f} seconds")
            
            # Set up grid and block dimensions for kernel launch
            # For image processing kernels
            image_blocks_per_grid = (
                (width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (height + self.threads_per_block[1] - 1) // self.threads_per_block[1]
            )
            
            # Step 1: Transform depth points to world coordinates
            kernel_start = time.time()
            
            transform_points_kernel[image_blocks_per_grid, self.threads_per_block](
                d_depth_image, d_rotation_matrix, d_camera_pos,
                width, height, step, params['tan_fov_h'], params['tan_fov_v'],
                min_depth, max_depth, d_world_points
            )
            
            # Synchronize to ensure kernel completion
            cuda.synchronize()
            transform_time = time.time() - kernel_start
            self.log('debug', f"Transform points kernel time: {transform_time:.4f} seconds")
            
            # Step 2: Update grid cells for ray endpoints (mark as occupied)
            endpoint_start = time.time()
            
            update_endpoints_kernel[image_blocks_per_grid, self.threads_per_block](
                d_world_points, grid_origin_x, grid_origin_y,
                resolution, grid_cols, grid_rows,
                camera_pos[0], camera_pos[1], max_ray_length,
                d_log_odds_grid, d_cell_height_grid, d_observation_count_grid,
                log_odds_occupied, log_odds_min, log_odds_max,
                current_alpha, temporal_filtering
            )
            
            # Synchronize to ensure kernel completion
            cuda.synchronize()
            endpoint_time = time.time() - endpoint_start
            self.log('debug', f"Update endpoints kernel time: {endpoint_time:.4f} seconds")
            
            # Step 3: Ray tracing to mark free cells
            raytrace_start = time.time()
            
            raytrace_kernel[image_blocks_per_grid, self.threads_per_block](
                camera_grid_x, camera_grid_y, d_world_points,
                grid_origin_x, grid_origin_y, resolution,
                grid_cols, grid_rows, d_log_odds_grid, d_observation_count_grid,
                log_odds_free, log_odds_min, log_odds_max, current_alpha,
                temporal_filtering, ray_step
            )
            
            # Synchronize to ensure kernel completion
            cuda.synchronize()
            raytrace_time = time.time() - raytrace_start
            self.log('debug', f"Ray tracing kernel time: {raytrace_time:.4f} seconds")
            
            # Copy results back to CPU
            copy_back_start = time.time()
            
            updated_log_odds_grid = d_log_odds_grid.copy_to_host()
            updated_cell_height_grid = d_cell_height_grid.copy_to_host()
            updated_observation_count_grid = d_observation_count_grid.copy_to_host()
            
            copy_back_time = time.time() - copy_back_start
            self.log('debug', f"GPU -> CPU transfer time: {copy_back_time:.4f} seconds")
            
            # Count valid points for statistics
            valid_points = np.count_nonzero(d_world_points[:, :, 0].copy_to_host() > -999.0)
            
            total_time = time.time() - start_time
            self.log('info', f"CUDA accelerated grid update completed in {total_time:.4f} seconds")
            self.log('info', f"Processed {valid_points} valid points")
            
            # Return updated grid data and statistics
            return {
                'log_odds_grid': updated_log_odds_grid,
                'cell_height_grid': updated_cell_height_grid,
                'observation_count_grid': updated_observation_count_grid,
                'stats': {
                    'valid_points': valid_points,
                    'total_time': total_time,
                    'transfer_time': transfer_time,
                    'transform_time': transform_time,
                    'endpoint_time': endpoint_time,
                    'raytrace_time': raytrace_time,
                    'copy_back_time': copy_back_time
                }
            }
            
        except Exception as e:
            self.log('error', f"Error in CUDA grid update: {str(e)}")
            return None
    
    def create_occupancy_grid_message(self, grid_data, params):
        """
        Convert log-odds grid to occupancy grid message data using CUDA
        
        Args:
            grid_data: Dict containing grid data (log_odds_grid, observation_count_grid)
            params: Dict containing parameters
            
        Returns:
            numpy array of int8 values representing occupancy probabilities
        """
        if not self.cuda_available:
            self.log('warn', "CUDA not available, cannot use GPU acceleration")
            return None
        
        try:
            # Extract grid data
            log_odds_grid = grid_data['log_odds_grid']
            observation_count_grid = grid_data['observation_count_grid']
            
            # Extract parameters
            min_observations = params['min_observations']
            occupied_threshold = params['occupied_threshold']
            free_threshold = params['free_threshold']
            
            # Get grid dimensions
            grid_rows, grid_cols = log_odds_grid.shape
            
            # Create output array
            grid_msg_data = np.full(grid_rows * grid_cols, -1, dtype=np.int8)
            
            # Transfer data to GPU
            d_log_odds_grid = cuda.to_device(log_odds_grid)
            d_observation_count_grid = cuda.to_device(observation_count_grid)
            d_grid_msg_data = cuda.to_device(grid_msg_data)
            
            # Set up grid and block dimensions for kernel launch
            threads_per_block = self.threads_per_block
            blocks_per_grid = (
                (grid_cols + threads_per_block[0] - 1) // threads_per_block[0],
                (grid_rows + threads_per_block[1] - 1) // threads_per_block[1]
            )
            
            # Launch kernel to convert log-odds to occupancy probabilities
            log_odds_to_occupancy_kernel[blocks_per_grid, threads_per_block](
                d_log_odds_grid, d_observation_count_grid, min_observations,
                occupied_threshold, free_threshold, d_grid_msg_data
            )
            
            # Synchronize to ensure kernel completion
            cuda.synchronize()
            
            # Copy results back to CPU
            grid_msg_data = d_grid_msg_data.copy_to_host()
            
            return grid_msg_data
            
        except Exception as e:
            self.log('error', f"Error in CUDA occupancy grid creation: {str(e)}")
            return None


# If this file is run directly, perform a basic test
if __name__ == "__main__":
    if CUDA_AVAILABLE:
        # Create a small test
        print("\nRunning basic CUDA acceleration test...")
        
        # Create test data
        depth_image = np.random.uniform(0.5, 10.0, (480, 640)).astype(np.float32)
        # Add some NaNs to simulate invalid depth values
        invalid_mask = np.random.random((480, 640)) > 0.9
        depth_image[invalid_mask] = np.nan
        
        # Create rotation matrix and camera position
        rotation_matrix = np.eye(3, dtype=np.float32)
        camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Create grid data
        grid_rows, grid_cols = 200, 200
        log_odds_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        cell_height_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        observation_count_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
        
        # Create parameters
        params = {
            'step': 4,
            'min_depth': 0.5,
            'max_depth': 20.0,
            'log_odds_free': -0.4,
            'log_odds_occupied': 0.9,
            'log_odds_min': -5.0,
            'log_odds_max': 5.0,
            'grid_origin_x': -5.0,
            'grid_origin_y': -5.0,
            'resolution': 0.05,
            'max_ray_length': 10.0,
            'current_alpha': 0.3,
            'temporal_filtering': True,
            'tan_fov_h': np.tan(np.radians(55)),
            'tan_fov_v': np.tan(np.radians(35)),
            'min_observations': 1,
            'occupied_threshold': 0.1,
            'free_threshold': -0.5
        }
        
        # Create transform data
        transform_data = {
            'rotation_matrix': rotation_matrix,
            'camera_pos': camera_pos,
            'camera_grid_x': 100,
            'camera_grid_y': 100
        }
        
        # Create grid data
        grid_data = {
            'log_odds_grid': log_odds_grid,
            'cell_height_grid': cell_height_grid,
            'observation_count_grid': observation_count_grid
        }
        
        # Create accelerator
        accelerator = CudaAccelerator()
        
        # Test grid update
        start_time = time.time()
        result = accelerator.update_grid(depth_image, transform_data, grid_data, params)
        total_time = time.time() - start_time
        
        if result:
            print(f"\nCUDA acceleration test completed in {total_time:.4f} seconds")
            print(f"Processed {result['stats']['valid_points']} valid points")
            print("Test successful!")
        else:
            print("CUDA acceleration test failed")
    else:
        print("\nCUDA is not available - cannot run test")
