#!/usr/bin/env python3

import os
import time
import numpy as np
import math

# Test CUDA imports
print("Testing CUDA imports...")
try:
    import cupy as cp
    from numba import cuda
    CUDA_AVAILABLE = True
    print("✅ CUDA libraries successfully imported!")
    
    # Print CUDA device info
    if cuda.is_available():
        print("\nCUDA Device Information:")
        print(f"CUDA Available: {cuda.is_available()}")
        device = cuda.get_current_device()
        print(f"Device Name: {device.name}")
        print(f"Compute Capability: {device.compute_capability}")
        print(f"Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
        print(f"Max Block Dimensions: {device.MAX_BLOCK_DIM_X}, {device.MAX_BLOCK_DIM_Y}, {device.MAX_BLOCK_DIM_Z}")
        print(f"Max Grid Dimensions: {device.MAX_GRID_DIM_X}, {device.MAX_GRID_DIM_Y}, {device.MAX_GRID_DIM_Z}")
        print(f"Warp Size: {device.WARP_SIZE}")
    else:
        print("CUDA is available but no CUDA-capable device was found.")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"❌ Failed to import CUDA libraries: {e}")

# Define sample data size
# Increase problem size to better showcase GPU acceleration
width, height = 1280, 720
grid_rows, grid_cols = 500, 500

# Create a simple performance benchmark
def run_benchmark():
    print("\nRunning CUDA vs CPU Performance Benchmark:")
    
    # Create sample depth data
    print("Creating sample depth data...")
    depth_image = np.random.uniform(0.5, 10.0, (height, width)).astype(np.float32)
    # Add some NaNs to simulate invalid depth values
    invalid_mask = np.random.random((height, width)) > 0.9
    depth_image[invalid_mask] = np.nan
    
    # Create a sample grid
    log_odds_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    cell_height_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    observation_count_grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    
    # Create rotation matrix and camera position
    rotation_matrix = np.eye(3, dtype=np.float32)
    camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Sample parameters
    resolution = 0.05
    grid_origin_x = -5.0
    grid_origin_y = -5.0
    tan_fov_h = math.tan(math.radians(55))
    tan_fov_v = math.tan(math.radians(35))
    # Use smaller step size to increase workload - this will create more parallel work
    step = 2
    min_depth = 0.5
    max_depth = 20.0
    log_odds_min = -5.0
    log_odds_max = 5.0
    log_odds_occupied = 2.0
    current_alpha = 0.3
    max_ray_length = 5.0
    
    # Benchmark CPU implementation (simplified)
    print("\nBenchmarking CPU implementation...")
    cpu_start = time.time()
    
    # Simplified CPU implementation
    for v in range(0, height, step):
        for u in range(0, width, step):
            # Get depth value
            depth = depth_image[v, u]
            
            # Skip invalid depth values
            if not np.isfinite(depth) or depth < min_depth or depth > max_depth:
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
                      rotation_matrix[0, 2] * ray_z) + camera_pos[0]
            
            world_y = (rotation_matrix[1, 0] * ray_x +
                      rotation_matrix[1, 1] * ray_y +
                      rotation_matrix[1, 2] * ray_z) + camera_pos[1]
            
            world_z = (rotation_matrix[2, 0] * ray_x +
                      rotation_matrix[2, 1] * ray_y +
                      rotation_matrix[2, 2] * ray_z) + camera_pos[2]
            
            # Convert world coordinates to grid cell coordinates
            grid_x = int((world_x - grid_origin_x) / resolution)
            grid_y = int((world_y - grid_origin_y) / resolution)
            
            # Skip if out of grid bounds
            if grid_x < 0 or grid_x >= grid_cols or grid_y < 0 or grid_y >= grid_rows:
                continue
            
            # Update log-odds for occupied cell (endpoint)
            if True:  # temporal_filtering
                log_odds_grid[grid_y, grid_x] = (1 - current_alpha) * log_odds_grid[grid_y, grid_x] + current_alpha * log_odds_occupied
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
            
            # Increment observation count
            observation_count_grid[grid_y, grid_x] += 1
    
    cpu_time = time.time() - cpu_start
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # Only benchmark CUDA if available
    if CUDA_AVAILABLE:
        print("\nBenchmarking CUDA implementation...")
        try:
            # Separate memory transfer time from kernel execution time
            copy_start = time.time()
            # Create and compile CUDA kernel
            @cuda.jit
            def transform_points_kernel(depth_image, rotation_matrix, camera_pos, 
                                       width, height, step, tan_fov_h, tan_fov_v,
                                       min_depth, max_depth, grid_origin_x, grid_origin_y,
                                       resolution, grid_cols, grid_rows,
                                       log_odds_grid, cell_height_grid, observation_count_grid,
                                       log_odds_occupied, log_odds_min, log_odds_max,
                                       current_alpha):
                # Get thread indices
                tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
                ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
                
                # Check if within image bounds and step size
                if tx < width and ty < height and tx % step == 0 and ty % step == 0:
                    # Get depth value
                    depth = depth_image[ty, tx]
                    
                    # Skip invalid depth values
                    if not math.isfinite(depth) or depth < min_depth or depth > max_depth:
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
                    
                    # Convert world coordinates to grid cell coordinates
                    grid_x = int((world_x - grid_origin_x) / resolution)
                    grid_y = int((world_y - grid_origin_y) / resolution)
                    
                    # Skip if out of grid bounds
                    if grid_x < 0 or grid_x >= grid_cols or grid_y < 0 or grid_y >= grid_rows:
                        return
                    
                    # Update log-odds for occupied cell (endpoint)
                    old_val = log_odds_grid[grid_y, grid_x]
                    new_val = (1 - current_alpha) * old_val + current_alpha * log_odds_occupied
                    
                    # Ensure log-odds value stays within bounds
                    new_val = max(log_odds_min, min(log_odds_max, new_val))
                    log_odds_grid[grid_y, grid_x] = new_val
                    
                    # Update height value
                    if cell_height_grid[grid_y, grid_x] == 0:
                        cell_height_grid[grid_y, grid_x] = world_z
                    else:
                        # Average with previous height
                        cell_height_grid[grid_y, grid_x] = 0.7 * cell_height_grid[grid_y, grid_x] + 0.3 * world_z
                    
                    # Increment observation count - use atomic add to prevent race conditions
                    cuda.atomic.add(observation_count_grid, (grid_y, grid_x), 1)
            
            # Copy data to GPU
            d_depth_image = cuda.to_device(depth_image)
            d_rotation_matrix = cuda.to_device(rotation_matrix)
            d_camera_pos = cuda.to_device(camera_pos)
            d_log_odds_grid = cuda.to_device(log_odds_grid)
            d_cell_height_grid = cuda.to_device(cell_height_grid)
            d_observation_count_grid = cuda.to_device(observation_count_grid)
            
            copy_time = time.time() - copy_start
            print(f"GPU Memory Transfer Time (to device): {copy_time:.4f} seconds")
            
            # Set up grid and block dimensions for kernel launch
            threads_per_block = (32, 32)  # Increased to 32x32
            blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            kernel_start = time.time()
            
            # Launch kernel
            transform_points_kernel[blocks_per_grid, threads_per_block](
                d_depth_image, d_rotation_matrix, d_camera_pos,
                width, height, step, tan_fov_h, tan_fov_v,
                min_depth, max_depth, grid_origin_x, grid_origin_y,
                resolution, grid_cols, grid_rows,
                d_log_odds_grid, d_cell_height_grid, d_observation_count_grid,
                log_odds_occupied, log_odds_min, log_odds_max,
                current_alpha
            )
            
            # Synchronize to ensure all operations are complete
            cuda.synchronize()
            
            kernel_time = time.time() - kernel_start
            print(f"GPU Kernel Execution Time: {kernel_time:.4f} seconds")
            
            # Copy results back
            copy_back_start = time.time()
            result_log_odds = d_log_odds_grid.copy_to_host()
            result_cell_height = d_cell_height_grid.copy_to_host()
            result_observation_count = d_observation_count_grid.copy_to_host()
            copy_back_time = time.time() - copy_back_start
            
            print(f"GPU Memory Transfer Time (back to host): {copy_back_time:.4f} seconds")
            
            total_cuda_time = copy_time + kernel_time + copy_back_time
            print(f"Total CUDA Time: {total_cuda_time:.4f} seconds")
            
            # Calculate speedup (overall and kernel-only)
            total_speedup = cpu_time / total_cuda_time
            kernel_speedup = cpu_time / kernel_time 
            print(f"\nCUDA Overall Speedup: {total_speedup:.2f}x faster than CPU")
            print(f"CUDA Kernel-only Speedup: {kernel_speedup:.2f}x faster than CPU")
            print(f"(Kernel-only excludes memory transfer overhead)")
            
            # Check for correctness by comparing non-zero cells
            cpu_points = np.sum(observation_count_grid > 0)
            cuda_points = np.sum(result_observation_count > 0)
            print(f"\nNumber of processed points - CPU: {cpu_points}, CUDA: {cuda_points}")
            
        except Exception as e:
            print(f"❌ CUDA benchmark failed: {e}")
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    run_benchmark()
