#!/bin/bash

# Script to run the ZED occupancy grid node with CUDA acceleration
# This script runs the occupancy grid node with CUDA acceleration enabled

echo "Starting ZED Occupancy Grid with CUDA Acceleration..."

# Set environment variables to enable CUDA
# Let the script auto-detect the CUDA driver path
export CUDA_VISIBLE_DEVICES=0

# Run the node with CUDA acceleration optimized for wall detection and navigation
ros2 run zed_occupancy_grid zed_occupancy_grid_node.py --ros-args -p use_cuda:=true -p cuda_step:=2 -p cuda_ray_step:=1 -p resolution:=0.03

echo "ZED Occupancy Grid with CUDA Acceleration terminated."
