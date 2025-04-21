#!/bin/bash

# Script to run the ZED occupancy grid node with CUDA acceleration
# This script runs the occupancy grid node with CUDA acceleration enabled

echo "Starting ZED Occupancy Grid with CUDA Acceleration..."

# Set environment variables to enable CUDA
export NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export CUDA_VISIBLE_DEVICES=0

# Run the node with CUDA acceleration
ros2 run zed_occupancy_grid zed_occupancy_grid_node.py --ros-args -p use_cuda:=true -p cuda_step:=4 -p cuda_ray_step:=2

echo "ZED Occupancy Grid with CUDA Acceleration terminated."
