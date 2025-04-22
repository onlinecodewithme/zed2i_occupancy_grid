#!/bin/bash

# ZED Occupancy Grid Launch Script
# This script builds the package and launches the ZED camera with occupancy grid generation

# Change to the workspace directory
cd /home/x4/ocupency_grid

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Build the package
echo "Building the package..."
colcon build --symlink-install --packages-select zed_occupancy_grid

# Source the local setup to use the newly built package
source install/setup.bash

# Set display for GUI applications
export DISPLAY=:1

# Set CUDA environment variables for GPU acceleration
# Let the script auto-detect the CUDA driver path
export CUDA_VISIBLE_DEVICES=0
export NUMBA_CUDA_ENABLE_CUDASIM=0
export NUMBA_CUDA_ARRAY_INTERFACE_SYNC=1

# Set ROS logging level for better debug information
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1

# Launch the ZED camera with occupancy grid optimized for navigation
echo "Launching ZED Occupancy Grid with wall-optimized settings..."
echo "Loop closure and rotation handling are ENABLED"
ros2 launch zed_occupancy_grid zed_occupancy_grid.launch.py \
  camera_model:=zed2i \
  resolution:=0.03 \
  grid_width:=15.0 \
  grid_height:=15.0 \
  min_depth:=0.3 \
  max_depth:=15.0
