#!/bin/bash

# ZED Optimized Occupancy Grid Launch Script
# This script builds and launches a ZED camera with optimized occupancy grid
# that responds immediately to camera movements

# Change to the workspace directory
cd /home/x4/ocupency_grid

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Build the package
echo "Building the package with all optimizations..."
colcon build --symlink-install --packages-select zed_occupancy_grid

# Source the local setup to use the newly built package
source install/setup.bash

# Set display for GUI applications
export DISPLAY=:1

# Set enhanced ROS logging level for better debugging
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message} ({function_name}:{line_number})"
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export RCUTILS_LOGGING_MIN_SEVERITY=DEBUG

# Echo setup instructions
echo "========================================================"
echo "ZED OPTIMIZED OCCUPANCY GRID"
echo "This script launches an optimized version of the occupancy"
echo "grid that is highly responsive to camera movements."
echo ""
echo "IMPROVEMENTS:"
echo "- Ultra-sensitive motion detection"
echo "- High-frequency transform updates"
echo "- Continuous grid updates regardless of motion"
echo "- Optimized camera parameters"
echo "========================================================"
echo "Starting ROS nodes..."

# Launch the ZED camera with occupancy grid using optimized settings
ros2 launch zed_occupancy_grid zed_occupancy_grid.launch.py \
  camera_model:=zed2i \
  resolution:=0.05 \
  grid_width:=15.0 \
  grid_height:=15.0 \
  min_depth:=0.3 \
  max_depth:=15.0 \
  position_change_threshold:=0.0000001 \
  rotation_change_threshold:=0.0000001
