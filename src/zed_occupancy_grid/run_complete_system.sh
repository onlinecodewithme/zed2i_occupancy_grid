#!/bin/bash

# ZED Complete System Launch Script
# This script builds and launches the complete ZED system
# with all optimizations for responsive occupancy grid mapping

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

# Set enhanced ROS logging level for better debugging
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export RCUTILS_LOGGING_MIN_SEVERITY=INFO

# Print banner
echo "========================================================"
echo "ZED COMPLETE SYSTEM WITH CONTINUOUS MAP UPDATES"
echo ""
echo "This script launches the complete ZED camera system with"
echo "optimized settings to ensure the occupancy grid updates"
echo "continuously as the camera moves around."
echo ""
echo "SYSTEM COMPONENTS:"
echo "- ZED Camera with optimized tracking parameters"
echo "- TF system with high-frequency updates (30Hz)"
echo "- Occupancy grid with ultra-sensitive motion detection"
echo "- Extra synthetic motion to force continuous updates"
echo "========================================================"
echo "Starting ROS nodes..."

# Launch the ZED camera with occupancy grid including all optimizations
ros2 launch zed_occupancy_grid zed_occupancy_grid.launch.py \
  camera_model:=zed2i \
  resolution:=0.05 \
  grid_width:=15.0 \
  grid_height:=15.0 \
  min_depth:=0.3 \
  max_depth:=15.0 \
  position_change_threshold:=0.0000001 \
  rotation_change_threshold:=0.0000001
