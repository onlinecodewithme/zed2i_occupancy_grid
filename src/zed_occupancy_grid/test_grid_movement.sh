#!/bin/bash

# ZED Occupancy Grid Movement Test Script
# This script launches the ZED camera with extremely sensitive movement detection

# Change to the workspace directory
cd /home/x4/ocupency_grid

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Source the local setup to use the newly built package
source install/setup.bash

# Set display for GUI applications
export DISPLAY=:1

# Set ROS logging level to get more debugging info
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export RCUTILS_LOGGING_MIN_SEVERITY=DEBUG

# Echo setup instructions
echo "========================================================"
echo "ZED OCCUPANCY GRID MOVEMENT TEST"
echo "This script uses hyper-sensitive movement detection to ensure"
echo "the occupancy grid updates when you move the camera."
echo "========================================================"
echo "Starting ROS nodes..."

# Launch the ZED camera with occupancy grid - using super sensitive settings
ros2 launch zed_occupancy_grid zed_occupancy_grid.launch.py \
  camera_model:=zed2i \
  resolution:=0.05 \
  grid_width:=15.0 \
  grid_height:=15.0 \
  min_depth:=0.3 \
  max_depth:=20.0 \
  position_change_threshold:=0.000001 \
  rotation_change_threshold:=0.000001
