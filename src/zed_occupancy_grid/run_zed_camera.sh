#!/bin/bash

# ZED Camera Only Launch Script
# This script launches just the ZED camera without the occupancy grid
# Useful for testing and debugging

# Change to the workspace directory
cd /home/x4/ocupency_grid

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Source the local setup
source install/setup.bash

# Set display for GUI applications
export DISPLAY=:1

# Set ROS logging level
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"
export RCUTILS_COLORIZED_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1

# Print banner
echo "========================================================"
echo "ZED CAMERA ONLY"
echo ""
echo "This script launches just the ZED camera without the"
echo "occupancy grid. Use this for testing if the camera is"
echo "working properly."
echo ""
echo "NOTE: This script does not build any packages."
echo "========================================================"
echo "Starting ZED camera node..."

# Get the absolute path to the ZED wrapper launch file
ZED_WRAPPER_DIR="/home/x4/ocupency_grid/src/zed-ros2-wrapper/zed_wrapper"

# Launch just the ZED camera
ros2 launch ${ZED_WRAPPER_DIR}/launch/zed_camera.launch.py \
  camera_model:=zed2i \
  publish_tf:=true \
  publish_map_tf:=true \
  pos_tracking.area_memory:=true \
  pos_tracking.imu_fusion:=true
