#!/bin/bash

# Run the optimized occupancy grid for ZED2i camera
# This script runs the optimized occupancy grid with the ZED2i camera

# Make the script executable
chmod +x $0

# Set environment variables for optimized performance
export ZED_VERBOSE=0  # Disable verbose logging from ZED SDK
export ZED_SDK_VERBOSE=0  # Disable ZED SDK verbose logging
export DISPLAY=:1  # Set display for GUI applications

echo "Starting optimized ZED occupancy grid..."

# Source ROS2 setup
source /opt/ros/humble/setup.bash
source ~/ocupency_grid/install/setup.bash

# First make sure the package is built with latest changes
echo "Building the package..."
cd ~/ocupency_grid
colcon build --packages-select zed_occupancy_grid

# Source the updated workspace
source ~/ocupency_grid/install/setup.bash

# Run just the occupancy grid node with optimized settings
# Without trying to also launch the ZED camera (which might already be running)
echo "Running optimized occupancy grid node..."
ros2 run zed_occupancy_grid zed_occupancy_grid_node \
    --ros-args \
    -p resolution:=0.05 \
    -p grid_width:=15.0 \
    -p grid_height:=15.0 \
    -p min_depth:=0.3 \
    -p max_depth:=10.0 \
    -p position_change_threshold:=0.005 \
    -p rotation_change_threshold:=0.005 \
    -p camera_frame:=zed_left_camera_frame \
    -p depth_topic:=/zed/zed_node/depth/depth_registered

echo "Occupancy grid node is running. Make sure the ZED camera node is also running."
echo "If not, run in another terminal: ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i"
