#!/bin/bash

# Run the ZED camera node for ZED2i camera
# This script launches just the ZED camera node

# Make the script executable
chmod +x $0

# Set environment variables for optimized performance
export ZED_VERBOSE=0  # Disable verbose logging from ZED SDK
export ZED_SDK_VERBOSE=0  # Disable ZED SDK verbose logging
export DISPLAY=:1  # Set display for GUI applications

echo "Starting ZED camera node..."

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Find the ZED wrapper package
ZED_WRAPPER_PATH=~/zed-ros2-examples
if [ -d "$ZED_WRAPPER_PATH" ]; then
    echo "Found ZED ROS2 examples at $ZED_WRAPPER_PATH"
    
    # Build and source the ZED wrapper workspace if needed
    if [ ! -d "$ZED_WRAPPER_PATH/install" ]; then
        echo "Building ZED ROS2 wrapper..."
        cd $ZED_WRAPPER_PATH
        colcon build --symlink-install
    fi
    
    # Source the ZED wrapper
    source $ZED_WRAPPER_PATH/install/setup.bash
    
    # Launch the ZED camera with optimized parameters for occupancy grid
    echo "Launching ZED camera..."
    ros2 launch zed_wrapper zed_camera.launch.py \
        camera_model:=zed2i \
        general.pub_frame_rate:=10.0 \
        depth.depth_mode:=NEURAL_LIGHT \
        depth.depth_stabilization:=100 \
        depth.min_depth:=0.3 \
        depth.max_depth:=10.0 \
        pos_tracking.pos_tracking_enabled:=true \
        pos_tracking.area_memory:=true \
        pos_tracking.imu_fusion:=true
else
    echo "ERROR: Could not find ZED ROS2 examples at $ZED_WRAPPER_PATH"
    echo "Please check if the ZED ROS2 examples are installed at ~/zed-ros2-examples"
    echo "If they are installed elsewhere, please update this script with the correct path"
    exit 1
fi
