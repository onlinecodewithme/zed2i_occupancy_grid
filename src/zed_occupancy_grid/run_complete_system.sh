#!/bin/bash

# Run the complete system: ZED camera + optimized occupancy grid
# This script directly runs the optimized occupancy grid to test it

# Make the script executable
chmod +x $0

# Set display for GUI applications
export DISPLAY=:1

echo "Starting the optimized occupancy grid..."

# Make sure the script is executable
chmod +x "$(dirname "$0")/run_optimized_grid.sh"

# Source ROS 2
source /opt/ros/humble/setup.bash
source ~/ocupency_grid/install/setup.bash

# Check if ZED node is already running
if ! ros2 topic list | grep -q "/zed/zed_node/depth/depth_registered"; then
    echo "WARNING: ZED camera node doesn't appear to be running."
    echo "In a separate terminal, you should run the ZED camera with:"
    echo "~/ocupency_grid/src/zed_occupancy_grid/run_zed_camera.sh"
    echo ""
    echo "Continuing with just the occupancy grid node..."
fi

# Directly run the optimized grid node - this is for testing
echo "Running the occupancy grid node..."
cd ~/ocupency_grid
source install/setup.bash

# Create static transforms for testing (in case ZED camera isn't running)
echo "Creating static transforms for testing..."
( 
    # Run transform publisher in background to create frames for testing
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom zed_camera_link &
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 zed_camera_link zed_left_camera_frame &
    
    # Run the node with performance-optimized parameters
    ros2 run zed_occupancy_grid zed_occupancy_grid_node \
        --ros-args \
        -p resolution:=0.05 \
        -p grid_width:=15.0 \
        -p grid_height:=15.0 \
        -p min_depth:=0.3 \
        -p max_depth:=10.0 \
        -p position_change_threshold:=0.001 \
        -p rotation_change_threshold:=0.001 \
        -p camera_frame:=zed_left_camera_frame \
        -p depth_topic:=/zed/zed_node/depth/depth_registered
) 2>&1

echo ""
echo "You can visualize the occupancy grid in RViz:"
echo "ros2 run rviz2 rviz2 -d ~/ocupency_grid/src/zed_occupancy_grid/config/occupancy_grid.rviz"
