#!/bin/bash

# Test script for the occupancy grid node with simulated camera movement
# This script creates the necessary transforms and publishes test depth images

# Make the script executable
chmod +x $0

# Set display for GUI applications
export DISPLAY=:1

echo "Starting the ZED occupancy grid test..."

# Source ROS 2
source /opt/ros/humble/setup.bash
source ~/ocupency_grid/install/setup.bash

# Create static transforms for testing
echo "Creating static transforms for testing..."
(
    # Kill any existing transform publishers
    pkill -f static_transform_publisher || true
    sleep 1
    
    # Run transform publishers in background
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
    TF_PID1=$!
    
    # Wait a moment for the first transform to be published
    sleep 0.5
    
    # Create a connected TF tree that simulates movement of the camera
    echo "Creating connected TF tree with moving camera..."
    (
        # This script will move the camera in a simple linear pattern
        for i in {1..100}; do
            # Create simple moving positions
            X=$(echo "scale=3; $i / 20" | bc)
            Y="0.0"  # Simple straight-line movement
            Z="0.5"
            
            # First: map to odom (static)
            ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom --ros-args -r __node:=map_to_odom &
            PID1=$!
            
            # Second: odom to camera (moves)
            ros2 run tf2_ros static_transform_publisher $X $Y $Z 0 0 0 odom zed_camera_link --ros-args -r __node:=moving_camera_tf &
            PID2=$!
            
            # Third: camera to left camera (static)
            ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 zed_camera_link zed_left_camera_frame --ros-args -r __node:=camera_to_left &
            PID3=$!
            
            # Wait a moment
            sleep 0.5
            
            # Kill the previous publishers
            kill $PID1 $PID2 $PID3
            
            echo "Camera position: $X, $Y, $Z"
        done
    ) &
    MOVEMENT_PID=$!
    
    # Create transform from camera to left camera frame
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 zed_camera_link zed_left_camera_frame &
    TF_PID3=$!
    
    # Run the optimized grid node
    echo "Running the occupancy grid node..."
    ros2 run zed_occupancy_grid zed_occupancy_grid_node \
        --ros-args \
        -p resolution:=0.05 \
        -p grid_width:=15.0 \
        -p grid_height:=15.0 \
        -p min_depth:=0.3 \
        -p max_depth:=10.0 \
        -p position_change_threshold:=0.001 \
        -p rotation_change_threshold:=0.001 \
        -p camera_frame:=zed_left_camera_frame
    
    # Clean up when the grid node exits
    kill $TF_PID1 $TF_PID3 $MOVEMENT_PID
) 2>&1

echo ""
echo "Test complete."
echo "You can visualize the occupancy grid in RViz by running:"
echo "ros2 run rviz2 rviz2 -d ~/ocupency_grid/src/zed_occupancy_grid/config/occupancy_grid.rviz"
