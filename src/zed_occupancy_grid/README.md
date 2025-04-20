# ZED Camera Occupancy Grid

A ROS2 package for generating 2D occupancy grid maps from ZED camera depth data. This package works with the ZED ROS2 wrapper and is designed for navigation and mapping applications on mobile robots.

## Overview

This package provides the following functionality:
- Converts ZED camera depth data to 2D occupancy grid format
- Integrates directly with the ZED SDK and ZED ROS2 wrapper
- Provides TF transforms necessary for navigation
- Includes RViz configuration for visualization
- Supports loop closure and floor alignment for improved mapping

## Prerequisites

- Ubuntu 22.04 with ROS2 Humble
- ZED SDK (v4.0 or later)
- ZED ROS2 Wrapper ([stereolabs/zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper))
- ZED2i Camera (compatible with ZED, ZED2, ZED Mini, etc.)
- Jetson Orin NX (or similar capable hardware)

## Installation

1. Clone this repository and the ZED ROS2 wrapper into your ROS2 workspace:
```bash
cd ~/your_ros2_workspace/src
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
# Clone this repository as well
```

2. Install dependencies:
```bash
cd ~/your_ros2_workspace
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the packages:
```bash
colcon build --symlink-install
source install/setup.bash
```

## Usage

The package includes a convenience script to launch everything:

```bash
./src/zed_occupancy_grid/run_occupancy_grid.sh
```

This script will:
1. Build the necessary packages
2. Configure environment variables
3. Launch the ZED camera node with optimal parameters for occupancy grid generation
4. Start the TF publisher node for proper coordinate transformations
5. Launch the occupancy grid node
6. Open RViz with a preconfigured view

### Configuration Parameters

The following parameters can be adjusted in the launch file or command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| camera_model | zed2i | ZED camera model (zed, zed2, zed2i, zedm) |
| resolution | 0.05 | Grid resolution in meters per cell |
| grid_width | 15.0 | Width of the occupancy grid in meters |
| grid_height | 15.0 | Height of the occupancy grid in meters |
| min_depth | 0.3 | Minimum depth to consider (meters) |
| max_depth | 20.0 | Maximum depth to consider (meters) |

Example of launching with custom parameters:

```bash
ros2 launch zed_occupancy_grid zed_occupancy_grid.launch.py camera_model:=zed2 resolution:=0.1 grid_width:=20.0
```

## Technical Details

### ZED Camera Configuration

We use the following key ZED parameters optimized for occupancy grid generation:

- **Depth Mode**: NEURAL_LIGHT - Provides high-quality depth data with neural depth sensing
- **Depth Stabilization**: 100 - Maximum stability for consistent depth measurements
- **Positional Tracking**: Enabled with area memory and loop closure
- **Floor Alignment**: Enabled to improve pose estimation
- **Point Cloud Frequency**: 15Hz - Balanced for real-time performance
- **Mapping Resolution**: Matches the occupancy grid resolution

### Occupancy Grid Generation

The occupancy grid node:
1. Subscribes to the depth images from the ZED camera
2. Converts depth points to 3D coordinates
3. Projects 3D points onto a 2D plane
4. Uses ray tracing to mark free and occupied space
5. Publishes the resulting grid as a standard nav_msgs/OccupancyGrid message

### Transform Frames

The package maintains the following TF tree:
- map → odom → base_link → camera_link → camera_center → camera_left_frame

## Troubleshooting

If you encounter issues with the occupancy grid not showing up in RViz:

1. Check if the ZED camera is properly connected and recognized:
```bash
lsusb | grep 2b03
```

2. Verify that the depth topic is being published:
```bash
ros2 topic echo /zed2i/zed_node/depth/depth_registered/header --once
```

3. Check the TF tree for proper connections:
```bash
ros2 run tf2_tools view_frames
```

4. Look for errors in the node output:
```bash
ros2 topic echo /rosout | grep -E "error|warn" --color
```

## Advanced Configuration

For more advanced configuration of the ZED camera parameters, refer to:

- [ZED Depth Settings](https://www.stereolabs.com/docs/depth-sensing/depth-settings)
- [ZED Confidence Filtering](https://www.stereolabs.com/docs/depth-sensing/confidence-filtering)
- [ZED Using Depth](https://www.stereolabs.com/docs/depth-sensing/using-depth)

## License

This package is licensed under the MIT License - see the LICENSE file for details.
