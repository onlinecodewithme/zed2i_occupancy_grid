#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Get the launch directory
    zed_occupancy_grid_dir = get_package_share_directory('zed_occupancy_grid')
    
    # Define arguments
    camera_model = LaunchConfiguration('camera_model', default='zed2i')
    resolution = LaunchConfiguration('resolution', default='0.05')
    grid_width = LaunchConfiguration('grid_width', default='10.0')
    grid_height = LaunchConfiguration('grid_height', default='10.0')
    min_depth = LaunchConfiguration('min_depth', default='0.5')
    max_depth = LaunchConfiguration('max_depth', default='20.0')
    
    # Declare launch arguments
    declare_camera_model_cmd = DeclareLaunchArgument(
        'camera_model',
        default_value='zed2i',
        description='Camera model (zed, zed2, zed2i, zedm)')
    
    declare_resolution_cmd = DeclareLaunchArgument(
        'resolution',
        default_value='0.05',
        description='Grid resolution in meters')
    
    declare_grid_width_cmd = DeclareLaunchArgument(
        'grid_width',
        default_value='10.0',
        description='Grid width in meters')
    
    declare_grid_height_cmd = DeclareLaunchArgument(
        'grid_height',
        default_value='10.0',
        description='Grid height in meters')
    
    declare_min_depth_cmd = DeclareLaunchArgument(
        'min_depth',
        default_value='0.5',
        description='Minimum depth considered (m)')
    
    declare_max_depth_cmd = DeclareLaunchArgument(
        'max_depth',
        default_value='20.0',
        description='Maximum depth considered (m)')
    
    # Get the absolute path to the ZED wrapper launch file
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.dirname(launch_dir)
    workspace_dir = os.path.dirname(os.path.dirname(pkg_dir))
    zed_wrapper_dir = os.path.join(workspace_dir, 'src', 'zed-ros2-wrapper', 'zed_wrapper')
    
    # Define the ZED wrapper common config file
    common_config_file = os.path.join(zed_wrapper_dir, 'config', 'common_stereo.yaml')
    
    # Create the ZED camera launch with enhanced parameters
    camera_params = {
        'camera_model': camera_model,
        'general.camera_model': camera_model,
        
        # Depth settings
        'depth.depth_mode': 'NEURAL_LIGHT',  # High quality depth mode
        'depth.min_depth': min_depth,
        'depth.max_depth': max_depth,
        'depth.depth_stabilization': '100',  # Maximum stability
        'depth.point_cloud_freq': '15.0',
        
        # Positional tracking settings
        'pos_tracking.pos_tracking_enabled': 'true',
        'pos_tracking.area_memory': 'true',
        'pos_tracking.imu_fusion': 'true',
        'pos_tracking.floor_alignment': 'true',
        'pos_tracking.depth_min_range': '0.3',
        'pos_tracking.reset_odom_with_loop_closure': 'true',
        
        # Mapping settings
        'mapping.mapping_enabled': 'true',
        'mapping.resolution': resolution,
        'mapping.max_mapping_range': '20.0',
        'mapping.fused_pointcloud_freq': '2.0',
    }

    # Create the ZED camera launch description
    zed_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(zed_wrapper_dir, 'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments=camera_params.items()
    )
    
    # Define our occupancy grid node
    occupancy_grid_node = Node(
        package='zed_occupancy_grid',
        executable='zed_occupancy_grid_node',
        name='zed_occupancy_grid_node',
        output='screen',
        parameters=[{
            'camera_frame': 'zed_left_camera_frame',  # Fixed frame name to match actual ZED camera frame
            'depth_topic': '/zed/zed_node/depth/depth_registered',  # Fixed topic path to match actual ZED camera output
            'min_depth': min_depth,
            'max_depth': max_depth,
            'resolution': resolution,  # Controls grid cell size - smaller values give more detail
            'grid_width': 15.0,  # Increased grid size for better coverage
            'grid_height': 15.0  # Increased grid size for better coverage
        }]
    )
    
    # Define RViz2 node for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(zed_occupancy_grid_dir, 'config', 'occupancy_grid.rviz')],
        output='screen'
    )
    
    # Define the TF setup node to broadcast necessary transforms
    tf_setup_node = Node(
        package='zed_occupancy_grid',
        executable='zed_tf_setup',
        name='zed_tf_setup',
        output='screen'
    )
    
    # Create the launch description and add actions
    ld = LaunchDescription()
    
    # Add all the declared arguments
    ld.add_action(declare_camera_model_cmd)
    ld.add_action(declare_resolution_cmd)
    ld.add_action(declare_grid_width_cmd)
    ld.add_action(declare_grid_height_cmd)
    ld.add_action(declare_min_depth_cmd)
    ld.add_action(declare_max_depth_cmd)
    
    # Add the ZED camera launch to our launch description
    ld.add_action(zed_camera_launch)
    
    # Add the TF setup node
    ld.add_action(tf_setup_node)
    
    # Add our occupancy grid node
    ld.add_action(occupancy_grid_node)
    
    # Add RViz2 node (optional, can be commented out if not needed)
    ld.add_action(rviz_node)
    
    return ld
