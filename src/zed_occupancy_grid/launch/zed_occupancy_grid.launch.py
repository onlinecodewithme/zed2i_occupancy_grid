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
        default_value='0.2',
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
        
        # QoS settings to fix point cloud compatibility
        'general.pub_frame_rate': '10.0',    # Lower rate for stability
        'general.qos_depth': '10',           # Increase queue size
        'general.qos_history': '1',          # 1 = KEEP_LAST
        'general.qos_reliability': '1',      # 1 = RELIABLE
        'general.qos_durability': '1',       # 1 = TRANSIENT_LOCAL (work with RViz)
        
        # Depth settings - optimized for reducing blinking
        'depth.depth_mode': 'NEURAL_LIGHT',  # High quality depth mode
        'depth.min_depth': min_depth,
        'depth.max_depth': max_depth,
        'depth.depth_stabilization': '100',  # Maximum stability
        'depth.point_cloud_freq': '5.0',     # Reduced frequency to prevent overwhelming RViz
        
        # Positional tracking settings
        'pos_tracking.pos_tracking_enabled': 'true',
        'pos_tracking.area_memory': 'true',
        'pos_tracking.imu_fusion': 'true',
        'pos_tracking.floor_alignment': 'true',
        'pos_tracking.depth_min_range': '0.3',
        'pos_tracking.reset_odom_with_loop_closure': 'true',
        # Enable the built-in loop closure detection
        'pos_tracking.slam_enabled': 'true',
        'pos_tracking.slam_mode': 'MEDIUM',  # MEDIUM is a good balance
        'pos_tracking.loop_closure_enabled': 'true',  # Explicitly enable loop closure
        
        # Mapping settings
        'mapping.mapping_enabled': 'true',
        'mapping.resolution': resolution,
        'mapping.max_mapping_range': '20.0',
        'mapping.fused_pointcloud_freq': '2.0',
    }
    # camera_params = {
    #     'camera_model': camera_model,
    #     'general.camera_model': camera_model,
        
    #     # Depth settings for low light
    #     'depth.depth_mode': 'ULTRA',  # Try ULTRA instead of NEURAL_LIGHT for low light
    #     'depth.min_depth': min_depth,
    #     'depth.max_depth': '10.0',  # Reduce max depth for better accuracy in limited range
    #     'depth.depth_stabilization': '100',  # Maximum stability
    #     'depth.point_cloud_freq': '10.0',  # Slightly lower frequency for better quality
        
    #     # Camera control for low light
    #     'video.brightness': '4',  # Increase brightness (default is 4, range 0-8)
    #     'video.contrast': '4',  # Adjust contrast (default is 4, range 0-8)
    #     'video.gain': '100',  # Increase gain for low light (range 0-100)
    #     'video.exposure': '100',  # Increase exposure for low light (range 0-100)
    #     'video.whitebalance_auto': 'true',  # Auto white balance helps in varying lighting
        
    #     # Positional tracking settings
    #     'pos_tracking.pos_tracking_enabled': 'true',
    #     'pos_tracking.area_memory': 'true',
    #     'pos_tracking.imu_fusion': 'true',
    #     'pos_tracking.floor_alignment': 'true',
    #     'pos_tracking.depth_min_range': '0.4',  # Slightly higher min range for better reliability
    #     'pos_tracking.reset_odom_with_loop_closure': 'true',
    #     'pos_tracking.initial_world_transform': 'GRAVITY',  # Use gravity for initial alignment
    #     'pos_tracking.set_gravity_as_origin': 'true',
        
    #     # SLAM settings optimized for indoor
    #     'pos_tracking.slam_enabled': 'true',
    #     'pos_tracking.slam_mode': 'MEDIUM',  # MEDIUM quality is good for indoor
    #     'pos_tracking.loop_closure_enabled': 'true',
        
    #     # Mapping settings
    #     'mapping.mapping_enabled': 'true',
    #     'mapping.resolution': resolution,
    #     'mapping.max_mapping_range': '8.0',  # Reduced range for more accurate indoor mapping
    #     'mapping.fused_pointcloud_freq': '1.0',  # Lower frequency for better quality
    # }

    # Create the ZED camera launch description
    zed_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(zed_wrapper_dir, 'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments=camera_params.items()
    )
    
    # Define our occupancy grid node with optimized parameters for dynamic updates
    occupancy_grid_node = Node(
        package='zed_occupancy_grid',
        executable='zed_occupancy_grid_node',
        name='zed_occupancy_grid_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'camera_frame': 'zed_left_camera_frame',  # Fixed frame name to match actual ZED camera frame
            'depth_topic': '/zed/zed_node/depth/depth_registered',  # Match the actual camera model and namespace
            'map_frame': 'map',  # Set the correct map frame to align with ZED's SLAM
            'min_depth': min_depth,
            'max_depth': max_depth,
            'resolution': resolution,  # Controls grid cell size - smaller values give more detail
            'grid_width': 15.0,  # Increased grid size for better coverage
            'grid_height': 15.0,  # Increased grid size for better coverage
            # More responsive movement detection thresholds
            'position_change_threshold': 0.005,  # Smaller threshold to detect minor movements
            'rotation_change_threshold': 0.005,  # Smaller threshold for rotation detection
            'moving_alpha': 0.2,  # Less temporal filtering when moving (more responsive)
            'static_alpha': 0.6,  # Less temporal filtering even when static
            'min_observations': 1,  # Allow quicker updates
            'use_sim_time': False,
            # CUDA acceleration parameters
            'use_cuda': True,
            'cuda_step': 4,  # Process 1 out of every 4 pixels (for speed)
            'cuda_ray_step': 2  # Ray tracing step size on GPU
        }]
        # Removing problematic QoS overrides - we'll handle QoS in the node itself
    )
    
    # Define RViz2 node for visualization with optimized parameters
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(zed_occupancy_grid_dir, 'config', 'occupancy_grid.rviz')],
        output='screen',
        parameters=[{
            # Increase buffer sizes for RViz consumers
            'tf_buffer_cache_time_ms': 30000,  # 30 seconds buffer (increased from 10 seconds)
            'tf_buffer_size': 200,  # Larger buffer size (doubled from 100)
            'update_rate': 10.0,  # Further limit update rate to 10Hz (reduced from 15Hz)
            'use_sim_time': False,
            'qos_reliability': 1,  # 1 = RELIABLE 
            'qos_durability': 1,   # 1 = TRANSIENT_LOCAL
            'qos_depth': 20        # Larger queue depth
        }]
    )
    
    # Define the TF setup node to broadcast necessary transforms
    tf_setup_node = Node(
        package='zed_occupancy_grid',
        executable='zed_tf_setup',
        name='zed_tf_setup',
        output='screen',
        parameters=[{
            'publish_frequency': 10.0  # Lower frequency for static transforms
        }]
    )
    
    # Define the Loop Closure node for map correction
    loop_closure_node = Node(
        package='zed_occupancy_grid',
        executable='zed_loop_closure',
        name='zed_loop_closure_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'camera_frame': 'zed_left_camera_frame',
            'map_frame': 'map',
            'grid_topic': '/occupancy_grid',
            'pose_topic': '/zed/zed_node/pose',
            'depth_topic': '/zed/zed_node/depth/depth_registered'
        }]
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
    
    # Add the ZED camera launch description first to set up camera
    ld.add_action(zed_camera_launch)
    
    # Add the TF setup node to ensure transforms are published
    ld.add_action(tf_setup_node)
    
    # Add our occupancy grid node after camera and TF are set up
    ld.add_action(occupancy_grid_node)
    
    # Add loop closure node for map correction
    ld.add_action(loop_closure_node)
    
    # Add RViz2 node (optional, can be commented out if not needed)
    ld.add_action(rviz_node)
    
    return ld
