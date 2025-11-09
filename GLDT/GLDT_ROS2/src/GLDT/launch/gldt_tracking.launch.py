#!/usr/bin/env python3

"""
ROS2 Launch file for GLDT tracking system
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package path
    pkg_share = get_package_share_directory('gldt')
    
    # Declare launch arguments
    global_model_arg = DeclareLaunchArgument(
        'global_model',
        default_value='Weigjhts/global_int8.engine',
        description='Path to global model engine file'
    )
    
    local_model_arg = DeclareLaunchArgument(
        'local_model', 
        default_value='Weights/local_fp16.engine',
        description='Path to local model engine file'
    )
    
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value='videos',
        description='Path to video file or directory'
    )
    
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='Results',
        description='Output directory for results'
    )
    
    detection_mode_arg = DeclareLaunchArgument(
        'detection_mode',
        default_value='0',
        description='Detection mode: 0=global only, 1=global+local combined'
    )
    
    use_bytetrack_arg = DeclareLaunchArgument(
        'use_bytetrack',
        default_value='false',
        description='Use original ByteTrack algorithm (true) or enhanced tracker (false)'
    )
    
    enable_ros_arg = DeclareLaunchArgument(
        'enable_ros',
        default_value='true',
        description='Enable ROS2 publishing'
    )
    
    # Create GLDT node
    gldt_node = Node(
        package='gldt',
        executable='gldt_node',
        name='gldt_tracking_node',
        output='screen',
        parameters=[{
            'global_model': LaunchConfiguration('global_model'),
            'local_model': LaunchConfiguration('local_model'),
            'video_path': LaunchConfiguration('video_path'),
            'output_dir': LaunchConfiguration('output_dir'),
            'detection_mode': LaunchConfiguration('detection_mode'),
            'use_bytetrack': LaunchConfiguration('use_bytetrack'),
            'enable_ros_publishing': LaunchConfiguration('enable_ros'),
        }],
        arguments=[
            LaunchConfiguration('global_model'),
            LaunchConfiguration('local_model'),
            LaunchConfiguration('video_path'),
            LaunchConfiguration('output_dir')
        ],
        condition=IfCondition(LaunchConfiguration('enable_ros'))
    )
    
    # Create non-ROS mode executable file launch
    gldt_process = ExecuteProcess(
        cmd=[
            'bash', '-c',
            PythonExpression([
                '"',
                'export USE_ORIGINAL_BYTETRACK=', LaunchConfiguration('use_bytetrack'), ' && ',
                'export DETECTION_MODE=', LaunchConfiguration('detection_mode'), ' && ',
                'export ENABLE_ROS_PUBLISHING=', LaunchConfiguration('enable_ros'), ' && ',
                'exec ', os.path.join(pkg_share, '..', '..', '..', 'install', 'gldt', 'lib', 'gldt', 'gldt_node'),
                ' "', LaunchConfiguration('global_model'), '" "', LaunchConfiguration('local_model'), 
                '" "', LaunchConfiguration('video_path'), '" "', LaunchConfiguration('output_dir'), '"'
            ])
        ],
        output='screen',
        condition=IfCondition(PythonExpression([LaunchConfiguration('enable_ros'), ' == "false"']))
    )
    
    # Display launch information
    launch_info = LogInfo(
        msg=[
            'Starting GLDT Tracking System with parameters:\n',
            '  Global Model: ', LaunchConfiguration('global_model'), '\n',
            '  Local Model: ', LaunchConfiguration('local_model'), '\n', 
            '  Video Path: ', LaunchConfiguration('video_path'), '\n',
            '  Output Dir: ', LaunchConfiguration('output_dir'), '\n',
            '  Detection Mode: ', LaunchConfiguration('detection_mode'), '\n',
            '  Use ByteTrack: ', LaunchConfiguration('use_bytetrack'), '\n',
            '  Enable ROS: ', LaunchConfiguration('enable_ros'), '\n',
            '\nPublished Topics:\n',
            '  /gldt/tracking_result - Tracking results\n',
            '  /gldt/tracking_image - Visualization image\n'
        ]
    )
    
    return LaunchDescription([
        global_model_arg,
        local_model_arg,
        video_path_arg,
        output_dir_arg,
        detection_mode_arg,
        use_bytetrack_arg,
        enable_ros_arg,
        launch_info,
        gldt_node,
        gldt_process,
    ])

