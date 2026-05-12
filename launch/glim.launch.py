#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    glim_ros_dir = get_package_share_directory('glim_ros')

    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=os.path.join(glim_ros_dir, 'config'),
        description='Path to GLIM config folder')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use /clock (set true when replaying bags with --clock)')

    glim_node = Node(
        package='glim_ros',
        executable='glim_rosnode',
        name='glim_ros',
        output='screen',
        emulate_tty=True,
        additional_env={'__NV_PRIME_RENDER_OFFLOAD': '0'},
        parameters=[
            {'config_path': LaunchConfiguration('config_path')},
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )

    return LaunchDescription([
        config_path_arg,
        use_sim_time_arg,
        glim_node,
    ])
