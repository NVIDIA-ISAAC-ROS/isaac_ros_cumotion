# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Jazzy-compatible UR launch file with cuMotion integration.

This file adapts the Jazzy UR launch architecture to include cuMotion planning
and handle the additional parameters that were supported in Humble.
"""

import os
from pathlib import Path
from typing import List

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_context import LaunchContext
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder

import yaml


def load_yaml(package_name, file_path):
    """Load YAML file from package."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def cumotion_params():
    """Load cuMotion planning parameters."""
    config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml'
    )
    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    return config


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[Node]:
    """Launch setup function that properly resolves launch configurations."""
    # Resolve launch configurations using the context
    launch_rviz = context.perform_substitution(LaunchConfiguration('launch_rviz'))
    ur_type = context.perform_substitution(LaunchConfiguration('ur_type'))
    warehouse_sqlite_path = context.perform_substitution(
        LaunchConfiguration('warehouse_sqlite_path'))
    launch_servo = context.perform_substitution(LaunchConfiguration('launch_servo'))
    use_sim_time = context.perform_substitution(LaunchConfiguration('use_sim_time'))
    publish_robot_description_semantic = context.perform_substitution(
        LaunchConfiguration('publish_robot_description_semantic'))

    # Build MoveIt configuration
    moveit_config = (
        MoveItConfigsBuilder(robot_name='ur', package_name='ur_moveit_config')
        .robot_description_semantic(Path('srdf') / 'ur.srdf.xacro', {'name': ur_type})
        .to_moveit_configs()
    )

    # Add cuMotion to planning pipelines
    cumotion_config = cumotion_params()
    moveit_config.planning_pipelines['planning_pipelines'].insert(0, 'isaac_ros_cumotion')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'

    # Warehouse configuration
    warehouse_ros_config = {
        'warehouse_plugin': 'warehouse_ros_sqlite::DatabaseConnection',
        'warehouse_host': warehouse_sqlite_path,
    }

    # Wait for robot description
    wait_robot_description = Node(
        package='ur_robot_driver',
        executable='wait_for_robot_description',
        output='screen',
    )

    # Move group node with cuMotion integration
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            warehouse_ros_config,
            {
                'use_sim_time': True if use_sim_time == 'true' else False,
                'publish_robot_description_semantic': publish_robot_description_semantic,
            },
        ],
    )

    # Servo node (optional) - use resolved value for condition
    servo_yaml = load_yaml('ur_moveit_config', 'config/ur_servo.yaml')
    servo_params = {'moveit_servo': servo_yaml}
    servo_nodes = []
    if launch_servo == 'true':
        servo_nodes.append(Node(
            package='moveit_servo',
            executable='servo_node',
            parameters=[
                moveit_config.to_dict(),
                servo_params,
            ],
            output='screen',
        ))

    # RViz node (optional) - use resolved value for condition
    rviz_nodes = []
    if launch_rviz == 'true':
        rviz_config_file = PathJoinSubstitution(
            [FindPackageShare('ur_moveit_config'), 'config', 'moveit.rviz']
        )
        rviz_nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_moveit',
            output='log',
            arguments=['-d', rviz_config_file],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                warehouse_ros_config,
                {
                    'use_sim_time': True if use_sim_time == 'true' else False,
                },
            ],
        ))

    # Register event handler to start nodes after robot description is ready
    event_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_robot_description,
            on_exit=[move_group_node] + servo_nodes + rviz_nodes,
        )
    )

    return [wait_robot_description, event_handler]


def generate_launch_description():
    """Generate launch description with cuMotion integration."""
    launch_args = [
        DeclareLaunchArgument('launch_rviz', default_value='true', description='Launch RViz?'),
        DeclareLaunchArgument(
            'ur_type',
            description='Type/series of used UR robot.',
            choices=[
                'ur3',
                'ur3e',
                'ur5',
                'ur5e',
                'ur7e',
                'ur10',
                'ur10e',
                'ur12e',
                'ur16e',
                'ur15',
                'ur20',
                'ur30',
            ],
        ),
        DeclareLaunchArgument(
            'warehouse_sqlite_path',
            default_value=os.path.expanduser('~/.ros/warehouse_ros.sqlite'),
            description='Path where the warehouse database should be stored',
        ),
        DeclareLaunchArgument(
            'launch_servo',
            default_value='false',
            description='Launch Servo?',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Using or not time from simulation',
        ),
        DeclareLaunchArgument(
            'publish_robot_description_semantic',
            default_value='true',
            description='MoveGroup publishes robot description semantic',
        ),
        # Additional parameters for cuMotion integration
        DeclareLaunchArgument(
            'robot_ip',
            default_value='192.168.56.101',
            description='IP address of the UR robot',
        ),
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
