# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# This launch file was originally derived from
# https://github.com/ros-planning/moveit2_tutorials/blob/efef1d3/doc/how_to_guides/isaac_panda/launch/isaac_demo.launch.py  # noqa
#
# BSD 3-Clause License
#
# Copyright (c) 2008-2013, Willow Garage, Inc.
# Copyright (c) 2015-2023, PickNik, LLC.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

import yaml


def generate_launch_description():

    # Command-line arguments
    ros2_control_hardware_type = DeclareLaunchArgument(
        'ros2_control_hardware_type',
        default_value='isaac',
        description=(
            'ROS2 control hardware interface type to use for the launch file -- '
            'possible values: [mock_components, isaac]'
        )
    )

    moveit_config = (
        MoveItConfigsBuilder('moveit_resources_panda')
        .robot_description(
            file_path='config/panda.urdf.xacro',
            mappings={
                'ros2_control_hardware_type': LaunchConfiguration(
                    'ros2_control_hardware_type'
                )
            },
        )
        .robot_description_semantic(file_path='config/panda.srdf')
        .trajectory_execution(file_path='config/gripper_moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'])
        .to_moveit_configs()
    )

    # Add cuMotion to list of planning pipelines.
    cumotion_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml'
    )
    with open(cumotion_config_file_path) as cumotion_config_file:
        cumotion_config = yaml.safe_load(cumotion_config_file)
    moveit_config.planning_pipelines['planning_pipelines'].append('isaac_ros_cumotion')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config

    # The current Franka asset in Isaac Sim 2023.1.1 tends to drift slightly from commanded joint
    # positions, which prevents trajectory execution if the drift exceeds `allowed_start_tolerance`
    # for any joint; the default tolerance is 0.01 radians.  This is more likely to occur if the
    # robot hasn't fully settled when the trajectory is computed or if significant time has
    # elapsed between trajectory computation and execution. For this simulation use case,
    # there's little harm in disabling this check by setting `allowed_start_tolerance` to 0.
    moveit_config.trajectory_execution['trajectory_execution']['allowed_start_tolerance'] = 0.0

    # Start the actual move_group node/action server
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict()],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    # RViz
    rviz_config_file = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'),
        'rviz',
        'franka_moveit_config.rviz',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Static TF
    world2robot_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=['--frame-id', 'world', '--child-frame-id', 'panda_link0'],
    )
    hand2camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=[
            '0.04',
            '0.0',
            '0.04',
            '0.0',
            '0.0',
            '0.0',
            'panda_hand',
            'sim_camera',
        ],
    )

    # Publish TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[moveit_config.robot_description],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory('moveit_resources_panda_moveit_config'),
        'config',
        'ros2_controllers.yaml',
    )
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[ros2_controllers_path],
        remappings=[
            ('/controller_manager/robot_description', '/robot_description'),
        ],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
        ],
    )

    panda_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_arm_controller', '-c', '/controller_manager'],
    )

    panda_hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_hand_controller', '-c', '/controller_manager'],
    )

    return LaunchDescription(
        [
            ros2_control_hardware_type,
            rviz_node,
            world2robot_tf_node,
            hand2camera_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
        ]
    )
