# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def get_robot_description():
    ur_type = LaunchConfiguration('ur_type')
    robot_ip = LaunchConfiguration('robot_ip')

    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare('ur_description'), 'config', ur_type, 'joint_limits.yaml']
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare('ur_description'), 'config', ur_type, 'default_kinematics.yaml']
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare('ur_description'), 'config', ur_type, 'physical_parameters.yaml']
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare('ur_description'), 'config', ur_type, 'visual_parameters.yaml']
    )
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
            PathJoinSubstitution([FindPackageShare('ur_description'), 'urdf', 'ur.urdf.xacro']),
            ' ', 'robot_ip:=', robot_ip,
            ' ', 'joint_limit_params:=', joint_limit_params,
            ' ', 'kinematics_params:=', kinematics_params,
            ' ', 'physical_params:=', physical_params,
            ' ', 'visual_params:=', visual_params,
            ' ', 'safety_limits:=true',
            ' ', 'safety_pos_margin:=0.15',
            ' ', 'safety_k_position:=20',
            ' ', 'name:=ur', ' ', 'ur_type:=', ur_type, ' ', 'prefix:=''',
        ]
    )

    robot_description = {'robot_description': robot_description_content}
    return robot_description


def get_robot_description_semantic():
    # MoveIt Configuration
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
            PathJoinSubstitution([FindPackageShare('ur_moveit_config'), 'srdf', 'ur.srdf.xacro']),
            ' ', 'name:=ur', ' ', 'prefix:=""',
        ]
    )
    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_content
    }
    return robot_description_semantic


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'ur_type',
            description='Type/series of used UR robot.',
            choices=['ur3', 'ur3e', 'ur5', 'ur5e', 'ur10', 'ur10e', 'ur16e', 'ur20'],
            default_value='ur5e',
        ),
        DeclareLaunchArgument(
            'robot_ip',
            description='IP address of the robot',
            default_value='192.56.1.2',
        ),

    ]
    moveit_kinematics_params = PathJoinSubstitution(
        [FindPackageShare('ur_moveit_config'), 'config', 'default_kinematics.yaml']
    )
    robot_description = get_robot_description()
    robot_description_semantic = get_robot_description_semantic()
    isaac_ros_moveit_goal_setter = Node(
        package='isaac_ros_moveit_goal_setter',
        executable='isaac_ros_moveit_goal_setter',
        name='isaac_ros_moveit_goal_setter',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            moveit_kinematics_params
        ],
    )

    return launch.LaunchDescription(launch_args + [isaac_ros_moveit_goal_setter])
