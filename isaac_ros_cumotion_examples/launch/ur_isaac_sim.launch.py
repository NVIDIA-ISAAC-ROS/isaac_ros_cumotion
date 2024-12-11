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
# Copyright (c) 2008-2013, Willow Garage, Inc. All rights reserved.
# Copyright (c) 2015-2023, PickNik, LLC. All rights reserved.
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


"""
This launch file is a minimal representative example on interfacing a UR robot with Isaac Sim.

It runs cuMotion planning calls using the MoveIt2 interface.
"""


import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import xacro
import yaml


def get_robot_description_contents(
    ur_type: str,
    dump_to_file: bool = False,
    output_file: str = None,
) -> str:
    """
    Get robot description contents and optionally dump content to file.

    Args
    ----
        asset_name (str): The asset name for robot description
        ur_type (str): UR Type
        use_sim_time (bool): Use sim time for isaac sim platform
        dump_to_file (bool, optional): Dumps xml to file. Defaults to False.
        output_file (str, optional): Output file path if dumps output is True. Defaults to None.

    Returns
    -------
        str: XMl contents of robot model

    """
    # Update the file extension and path as needed
    urdf_xacro_file = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'),
        'ur_config',
        'ur.urdf.xacro',
    )

    # Process the .xacro file to convert it to a URDF string
    xacro_processed = xacro.process_file(
        urdf_xacro_file,
        mappings={
            'ur_type': ur_type,
            'name': f'{ur_type}_robot',
            'use_fake_hardware': 'true',
            'generate_ros2_control_tag': 'false',
        },
    )
    robot_description = xacro_processed.toxml()

    if dump_to_file and output_file:
        with open(output_file, 'w') as file:
            file.write(robot_description)

    return robot_description


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[Node]:
    """
    Launch the group of nodes together in 1 process.

    Args
    ----
        context (LaunchContext): Context of launch file

    Returns
    -------
        List[Node]: List of nodes

    """
    ur_type = str(context.perform_substitution(LaunchConfiguration('ur_type')))

    moveit_config = (
        MoveItConfigsBuilder(ur_type, package_name='ur_moveit_config')
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory('isaac_ros_cumotion_examples'),
                'ur_config',
                'ur.urdf.xacro',
            ),
            mappings={'ur_type': ur_type},
        )
        .robot_description_semantic(
            file_path='srdf/ur.srdf.xacro', mappings={'name': 'ur'}
        )
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .trajectory_execution(file_path='config/controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )

    # Add cuMotion to list of planning pipelines.
    cumotion_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml',
    )
    with open(cumotion_config_file_path) as cumotion_config_file:
        cumotion_config = yaml.safe_load(cumotion_config_file)
    moveit_config.planning_pipelines['planning_pipelines'].insert(
        0, 'isaac_ros_cumotion'
    )
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'

    # The workarounds below are based on the following ur_moveit_config and ur_description versions
    # ur_moveit_config: github.com/UniversalRobots/Universal_Robots_ROS2_Driver/tree/2.2.14
    # ur_description: github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/2.1.5

    moveit_config.trajectory_execution = {
        'moveit_simple_controller_manager': moveit_config.trajectory_execution
    }
    del moveit_config.trajectory_execution['moveit_simple_controller_manager'][
        'moveit_manage_controllers'
    ]
    moveit_config.trajectory_execution['moveit_manage_controllers'] = True
    moveit_config.trajectory_execution['trajectory_execution'] = {
        'allowed_start_tolerance': 0.01
    }
    moveit_config.trajectory_execution['moveit_controller_manager'] = (
        'moveit_simple_controller_manager/MoveItSimpleControllerManager'
    )

    moveit_config.robot_description_kinematics['robot_description_kinematics'][
        'ur_manipulator'] = moveit_config.robot_description_kinematics[
            'robot_description_kinematics']['/**'][
                'ros__parameters']['robot_description_kinematics']['ur_manipulator']
    del moveit_config.robot_description_kinematics['robot_description_kinematics']['/**']

    moveit_config.joint_limits['robot_description_planning'] = xacro.load_yaml(
        os.path.join(
            get_package_share_directory(
                'ur_description'), 'config', ur_type, 'joint_limits.yaml',
        )
    )
    moveit_config.moveit_cpp.update({'use_sim_time': True})

    # Add limits from ur_moveit_config joint_limits.yaml to limits from ur_description
    for joint in moveit_config.joint_limits['robot_description_planning']['joint_limits']:
        moveit_config.joint_limits['robot_description_planning']['joint_limits'][joint][
            'has_acceleration_limits'] = True
        moveit_config.joint_limits['robot_description_planning']['joint_limits'][joint][
            'max_acceleration'] = 5.0

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
        'ur_moveit_config.rviz',
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
            {'use_sim_time': True},
        ],
    )

    # Static TF
    world2robot_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='log',
        arguments=['--frame-id', 'world', '--child-frame-id', 'base_link'],
        parameters=[{'use_sim_time': True}],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'),
        'ur_config',
        'ros2_controllers.yaml',
    )
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[ros2_controllers_path, {'use_sim_time': True}],
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

    scaled_joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['scaled_joint_trajectory_controller',
                   '-c', '/controller_manager'],
    )

    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '-c', '/controller_manager'],
    )
    urdf_path = '/tmp/collated_ur_urdf.urdf'
    # Define robot state publisher
    robot_description = get_robot_description_contents(
        ur_type=ur_type,
        dump_to_file=True,
        output_file=urdf_path,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description, 'use_sim_time': True}],
        remappings=[('/joint_states', '/isaac_joint_states')],
    )

    # Add cumotion planner node
    cumotion_planner_node = Node(
        name='cumotion_planner',
        package='isaac_ros_cumotion',
        namespace='',
        executable='cumotion_planner_node',
        parameters=[
            {
                'robot': 'ur10e.xrdf',
                'urdf_path': urdf_path
            }
        ],
        output='screen',
    )

    return [
        rviz_node,
        robot_state_publisher,
        world2robot_tf_node,
        move_group_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        scaled_joint_trajectory_controller_spawner,
        joint_trajectory_controller_spawner,
        cumotion_planner_node
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'ur_type',
            default_value='ur10e',
            description='UR robot type',
            choices=[
                'ur3',
                'ur3e',
                'ur5',
                'ur5e',
                'ur10',
                'ur10e',
                'ur16e',
                'ur20',
                'ur30',
            ],
        )
    ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
