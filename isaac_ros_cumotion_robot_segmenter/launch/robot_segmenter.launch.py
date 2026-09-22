# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES',
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from ament_index_python.packages import get_package_share_directory
from isaac_ros_launch_utils.all_types import (
    ComposableNodeContainer, GroupAction, LoadComposableNodes
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
import yaml


def read_params(pkg_name, params_dir, params_file_name):
    params_file = os.path.join(
        get_package_share_directory(pkg_name), params_dir, params_file_name)
    return yaml.safe_load(open(params_file, 'r'))


def launch_args_from_params(pkg_name, params_dir, params_file_name,  prefix: str = None,
                            declare_launch_args: bool = True):
    launch_args = []
    launch_configs = {}
    params = read_params(pkg_name, params_dir, params_file_name)

    for param, value in params['/**']['ros__parameters'].items():
        if value is not None:
            arg_name = param if prefix is None else f'{prefix}.{param}'
            if declare_launch_args:
                launch_args.append(DeclareLaunchArgument(name=arg_name, default_value=str(value)))
            launch_configs[param] = LaunchConfiguration(arg_name)

    return launch_args, launch_configs


def launch_setup(context, *args, **kwargs):

    declare_launch_args = context.perform_substitution(
        LaunchConfiguration('declare_launch_args')) == 'true'
    _, launch_configs = launch_args_from_params(
        'isaac_ros_cumotion_robot_segmenter', 'params',
        'robot_segmentation_params.yaml', 'robot_segmenter',
        declare_launch_args=declare_launch_args)

    env_variables = dict(os.environ)

    enable_cuda_mps = context.perform_substitution(
        LaunchConfiguration('robot_segmenter.enable_cuda_mps')) == 'true'
    if enable_cuda_mps:
        env_variables.update({
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE':
                launch_configs['cuda_mps_active_thread_percentage'],
            'CUDA_MPS_PIPE_DIRECTORY': launch_configs['cuda_mps_pipe_directory'],
            'CUDA_MPS_CLIENT_PRIORITY': launch_configs['cuda_mps_client_priority']
        })

    robot_segmenter_nodes = []

    # Get the number of cameras from the launch configuration
    num_cameras = int(context.perform_substitution(
        LaunchConfiguration('robot_segmenter.num_cameras')))

    input_img_topics = context.perform_substitution(
            LaunchConfiguration('robot_segmenter.depth_image_topics'))
    input_camera_infos = context.perform_substitution(
        LaunchConfiguration('robot_segmenter.depth_camera_infos'))
    output_mask_topics = context.perform_substitution(
        LaunchConfiguration('robot_segmenter.robot_mask_publish_topics'))
    output_depth_topics = context.perform_substitution(
        LaunchConfiguration('robot_segmenter.world_depth_publish_topics'))
    input_qos = context.perform_substitution(
        LaunchConfiguration('robot_segmenter.input_qos'))
    output_qos = context.perform_substitution(
        LaunchConfiguration('robot_segmenter.output_qos'))

    def parse_list_arg(arg_string):
        # Remove brackets if present and split by comma
        clean_string = arg_string.strip('[]').strip()
        # Filter out empty strings and strip whitespace
        return [item.strip().strip("'\"") for item in clean_string.split(',') if item.strip()]

    list_input_img_topics = parse_list_arg(input_img_topics)
    list_input_camera_infos = parse_list_arg(input_camera_infos)
    list_output_mask_topics = parse_list_arg(output_mask_topics)
    list_output_depth_topics = parse_list_arg(output_depth_topics)

    for i in range(num_cameras):
        robot_segmenter_node = ComposableNode(
            name=f'robot_segmenter_{i+1}',
            package='isaac_ros_cumotion_robot_segmenter',
            plugin='nvidia::isaac_ros::manipulator::RobotSegmenter',
            parameters=[{
                    'input_qos': input_qos,
                    'output_qos': output_qos,
                    'input_qos_depth': 1,
                    'output_qos_depth': 1,
                    'enable_performance_logging': launch_configs['enable_performance_logging'],
                    'urdf_path': launch_configs['urdf_path'],
                    'xrdf_path': launch_configs['xrdf_path'],
                    'additional_buffer_distance': launch_configs['distance_threshold'],
                    'reload_robot_description_topic':
                        launch_configs['reload_robot_description_topic'],
                    'robot_description_service_name':
                        launch_configs['robot_description_service_name'],
                    'robot_base_frame': launch_configs['robot_base_frame'],
            }],
            remappings=[
                ('depth_image', list_input_img_topics[i]),
                ('camera_info_depth', list_input_camera_infos[i]),
                ('joint_states', launch_configs['joint_states_topic']),
                ('robot_mask', list_output_mask_topics[i]),
                ('robot_depth', list_output_depth_topics[i]),
            ]
        )

        robot_segmenter_nodes.append(robot_segmenter_node)

    container_name = str(context.perform_substitution(
        LaunchConfiguration('robot_segmenter.container_name'))).strip()
    if container_name:
        load_composable_nodes = LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=robot_segmenter_nodes,
        )
        final_launch = GroupAction(
            actions=[
                load_composable_nodes
            ],
        )
        return [final_launch]

    robot_segmenter_container = ComposableNodeContainer(
        name='robot_segmenter_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        env=env_variables,
        composable_node_descriptions=robot_segmenter_nodes,
        output='screen',
    )
    return [robot_segmenter_container]


def generate_launch_description():
    """Launch file to bring up robot segmenter node."""
    declare_launch_args = [
        DeclareLaunchArgument('declare_launch_args', default_value='false'),
    ]

    launch_args, launch_configs = launch_args_from_params(
        'isaac_ros_cumotion_robot_segmenter', 'params', 'robot_segmentation_params.yaml',
        'robot_segmenter')

    return LaunchDescription(
        launch_args + declare_launch_args + [OpaqueFunction(function=launch_setup)]
    )
