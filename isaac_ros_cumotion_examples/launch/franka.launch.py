# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# To avoid code duplication, we patch and then execute the Franka demo launch file provided by
# the moveit2_tutorials package.

from os import path

from ament_index_python.packages import get_package_share_directory

import yaml


def augment_moveit_config(moveit_config):
    """Add cuMotion and its config to the planning_pipelines dict of a MoveItConfigs object."""
    config_file_path = path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml'
    )
    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)
    moveit_config.planning_pipelines['planning_pipelines'].append('isaac_ros_cumotion')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'


def generate_launch_description():
    franka_demo_launch_file = path.join(
        get_package_share_directory('moveit2_tutorials'),
        'launch',
        'demo.launch.py'
    )
    lf = open(franka_demo_launch_file).read()

    # Rename generate_launch_description() in base launch file.
    lf = lf.replace('generate_launch_description', 'generate_base_launch_description')

    # The standard way to make isaac_ros_cumotion_planning.yaml available to MoveIt would be to
    # copy the file into the config/ directory within a given robot's moveit_config package.
    # It would then suffice to add "isaac_ros_cumotion" to the list of planning_pipelines in the
    # MoveItConfigsBuilder, e.g., via the following substitution.
    #
    #     lf = lf.replace('"ompl"', '"isaac_ros_cumotion", "ompl"')
    #
    # Here we avoid adding the file to the moveit_resources_panda package by loading the file
    # manually and augmenting the MoveItConfigs object after it's built.
    lf = lf.replace(
        'run_move_group_node =',
        'augment_moveit_config(moveit_config)\n    run_move_group_node ='
    )

    # Execute modified launch file.
    exec(lf, globals())

    return generate_base_launch_description()  # noqa: F821
