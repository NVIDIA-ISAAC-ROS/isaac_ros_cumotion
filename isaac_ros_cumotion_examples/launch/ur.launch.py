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

# To avoid code duplication, we patch and then execute the launch file provided by the
# ur_moveit_config package.

from os import path

from ament_index_python.packages import get_package_share_directory

import yaml


def cumotion_params():
    # The standard way to make isaac_ros_cumotion_planning.yaml available to MoveIt would be to
    # copy the file into the config/ directory within a given robot's moveit_config package.
    # It would then suffice to add "isaac_ros_cumotion" to the list of planning_pipelines.
    # Here we avoid adding the file to the ur_moveit_config package by loading the file manually
    # and adding its contents to the parameter list.
    config_file_path = path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config',
        'isaac_ros_cumotion_planning.yaml'
    )
    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    return (
        {'planning_pipelines': ['ompl', 'isaac_ros_cumotion']},
        {'isaac_ros_cumotion': config}
    )


def generate_launch_description():
    ur_moveit_launch_file = path.join(
        get_package_share_directory('ur_moveit_config'),
        'launch',
        'ur_moveit.launch.py'
    )
    lf = open(ur_moveit_launch_file).read()

    # Rename generate_launch_description() in base launch file.
    lf = lf.replace('generate_launch_description', 'generate_base_launch_description')

    # Add required parameters to the move_group node.  This substitution relies on the fact that
    # the string "moveit_controllers," appears only once in the base launch file.
    lf = lf.replace(
        'moveit_controllers,',
        'moveit_controllers, *cumotion_params(),'
    )

    # Execute modified launch file.
    exec(lf, globals())

    return generate_base_launch_description()  # noqa: F821
