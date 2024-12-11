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

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'isaac_ros_moveit_goal_setter'

setup(
    name=package_name,
    version='3.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    author='Kajanan Chinniah',
    description='This package sets the goal for MoveIt',
    license='NVIDIA Isaac ROS Software License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_to_pose_node = isaac_ros_moveit_goal_setter.pose_to_pose:main',
            'goal_initializer_node = isaac_ros_moveit_goal_setter.goal_initializer:main'
        ],
    },
)
