# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
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
import pathlib
import time

import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image


RUN_POL_TEST = os.getenv('RUN_ROBOT_SEGMENTOR_POL_TEST', '').lower() == 'true'


TEST_FOLDER = pathlib.Path(__file__).parent / 'test_data'
ISAAC_ROS_WS = os.getenv('ISAAC_ROS_WS')
if ISAAC_ROS_WS is None:
    raise ValueError('ISAAC_ROS_WS is not set')
ROSBAG_PATH = os.path.join(ISAAC_ROS_WS, 'isaac_ros_assets', 'r2b_2024', 'r2b_robotarm')
if not os.path.exists(ROSBAG_PATH) and RUN_POL_TEST:
    raise ValueError(f'ROSBAG_PATH does not exist: {ROSBAG_PATH}')


@pytest.mark.rostest
def generate_test_description():
    robot_segmenter_node = ComposableNode(
        name='robot_segmenter_node',
        package='isaac_ros_cumotion_robot_segmenter',
        plugin='nvidia::isaac_ros::manipulator::RobotSegmenter',
        namespace=IsaacROSRobotSegmentorPolTest.generate_namespace(),
        parameters=[
            {'enable_performance_logging': True,
             'urdf_path': str(TEST_FOLDER) + '/robot.urdf',
             'xrdf_path': str(TEST_FOLDER) + '/robot.xrdf'}
        ],
        remappings=[
            ('depth_image', '/camera_1/aligned_depth_to_color/image_raw'),
            ('camera_info_depth', '/camera_1/color/camera_info'),
            ('joint_states', '/joint_states')
        ]
    )

    # Launch realsense camera
    container = ComposableNodeContainer(
        name='robot_segmenter_container',
        namespace=IsaacROSRobotSegmentorPolTest.generate_namespace(),
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[robot_segmenter_node],
        arguments=['--ros-args', '--log-level', 'isaac_ros_test.robot_segmenter_node:=error'],
        output='screen'
    )

    # Run a rosbag play.
    rosbag_path = str(TEST_FOLDER) + '/r2b_robotarm_0'
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path],
        output='screen'
    )

    all_nodes = [container, rosbag_play]

    if not RUN_POL_TEST:
        all_nodes = []
        all_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'world', '--child-frame-id', 'base_link',
            ],
        ))

    return IsaacROSRobotSegmentorPolTest.generate_test_description(all_nodes)


WRITE_IMAGES = True


class IsaacROSRobotSegmentorPolTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_robot_segmenter(self):

        if not RUN_POL_TEST:
            return

        received_messages = {}

        cv_bridge = CvBridge()

        self.generate_namespace_lookup(
            ['robot_mask', 'robot_depth'])

        self.create_logging_subscribers(
            [('robot_mask', Image), ('robot_depth', Image)], received_messages,
            accept_multiple_messages=True)

        while not (
            'robot_mask' in received_messages and
            'robot_depth' in received_messages and
            len(received_messages['robot_mask']) > 0 and
            len(received_messages['robot_depth']) > 0
        ):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.1)

        # Process the received robot mask
        robot_mask_msg = received_messages['robot_mask'][-1]

        robot_depth_msg = received_messages['robot_depth'][-1]

        self.node.get_logger().info('Robot mask and depth received')

        # Process the received aligned depth
        robot_mask_output = cv_bridge.imgmsg_to_cv2(robot_mask_msg)

        robot_depth_output = cv_bridge.imgmsg_to_cv2(robot_depth_msg)

        # Optionally save the aligned depth output to a file
        np.save('robot_mask_output_from_node.npy', robot_mask_output)
        np.save('robot_depth_output_from_node.npy', robot_depth_output)

        # Find out who elements are non zero.
        non_zero_elements = np.nonzero(robot_mask_output)
        self.node.get_logger().info(f'Non zero elements: {non_zero_elements}')

        depth_mean = np.mean(robot_depth_output)
        self.node.get_logger().info(f'Depth mean: {depth_mean}')
        depth_std = np.std(robot_depth_output)
        self.node.get_logger().info(f'Depth std: {depth_std}')

        depth_min = np.min(robot_depth_output)
        self.node.get_logger().info(f'Depth min: {depth_min}')
        depth_max = np.max(robot_depth_output)
        self.node.get_logger().info(f'Depth max: {depth_max}')

        # find out uniquer values in the robot mask
        unique_values = np.unique(robot_mask_output)
        self.node.get_logger().info(f'Unique values: {unique_values}')

        depth_normalized = cv2.normalize(robot_depth_output, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = np.uint8(depth_normalized)

        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_VIRIDIS)

        # Save the colorized depth image
        if WRITE_IMAGES:
            cv2.imwrite('robot_mask_newest.png', robot_mask_output)
            cv2.imwrite('robot_depth_newest.png', depth_colored)

        self.node.get_logger().info('A message was successfully received')
