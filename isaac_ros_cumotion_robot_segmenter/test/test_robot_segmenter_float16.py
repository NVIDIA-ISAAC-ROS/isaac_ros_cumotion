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

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer, Node
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image, JointState


TEST_FOLDER = pathlib.Path(__file__).parent / 'test_data'
WRITE_IMAGES = False


@pytest.mark.rostest
def generate_test_description():
    launch_files_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_segmenter'),
        'launch')

    robot_segmenter_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [launch_files_include_dir, '/robot_segmenter.launch.py']
        ),
        launch_arguments={
            'robot_segmenter.urdf_path': str(TEST_FOLDER) + '/robot.urdf',
            'robot_segmenter.xrdf_path': str(TEST_FOLDER) + '/robot.xrdf',
            'robot_segmenter.input_qos': 'DEFAULT',
            'robot_segmenter.output_qos': 'DEFAULT',
            'robot_segmenter.depth_image_topics': ['/isaac_ros_test/depth_image'],
            'robot_segmenter.depth_camera_infos': ['/isaac_ros_test/camera_info_depth'],
            'robot_segmenter.robot_mask_publish_topics': ['/isaac_ros_test/robot_mask'],
            'robot_segmenter.world_depth_publish_topics': ['/isaac_ros_test/robot_depth'],
            'robot_segmenter.joint_states_topic': '/isaac_ros_test/joint_states',
            'robot_segmenter.container_name': '/isaac_ros_test/robot_segmenter_container',
        }.items(),
    )

    # Launch realsense camera
    container = ComposableNodeContainer(
        name='robot_segmenter_container',
        namespace=IsaacROSRobotSegmentorTest.generate_namespace(),
        package='rclcpp_components',
        executable='component_container',
        arguments=['--ros-args', '--log-level', 'isaac_ros_test.robot_segmenter_node:=info'],
        output='screen'
    )

    transform_publishers = []

    # 1. world -> camera_1_link
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_camera_link',
        arguments=[
            '--x', '-0.686180830001831',
            '--y', '0.5951766967773438',
            '--z', '0.9960432648658752',
            '--qx', '-0.007744422182440758',
            '--qy', '0.9010432958602905',
            '--qz', '-0.42730608582496643',
            '--qw', '0.07396451383829117',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_1_infra1_optical_frame',
        ],
    ))

    all_nodes = [container] + transform_publishers + [robot_segmenter_node]

    return IsaacROSRobotSegmentorTest.generate_test_description(all_nodes)


class IsaacROSRobotSegmentorTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def _get_joint_state(self):
        joint_state = JointState()
        joint_state.header.stamp = self.node.get_clock().now().to_msg()
        joint_state.name = [
            'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
            'wrist_2_joint', 'wrist_3_joint', 'shoulder_pan_joint'
        ]
        joint_state.position = [
            -2.216074605981344, -0.9371711015701294, 4.745356357633629,
            1.5836931467056274, 1.3514267206192017, -0.23051911989320928
        ]
        return joint_state

    def test_robot_segmenter(self):
        TIMEOUT = 20
        received_messages = {}

        self.generate_namespace_lookup(
            ['robot_mask', 'depth_image', 'robot_depth', 'camera_info_depth', 'joint_states'])

        subs = self.create_logging_subscribers(
            [('robot_mask', Image), ('robot_depth', Image)], received_messages,
            accept_multiple_messages=True)

        # Create publishers
        depth_pub = self.node.create_publisher(
            Image, self.namespaces['depth_image'], self.DEFAULT_QOS
        )
        depth_info_pub = self.node.create_publisher(
            CameraInfo, self.namespaces['camera_info_depth'], self.DEFAULT_QOS
        )
        joint_states_pub = self.node.create_publisher(
            JointState, self.namespaces['joint_states'], self.DEFAULT_QOS
        )
        try:
            # raw_depth_data = np.zeros((720, 1280), dtype=np.float32)
            # load raw image.
            path_npy_file_for_depth = str(TEST_FOLDER) + '/input_depth_image.npy'
            # original depth stored in mms, need to convert to meters
            raw_depth_data = np.load(path_npy_file_for_depth).astype(np.float32) / 1000.0
            # max values
            max_value = np.max(raw_depth_data)
            self.node.get_logger().info(f'Max value: {max_value}')
            # min values
            min_value = np.min(raw_depth_data)
            self.node.get_logger().info(f'Min value: {min_value}')
            # mean values
            mean_value = np.mean(raw_depth_data)
            self.node.get_logger().info(f'Mean value: {mean_value}')
            self.node.get_logger().info(f'Raw depth data shape: {raw_depth_data.shape}')

            # Load camera info from JSON files using JSONConversion
            depth_camera_info_path = TEST_FOLDER / 'depth_camera_info.json'
            depth_camera_info = JSONConversion.load_camera_info_from_json(depth_camera_info_path)

            # Update frame IDs to match our test setup
            depth_camera_info.header.frame_id = 'camera_1_infra1_optical_frame'

            for i in range(1):

                # Create depth image message from loaded data
                cv_bridge = CvBridge()
                depth_image = cv_bridge.cv2_to_imgmsg(raw_depth_data, '32FC1')
                depth_image.header.frame_id = 'camera_1_infra1_optical_frame'

                end_time = time.time() + TIMEOUT
                done = False
                while time.time() < end_time:
                    # Update timestamps
                    current_time = self.node.get_clock().now().to_msg()
                    depth_image.header.stamp = current_time
                    depth_camera_info.header.stamp = current_time

                    joint_state = self._get_joint_state()

                    # Publish all inputs
                    depth_pub.publish(depth_image)
                    depth_info_pub.publish(depth_camera_info)
                    joint_states_pub.publish(joint_state)

                    rclpy.spin_once(self.node, timeout_sec=0.1)

                    if (
                        'robot_mask' in received_messages and
                        'robot_depth' in received_messages and
                        len(received_messages['robot_mask']) > 0 and
                        len(received_messages['robot_depth']) > 0
                    ):
                        done = True
                        break
                    time.sleep(0.1)

            self.assertTrue(done, 'Appropiate output not received')

            # Process the received robot mask
            robot_mask_msg = received_messages['robot_mask'][-1]
            depth_image = received_messages['robot_depth'][-1]

            # Dump this image to a file.
            robot_mask_output = cv_bridge.imgmsg_to_cv2(robot_mask_msg)
            robot_depth_output = cv_bridge.imgmsg_to_cv2(depth_image)

            # find unique values in robot depth.
            unique_values = np.unique(robot_depth_output)
            unique_values_mask = np.unique(robot_mask_output)
            self.node.get_logger().info(f'Unique values mask: {unique_values_mask}')

            # For float16, since its 0 or 1. we want robot to be white so.
            # (0 == robot, 1 is background)
            robot_mask_output[robot_mask_output == 0] = 255

            self.node.get_logger().info(f'Unique values: {unique_values}')

            depth_normalized = cv2.normalize(robot_depth_output, None, 0, 255, cv2.NORM_MINMAX)
            depth_8bit = np.uint8(depth_normalized)
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_VIRIDIS)
            if WRITE_IMAGES:
                cv2.imwrite(
                    str(TEST_FOLDER) + '/robot_segmenter_output_mask_float32.png',
                    robot_mask_output)
                cv2.imwrite(
                    str(TEST_FOLDER) + '/robot_segmenter_output_depth_float32.png',
                    depth_colored)

            # Make sure that the output is the same as cached depth and mask.
            cached_depth_path = str(TEST_FOLDER) + '/golden_robot_segmenter_depth_float16.npy'
            cached_depth_data = np.load(cached_depth_path).astype(np.float32)
            cached_mask_path = str(TEST_FOLDER) + '/golden_robot_segmenter_mask_float16.npy'
            cached_mask_data = np.load(cached_mask_path).astype(np.float32)
            self.assertTrue(np.allclose(robot_depth_output, cached_depth_data))
            self.assertTrue(np.allclose(robot_mask_output, cached_mask_data))

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(depth_info_pub)
            self.node.destroy_publisher(joint_states_pub)
