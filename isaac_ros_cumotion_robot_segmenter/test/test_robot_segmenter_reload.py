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

"""Test for robot segmenter reload functionality via service call."""

import os
import pathlib
import time

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from isaac_ros_cumotion_interfaces.srv import GetRobotDescription
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer, Node
import numpy as np
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import UInt64
import yaml

TEST_FOLDER = pathlib.Path(__file__).parent / 'test_data'

# Service name that the robot segmenter will call
ROBOT_DESCRIPTION_SERVICE_NAME = '/test/get_robot_description'
# Topic name for triggering reload
RELOAD_TOPIC_NAME = '/test/robot_description_updated'


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
            'robot_segmenter.reload_robot_description_topic': RELOAD_TOPIC_NAME,
            'robot_segmenter.robot_description_service_name': ROBOT_DESCRIPTION_SERVICE_NAME,
        }.items(),
    )

    # Launch container
    container = ComposableNodeContainer(
        name='robot_segmenter_container',
        namespace=IsaacROSRobotSegmenterReloadTest.generate_namespace(),
        package='rclcpp_components',
        executable='component_container',
        arguments=['--ros-args', '--log-level', 'isaac_ros_test.robot_segmenter_node:=info'],
        output='screen'
    )

    transform_publishers = []

    # world -> camera_1_link
    transform_publishers.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_camera_link',
        arguments=[
            # Keep the attached-object frame centered in the camera. The
            # recorded camera pose places grasp_frame below the image, making
            # an output-difference assertion dependent on incidental pixels.
            '--x', '-0.70432473',
            '--y', '0.29825179',
            '--z', '-0.79990610',
            '--qx', '0.0',
            '--qy', '0.0',
            '--qz', '0.0',
            '--qw', '1.0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_1_infra1_optical_frame',
        ],
    ))

    all_nodes = [container] + transform_publishers + [robot_segmenter_node]

    return IsaacROSRobotSegmenterReloadTest.generate_test_description(all_nodes)


class IsaacROSRobotSegmenterReloadTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    def setUp(self):
        super().setUp()
        # Load URDF and XRDF content from files
        urdf_path = TEST_FOLDER / 'robot.urdf'
        xrdf_path = TEST_FOLDER / 'robot.xrdf'

        with open(urdf_path, 'r') as f:
            self.urdf_content = f.read()
        with open(xrdf_path, 'r') as f:
            self.xrdf_content = f.read()

        updated_xrdf = yaml.safe_load(self.xrdf_content)
        geometry_name = updated_xrdf['collision']['geometry']
        updated_xrdf['geometry'][geometry_name]['spheres']['attached_object'] = [{
            'center': [0.0, 0.08, 0.005],
            'radius': 0.2,
        }]
        self.updated_xrdf_content = yaml.safe_dump(updated_xrdf)

        self.node.get_logger().info(
            f'Loaded URDF ({len(self.urdf_content)} bytes) and '
            f'XRDF ({len(self.xrdf_content)} bytes) from disk')

        # Create service to provide robot description
        self.robot_description_service = self.node.create_service(
            GetRobotDescription,
            ROBOT_DESCRIPTION_SERVICE_NAME,
            self._handle_get_robot_description
        )
        self.node.get_logger().info(
            f'Created GetRobotDescription service at {ROBOT_DESCRIPTION_SERVICE_NAME}')

        self.service_called = False

    def _handle_get_robot_description(self, request, response):
        """Service callback that returns URDF/XRDF from disk."""
        self.node.get_logger().info('GetRobotDescription service called!')
        response.urdf = self.urdf_content
        response.xrdf = self.xrdf_content
        self.service_called = True
        self.node.get_logger().info(
            f'Returning URDF ({len(response.urdf)} bytes) and '
            f'XRDF ({len(response.xrdf)} bytes)')
        return response

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

    def test_robot_segmenter_reload(self):
        """Test that robot segmenter can reload robot description via service."""
        TIMEOUT = 30
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

        # Create publisher for reload signal
        reload_pub = self.node.create_publisher(
            UInt64, RELOAD_TOPIC_NAME, QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        )

        try:
            # Use a constant 1 m depth plane so the test isolates robot geometry.
            raw_depth_data = np.full((720, 1280), 1000, dtype=np.uint16)
            self.node.get_logger().info(f'Raw depth data shape: {raw_depth_data.shape}')

            # Load camera info from JSON
            depth_camera_info_path = TEST_FOLDER / 'depth_camera_info.json'
            depth_camera_info = JSONConversion.load_camera_info_from_json(depth_camera_info_path)
            depth_camera_info.header.frame_id = 'camera_1_infra1_optical_frame'

            cv_bridge = CvBridge()
            depth_image = cv_bridge.cv2_to_imgmsg(raw_depth_data, '16UC1')
            depth_image.header.frame_id = 'camera_1_infra1_optical_frame'

            # First, get the segmenter working with initial robot description
            self.node.get_logger().info('Step 1: Waiting for initial segmentation to work...')
            end_time = time.time() + TIMEOUT
            initial_segmentation_done = False

            while time.time() < end_time:
                current_time = self.node.get_clock().now().to_msg()
                depth_image.header.stamp = current_time
                depth_camera_info.header.stamp = current_time

                joint_state = self._get_joint_state()

                depth_pub.publish(depth_image)
                depth_info_pub.publish(depth_camera_info)
                joint_states_pub.publish(joint_state)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'robot_mask' in received_messages and received_messages['robot_mask']:
                    initial_segmentation_done = True
                    self.node.get_logger().info(
                        'Initial segmentation working! Received robot_mask output.')
                    break
                time.sleep(0.1)

            self.assertTrue(initial_segmentation_done,
                            'Initial segmentation did not produce output')
            initial_mask = cv_bridge.imgmsg_to_cv2(
                received_messages['robot_mask'][-1], desired_encoding='passthrough').copy()

            # Step 2: Send reload signal
            self.node.get_logger().info(
                'Step 2: Sending reload robot description signal...')

            # Keep the initial description active through initial segmentation;
            # switch the mock service response only when requesting the reload.
            self.xrdf_content = self.updated_xrdf_content
            reload_msg = UInt64()
            reload_msg.data = 1
            reload_pub.publish(reload_msg)

            self.node.get_logger().info(
                f'Published robot description update 1 to {RELOAD_TOPIC_NAME}')

            # Wait for service to be called
            service_wait_end = time.time() + 10
            while time.time() < service_wait_end:
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if self.service_called:
                    self.node.get_logger().info('Service was called successfully!')
                    break
                # Keep publishing inputs to keep the node active
                current_time = self.node.get_clock().now().to_msg()
                depth_image.header.stamp = current_time
                depth_camera_info.header.stamp = current_time
                joint_state = self._get_joint_state()
                depth_pub.publish(depth_image)
                depth_info_pub.publish(depth_camera_info)
                joint_states_pub.publish(joint_state)
                time.sleep(0.1)

            self.assertTrue(self.service_called,
                            'GetRobotDescription service was not called after reload signal')

            # Step 3: Verify segmentation still works after reload
            self.node.get_logger().info(
                'Step 3: Verifying segmentation works after reload...')

            # Clear previous messages
            initial_mask_count = len(received_messages.get('robot_mask', []))
            post_reload_done = False
            attached_object_mask_updated = False
            end_time = time.time() + TIMEOUT

            while time.time() < end_time:
                current_time = self.node.get_clock().now().to_msg()
                depth_image.header.stamp = current_time
                depth_camera_info.header.stamp = current_time

                joint_state = self._get_joint_state()

                depth_pub.publish(depth_image)
                depth_info_pub.publish(depth_camera_info)
                joint_states_pub.publish(joint_state)

                rclpy.spin_once(self.node, timeout_sec=0.1)

                current_mask_count = len(received_messages.get('robot_mask', []))
                if current_mask_count > initial_mask_count:
                    post_reload_done = True
                    updated_mask = cv_bridge.imgmsg_to_cv2(
                        received_messages['robot_mask'][-1], desired_encoding='passthrough')
                    mask_changed = np.count_nonzero(initial_mask != updated_mask) > 0
                    if mask_changed:
                        attached_object_mask_updated = True
                        self.node.get_logger().info(
                            'Robot mask includes the updated attached-object geometry')
                        break
                time.sleep(0.1)

            self.assertTrue(post_reload_done,
                            'Segmentation did not work after robot description reload')
            self.assertTrue(
                attached_object_mask_updated,
                'Attached-object sphere update did not change the robot mask')

            self.node.get_logger().info('TEST PASSED: Robot description reload successful!')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(depth_pub)
            self.node.destroy_publisher(depth_info_pub)
            self.node.destroy_publisher(joint_states_pub)
            self.node.destroy_publisher(reload_pub)
            self.node.destroy_service(self.robot_description_service)
