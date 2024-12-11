#!/usr/bin/env python3

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

from geometry_msgs.msg import Pose, PoseStamped
from isaac_ros_moveit_goal_setter.move_group_client import MoveGroupClient
from moveit_msgs.msg import MoveItErrorCodes
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class PoseToPoseNode(Node):

    def __init__(self):
        super().__init__('pose_to_pose_node')

        self._world_frame = self.declare_parameter(
            'world_frame', 'base_link').get_parameter_value().string_value

        self._target_frames = self.declare_parameter(
            'target_frames', ['target1_frame']).get_parameter_value().string_array_value

        self._target_frame_idx = 0

        self._plan_timer_period = self.declare_parameter(
            'plan_timer_period', 0.01).get_parameter_value().double_value

        planner_group_name = self.declare_parameter(
            'planner_group_name', 'ur_manipulator').get_parameter_value().string_value

        pipeline_id = self.declare_parameter(
            'pipeline_id', 'isaac_ros_cumotion').get_parameter_value().string_value

        planner_id = self.declare_parameter(
            'planner_id', 'cuMotion').get_parameter_value().string_value

        ee_link = self.declare_parameter(
            'end_effector_link', 'wrist_3_link').get_parameter_value().string_value

        self.move_group_client = MoveGroupClient(
            self, planner_group_name, pipeline_id, planner_id, ee_link)

        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.timer = self.create_timer(self._plan_timer_period, self.on_timer)

    def _transform_msg_to_pose_msg(self, tf_msg):
        pose = Pose()
        pose.position.x = tf_msg.translation.x
        pose.position.y = tf_msg.translation.y
        pose.position.z = tf_msg.translation.z

        pose.orientation.x = tf_msg.rotation.x
        pose.orientation.y = tf_msg.rotation.y
        pose.orientation.z = tf_msg.rotation.z
        pose.orientation.w = tf_msg.rotation.w
        return pose

    def on_timer(self):

        # Check if there is a valid transform between world and target frame
        try:
            world_frame_pose_target_frame = self._tf_buffer.lookup_transform(
                self._world_frame, self._target_frames[self._target_frame_idx],
                self.get_clock().now(), rclpy.duration.Duration(seconds=10.0)
            )
        except TransformException as ex:
            self.get_logger().warning(f'Waiting for target_frame pose transform to be available \
                                      in TF, between {self._world_frame} and \
                                      {self._target_frames[self._target_frame_idx]}. if \
                                      warning persists, check if the transform is \
                                      published to tf. Message from TF: {ex}')
            return

        output_msg = PoseStamped()
        output_msg.header.stamp = self.get_clock().now().to_msg()
        output_msg.header.frame_id = self._world_frame
        output_msg.pose = self._transform_msg_to_pose_msg(world_frame_pose_target_frame.transform)

        response = self.move_group_client.send_goal_pose(output_msg)
        if response.error_code.val == MoveItErrorCodes.SUCCESS:
            self._target_frame_idx = (self._target_frame_idx + 1) % len(self._target_frames)
        else:
            self.get_logger().warning('target pose was not reachable by planner, trying again \
                                      on the next iteration')


def main(args=None):

    rclpy.init(args=args)

    pose_to_pose_node = PoseToPoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(pose_to_pose_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pose_to_pose_node.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    pose_to_pose_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
