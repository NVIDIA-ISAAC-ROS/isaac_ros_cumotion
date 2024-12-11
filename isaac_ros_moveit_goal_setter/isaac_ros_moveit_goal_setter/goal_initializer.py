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
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class GoalInitializerNode(Node):

    def __init__(self):
        super().__init__('goal_initializer_node')

        self._previous_goal_position = None

        self._world_frame = self.declare_parameter(
            'world_frame', 'base_link').get_parameter_value().string_value

        self._grasp_frame = self.declare_parameter(
            'grasp_frame', 'grasp_frame').get_parameter_value().string_value

        self._grasp_frame_stale_time_threshold = self.declare_parameter(
            'grasp_frame_stale_time_threshold', 30.0).get_parameter_value().double_value

        self._goal_change_position_threshold = self.declare_parameter(
            'goal_change_position_threshold', 0.1).get_parameter_value().double_value

        self._plan_timer_period = self.declare_parameter(
            'plan_timer_period', 2.0).get_parameter_value().double_value

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

        self._tf_buffer = Buffer()
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

        # Check if there is a valid transform between detection and grasp frame
        try:
            # Using rclpy.time.Time() listens to the latest TF in the buffer
            world_frame_pose_grasp_frame = self._tf_buffer.lookup_transform(
                self._world_frame, self._grasp_frame, rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warning(f'Waiting for object pose transform to be available in TF, \
                                      between {self._world_frame} and {self._grasp_frame}. if \
                                      warning persists, check if object pose is detected and \
                                      published to tf. Message from TF: {ex}')
            return

        # Send a warning if the latest grasp frame is older than grasp_frame_stale_time_threshold
        stale_check_time = (self.get_clock().now() - rclpy.time.Time().from_msg(
            world_frame_pose_grasp_frame.header.stamp)).nanoseconds / 1e9
        if stale_check_time > self._grasp_frame_stale_time_threshold:
            self.get_logger().warn(f'A new grasp frame has not been received for \
                                   {self._grasp_frame_stale_time_threshold} seconds.')

        output_msg = PoseStamped()
        output_msg.header.stamp = self.get_clock().now().to_msg()
        output_msg.header.frame_id = self._world_frame
        output_msg.pose = self._transform_msg_to_pose_msg(world_frame_pose_grasp_frame.transform)
        new_goal = np.array([
            output_msg.pose.position.x,
            output_msg.pose.position.y,
            output_msg.pose.position.z,
            ])
        if self._previous_goal_position is not None:

            goal_change_distance = np.linalg.norm(self._previous_goal_position - new_goal)
            if (goal_change_distance <= self._goal_change_position_threshold):
                self.get_logger().warning(
                    f'New goal position is within {self._goal_change_position_threshold} meters at \
                    {goal_change_distance}, not setting new goal. Move goal further to set \
                    new goal.')
                return

        self.move_group_client.send_goal_pose(output_msg)
        self._previous_goal_position = new_goal


def main(args=None):

    rclpy.init(args=args)

    goal_initializer_node = GoalInitializerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(goal_initializer_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        goal_initializer_node.get_logger().info(
            'KeyboardInterrupt, shutting down.\n'
        )
    goal_initializer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
