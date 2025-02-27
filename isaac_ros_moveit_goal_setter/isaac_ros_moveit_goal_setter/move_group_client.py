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

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, \
                            OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import time


class MoveGroupClient:

    def __init__(self, node, planner_group_name, pipeline_id, planner_id, ee_link):
        self._node = node
        self._planner_group_name = planner_group_name
        self._pipeline_id = pipeline_id
        self._planner_id = planner_id
        self._end_effector_link = ee_link

        self._action_client_cb_group = MutuallyExclusiveCallbackGroup()

        self._action_client = ActionClient(node, MoveGroup, '/move_action',
                                           callback_group=self._action_client_cb_group)

        self._result = None

        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().info('Server move_action not available! Waiting...')

    def _get_pose_constraints(self, pose):
        constraints = Constraints()
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = pose.header.frame_id
        position_constraint.link_name = self._end_effector_link
        position_constraint.constraint_region.primitive_poses.append(pose.pose)
        constraints.position_constraints.extend([position_constraint])

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = pose.header.frame_id
        orientation_constraint.link_name = self._end_effector_link
        orientation_constraint.orientation = pose.pose.orientation
        constraints.orientation_constraints.extend([orientation_constraint])
        return constraints

    def _get_joint_constraints(self, positions, joint_names):
        constraints = Constraints()
        for position, joint_name in zip(positions, joint_names):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = position
            constraints.joint_constraints.append(joint_constraint)

    def send_goal_pose(self, pose, allowed_planning_time=10.0):
        self._result = None
        goal_msg = MoveGroup.Goal()
        goal_msg.request.planner_id = self._planner_id
        goal_msg.request.pipeline_id = self._pipeline_id
        goal_msg.request.group_name = self._planner_group_name
        goal_msg.request.goal_constraints.append(self._get_pose_constraints(pose))
        goal_msg.planning_options.plan_only = False
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.allowed_planning_time = allowed_planning_time

        self._node.get_logger().debug('Sending pose target to planner.')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        while self._result is None:
            time.sleep(0.01)

        return self._result

    def send_goal_joints(self, positions, joint_names, allowed_planning_time=10.0):
        self._result = None
        goal_msg = MoveGroup.Goal()
        goal_msg.request.planner_id = self._planner_id
        goal_msg.request.pipeline_id = self._pipeline_id
        goal_msg.request.group_name = self._planner_group_name
        goal_msg.request.goal_constraints.append(
            self._get_joint_constraints(positions, joint_names))
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.allowed_planning_time = allowed_planning_time
        self._action_client.wait_for_server()
        self._node.get_logger().debug('Sending joint targets to planner.')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        while self._result is None:
            time.sleep(0.01)

        return self._result

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().error('Failed to set target!')
            return
        self._node.get_logger().info('Planning Goal request accepted!')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self._result = future.result().result
        if self._result.error_code.val == MoveItErrorCodes.SUCCESS:
            self._node.get_logger().info('Motion planning succeeded.')
        else:
            self._node.get_logger().error('Planning failed!')
