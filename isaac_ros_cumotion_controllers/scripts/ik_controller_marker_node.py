#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive marker node for controlling the BimanualIkController bimanual end-effector targets.

Publishes two 6-DOF draggable markers in RViz — one for the left arm and one for the right arm.
A PoseArray is published at 50 Hz to /ik_controller/reference_pose where:
  poses[0] = left end-effector target
  poses[1] = right end-effector target

Interaction: click on a ring handle to rotate, click on an arrow to translate.
"""

from geometry_msgs.msg import Pose, PoseArray
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)

# 6-DOF ring/arrow controls.
_6DOF_AXES = [
    ('rotate_x', InteractiveMarkerControl.ROTATE_AXIS, (1.0, 1.0, 0.0, 0.0)),
    ('move_x',   InteractiveMarkerControl.MOVE_AXIS,   (1.0, 1.0, 0.0, 0.0)),
    ('rotate_y', InteractiveMarkerControl.ROTATE_AXIS, (1.0, 0.0, 0.0, 1.0)),
    ('move_y',   InteractiveMarkerControl.MOVE_AXIS,   (1.0, 0.0, 0.0, 1.0)),
    ('rotate_z', InteractiveMarkerControl.ROTATE_AXIS, (1.0, 0.0, 1.0, 0.0)),
    ('move_z',   InteractiveMarkerControl.MOVE_AXIS,   (1.0, 0.0, 1.0, 0.0)),
]


class BimanualIkControllerMarkerNode(Node):

    def __init__(self) -> None:
        super().__init__('ik_controller_marker')

        self.declare_parameter('robot_base_frame', 'pelvis')
        # Left arm initial position (positive Y = robot's left side).
        self.declare_parameter('left_initial_x', 0.3)
        self.declare_parameter('left_initial_y', 0.2)
        self.declare_parameter('left_initial_z', 0.1)
        # Right arm initial position (negative Y = robot's right side).
        self.declare_parameter('right_initial_x', 0.3)
        self.declare_parameter('right_initial_y', -0.2)
        self.declare_parameter('right_initial_z', 0.1)
        self._frame = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self._scale = 0.3

        # Current poses for each arm — initialized from parameters.
        self._left_pose = Pose()
        self._left_pose.position.x = \
            self.get_parameter('left_initial_x').get_parameter_value().double_value
        self._left_pose.position.y = \
            self.get_parameter('left_initial_y').get_parameter_value().double_value
        self._left_pose.position.z = \
            self.get_parameter('left_initial_z').get_parameter_value().double_value
        self._left_pose.orientation.w = 1.0

        self._right_pose = Pose()
        self._right_pose.position.x = \
            self.get_parameter('right_initial_x').get_parameter_value().double_value
        self._right_pose.position.y = \
            self.get_parameter('right_initial_y').get_parameter_value().double_value
        self._right_pose.position.z = \
            self.get_parameter('right_initial_z').get_parameter_value().double_value
        self._right_pose.orientation.w = 1.0

        self._pub = self.create_publisher(
            PoseArray, '/ik_controller/reference_pose', 10)

        self.create_timer(1.0 / 50.0, self._publish_reference)

        self._server = InteractiveMarkerServer(self, 'ik_controller_marker')
        self._create_marker(
            name='left_ik_target',
            description='Left Arm IK Target',
            pose=self._left_pose,
            color=(0.0, 0.9, 0.2),   # green
        )
        self._create_marker(
            name='right_ik_target',
            description='Right Arm IK Target',
            pose=self._right_pose,
            color=(0.2, 0.4, 1.0),   # blue
        )
        self._server.applyChanges()

        self.get_logger().info(
            f"Bimanual IK markers ready in frame '{self._frame}'. "
            "Click ring handles to rotate, arrow handles to translate.")

    def _create_marker(
        self,
        name: str,
        description: str,
        pose: Pose,
        color: tuple,
    ) -> None:
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._frame
        int_marker.name = name
        int_marker.description = description
        int_marker.scale = self._scale
        int_marker.pose = pose

        # Visual sphere at the target point — display only, no interaction.
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = 0.06
        sphere.scale.y = 0.06
        sphere.scale.z = 0.06
        sphere.color.r = color[0]
        sphere.color.g = color[1]
        sphere.color.b = color[2]
        sphere.color.a = 0.9

        visual_ctrl = InteractiveMarkerControl()
        visual_ctrl.always_visible = True
        visual_ctrl.markers.append(sphere)
        int_marker.controls.append(visual_ctrl)

        # 6-DOF ring/arrow handles.
        for ctrl_name, mode, (w, x, y, z) in _6DOF_AXES:
            ctrl = InteractiveMarkerControl()
            ctrl.name = ctrl_name
            ctrl.interaction_mode = mode
            ctrl.orientation.w = w
            ctrl.orientation.x = x
            ctrl.orientation.y = y
            ctrl.orientation.z = z
            int_marker.controls.append(ctrl)

        self._server.insert(int_marker, feedback_callback=self._on_feedback)

    def _publish_reference(self) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.poses.append(self._left_pose)   # poses[0] = left EE
        msg.poses.append(self._right_pose)  # poses[1] = right EE
        self._pub.publish(msg)

    def _on_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        if feedback.event_type not in (
            InteractiveMarkerFeedback.POSE_UPDATE,
            InteractiveMarkerFeedback.MOUSE_UP,
        ):
            return

        # Update the stored pose for whichever marker moved.
        if feedback.marker_name == 'left_ik_target':
            self._left_pose = feedback.pose
        elif feedback.marker_name == 'right_ik_target':
            self._right_pose = feedback.pose

        self._server.setPose(feedback.marker_name, feedback.pose)
        self._server.applyChanges()


def main() -> None:
    rclpy.init()
    node = BimanualIkControllerMarkerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
