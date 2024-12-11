# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from isaac_manipulator_ros_python_utils.launch_utils import (
    extract_pose_from_parameter,
    extract_vector3_from_parameter,
)
from isaac_manipulator_ros_python_utils.types import (
    AttachState, Mode, ObjectAttachmentShape
)
from isaac_ros_cumotion_interfaces.action import AttachObject

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from visualization_msgs.msg import Marker


class AttachObjectClient(Node):
    """
    Client node to manage object attachment and detachment via ROS 2 actions.

    This client can operate in ONCE or CYCLE mode. In ONCE mode, it sends a single
    attach or detach goal. In CYCLE mode, it alternates between attaching and detaching
    the object at regular intervals.

    mode (Mode): The mode of operation, either ONCE (executes the action once)
            or CYCLE (continuously alternates between attach and detach).
    attach_object (AttachState): The initial state, whether to attach or detach the object.
    object_shape (ObjectAttachmentShape): Shape of the object (SPHERE, CUBOID, CUSTOM_MESH).
    """

    def __init__(self) -> None:
        """Initialize the AttachObjectClient node."""
        super().__init__('attach_object_client')
        self.get_logger().info('Started client...')

        self.declare_parameter('object_attachment_attach_object',
                               AttachState.ATTACH.value)
        self.declare_parameter('object_attachment_mesh_resource',
                               '/workspaces/isaac_ros-dev/temp/nontextured.stl')
        self.declare_parameter('object_attachment_mode',
                               Mode.ONCE.value)
        self.declare_parameter('object_attachment_object_shape',
                               ObjectAttachmentShape.SPHERE.value)
        self.declare_parameter('object_attachment_fallback_radius', 0.15)

        # translation: x, y, z and then orientation x, y, z, w
        self.declare_parameter('object_attachment_object_pose',
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('object_attachment_object_scale',
                               [0.09, 0.185, 0.035])

        self.attach_object = AttachState(
            self.get_parameter('object_attachment_attach_object').value)
        self.mesh_resource = self.get_parameter(
            'object_attachment_mesh_resource').get_parameter_value().string_value
        self.mode = Mode(self.get_parameter(
            'object_attachment_mode').value)
        self.object_shape = ObjectAttachmentShape(
            self.get_parameter('object_attachment_object_shape').value)
        self.fallback_radius = self.get_parameter(
            'object_attachment_fallback_radius').get_parameter_value().double_value
        self._attach_object_client = ActionClient(
            self, AttachObject, 'attach_object')
        self.object_pose = extract_pose_from_parameter(
            self.get_parameter('object_attachment_object_pose'
                               ).get_parameter_value().double_array_value)
        self.object_scale = extract_vector3_from_parameter(
            self.get_parameter('object_attachment_object_scale'
                               ).get_parameter_value().double_array_value)

        self._initialized = True
        self.active_goal = False  # Track if there's an active goal
        self.goal_handle = None  # Handle to manage the active goal
        self.goal_sent = False  # Track if the goal has been sent (for 'once' mode)
        self.cycle_counter = 0  # Counter for the 'cycle' mode

        # Set timer for goal sending
        self.timer = self.create_timer(
            4.00, self.send_goal_callback
        )

    def create_marker(self) -> Marker:
        """
        Create a Marker based on the object type and other parameters.

        Returns
        -------
            Marker: The marker object that defines attachment process.

        """
        marker = Marker()
        marker.pose = self.object_pose
        marker.scale = self.object_scale

        # Handle different object types
        if self.object_shape == ObjectAttachmentShape.SPHERE:
            marker.type = Marker.SPHERE
        elif self.object_shape == ObjectAttachmentShape.CUBOID:
            marker.type = Marker.CUBE
        elif self.object_shape == ObjectAttachmentShape.CUSTOM_MESH:
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = self.mesh_resource
        else:
            self.get_logger().error('Unknown object type')

        return marker

    def send_goal_callback(self) -> None:
        """
        Send a goal based on the current mode.

        In ONCE mode, sends a goal if it hasn't been sent already. In CYCLE mode,
        alternates between attach and detach states.
        """
        if self.mode == Mode.ONCE and not self.goal_sent:
            self.send_goal()
            self.goal_sent = True
        elif self.mode == Mode.CYCLE:
            if not self.active_goal:
                self.attach_object = (
                    AttachState.ATTACH if self.cycle_counter % 2 == 0 else AttachState.DETACH
                )
                self.send_goal()
                self.cycle_counter += 1  # Increment the counter

    def send_goal(self) -> None:
        """
        Construct and send a goal to attach or detach the object.

        Waits for the action server, then sends the goal with a feedback callback.
        """
        goal_msg = AttachObject.Goal()
        goal_msg.attach_object = self.attach_object.value
        goal_msg.fallback_radius = self.fallback_radius
        goal_msg.object_config = self.create_marker()
        self._attach_object_client.wait_for_server()
        self.get_logger().info('Sending new action call...')
        self._send_goal_future = self._attach_object_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future) -> None:
        """
        Handle the response from the action server when a goal is sent.

        If the goal is accepted, track it and set up the result callback.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal Rejected')
            return
        self.goal_handle = goal_handle  # Store the goal handle
        self.active_goal = True  # Mark the goal as active
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg: AttachObject.Feedback) -> None:
        """
        Handle feedback from the action server.

        Logs the status of the ongoing goal.
        """
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.status}')

    def get_result_callback(self, future: Future) -> None:
        """
        Handle the result from the action server.

        Logs the final outcome of the action and marks the goal as complete.
        """
        result = future.result().result
        self.get_logger().info(f'Final Outcome: {result.outcome}')
        self.active_goal = False  # Mark the goal as complete


def main(args=None) -> None:
    rclpy.init(args=args)
    attach_object_client = AttachObjectClient()
    rclpy.spin(attach_object_client)


if __name__ == '__main__':
    main()
