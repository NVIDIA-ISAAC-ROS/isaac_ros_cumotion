#!/usr/bin/env python3

import copy
import time
from typing import List, Optional, Union

from curobo.types.math import Pose as CuPose
from curobo.types.state import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from geometry_msgs.msg import Pose
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import DisplayTrajectory
import numpy as np
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class CumotionGoalSetClient:
    """Client for sending goals to CumotionGoalSetServer."""

    def __init__(self, node, cgroup=None):
        self.node = node
        self.cgroup = cgroup
        self.node.declare_parameter('joint_states_topic', '/joint_states')

        if cgroup is None:
            self.cgroup = MutuallyExclusiveCallbackGroup()

        self.action_client = ActionClient(self.node, MotionPlan, '/cumotion/motion_plan',
                                          callback_group=self.cgroup)

        self.__use_sim_time = (
            self.node.get_parameter('use_sim_time').get_parameter_value().bool_value
        )

        self.__traj_pub = self.node.create_publisher(
            DisplayTrajectory, '/cumotion/display_trajectory', 1)
        self.__joint_states_topic = (
            self.node.get_parameter('joint_states_topic').get_parameter_value().string_value)
        self.subscription = self.node.create_subscription(
            JointState, self.__joint_states_topic, self.js_callback, 10,
            callback_group=self.cgroup,)
        self.__js_buffer = None
        self.__robot_executing = False
        self.execute_plan_client = ActionClient(
            self.node, ExecuteTrajectory, '/execute_trajectory', callback_group=self.cgroup,)

        self.result = None

    def js_callback(self, msg):

        if len(msg.velocity) == 0 or len(msg.position) == 0:
            self.node.get_logger().error('Velocity or position is empty in joint state message'
                                         'The joint state topic connected might have an issue.')
            return

        self.__js_buffer = {
            'joint_names': msg.name,
            'position': msg.position,
            'velocity': msg.velocity,
        }

    def send_plan_goal(self, goal_msg, visualize_trajectory):
        self.node.get_logger().info('Sending goal')
        self.result = None

        self.action_client.wait_for_server()
        self.node.get_logger().info('Found action server')

        self.send_goal_future = self.action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

        while self.result is None:
            time.sleep(0.01)

        if visualize_trajectory and self.result.success:
            msg = DisplayTrajectory()
            msg.trajectory += self.result.planned_trajectory
            self.__traj_pub.publish(msg)

        return self.result

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().info('goal rejected')
            return

        self.node.get_logger().info('goal accepted')

        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.result = future.result().result

    def move_pose(
        self,
        goal_pose: CuPose,
        link_name: str,
        start_state: Optional[CuJointState] = None,
        plan_config: Optional[MotionGenPlanConfig] = None,
        visualize_trajectory: bool = True,
        execute: bool = False,
        goal_pose_array=None,
        disable_collision_links: List[str] = [],
        update_planning_scene: bool = False,

    ):
        # generate request:
        self.node.get_logger().info('Moving to pose')
        goal_msg = MotionPlan.Goal()
        if goal_pose_array is None:
            position = goal_pose.position.cpu().view(1, goal_pose.n_goalset, 3)
            orientation = goal_pose.quaternion.cpu().view(1, goal_pose.n_goalset, 4)
            for i in range(goal_pose.n_goalset):
                pose = Pose()
                pose.position.x = float(position[0, i, 0])
                pose.position.y = float(position[0, i, 1])
                pose.position.z = float(position[0, i, 2])
                pose.orientation.w = float(orientation[0, i, 0])
                pose.orientation.x = float(orientation[0, i, 1])
                pose.orientation.y = float(orientation[0, i, 2])
                pose.orientation.z = float(orientation[0, i, 3])

                goal_msg.goal_pose.poses.append(pose)

            goal_msg.goal_pose.header.frame_id = link_name
        else:
            goal_msg.goal_pose = goal_pose_array
        goal_msg.plan_pose = True
        goal_msg.use_current_state = True
        if start_state is not None:
            goal_msg.use_current_state = False
            goal_msg.start_state.position = start_state.position.cpu().flatten().to_list()
            goal_msg.start_state.name = start_state.joint_names
        goal_msg.use_planning_scene = update_planning_scene
        goal_msg.hold_partial_pose = False
        if plan_config is not None:
            if plan_config.time_dilation_factor is not None:
                goal_msg.time_dilation_factor = plan_config.time_dilation_factor
            if plan_config.pose_cost_metric is not None:
                if plan_config.pose_cost_metric.hold_partial_pose:
                    goal_msg.hold_partial_pose = plan_config.pose_cost_metric.hold_partial_pose
                    goal_msg.hold_partial_pose_vec_weight = (
                        plan_config.pose_cost_metric.hold_vec_weight.cpu().flatten().tolist()
                    )

        # send goal to client:
        goal_msg.disable_collision_links = disable_collision_links
        result = self.send_plan_goal(goal_msg, visualize_trajectory)
        if execute:
            if result.success:
                self.execute_plan(result.planned_trajectory[0])
        # return trajectory:
        return result

    def move_grasp(
        self,
        goal_pose: CuPose,
        link_name: str,
        start_state: Optional[CuJointState] = None,
        grasp_approach_offset_distance: List[float] = [0, 0, -0.15],
        grasp_approach_path_constraint: Union[None, List[float]] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
        retract_offset_distance: List[float] = [0, 0, -0.15],
        retract_path_constraint: Union[None, List[float]] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
        grasp_approach_constraint_in_goal_frame: bool = True,
        retract_constraint_in_goal_frame: bool = True,
        time_dilation_factor: float = 0.2,
        visualize_trajectory: bool = True,
        execute: bool = False,
        goal_pose_array=None,
        disable_collision_links: List[str] = [],
        offset_linear_grasp: bool = True,
        update_planning_scene: bool = False,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_retract: bool = True,
    ):
        # generate request:
        self.node.get_logger().info('Moving to grasp')
        goal_msg = MotionPlan.Goal()
        if goal_pose_array is None:
            position = goal_pose.position.cpu().view(1, goal_pose.n_goalset, 3)
            orientation = goal_pose.quaternion.cpu().view(1, goal_pose.n_goalset, 4)
            for i in range(goal_pose.n_goalset):
                pose = Pose()
                pose.position.x = float(position[0, i, 0])
                pose.position.y = float(position[0, i, 1])
                pose.position.z = float(position[0, i, 2])
                pose.orientation.w = float(orientation[0, i, 0])
                pose.orientation.x = float(orientation[0, i, 1])
                pose.orientation.y = float(orientation[0, i, 2])
                pose.orientation.z = float(orientation[0, i, 3])

                goal_msg.goal_pose.poses.append(pose)

            goal_msg.goal_pose.header.frame_id = link_name
        else:
            goal_msg.goal_pose = goal_pose_array
            goal_msg.goal_pose.header.frame_id = link_name
        goal_msg.plan_pose = True
        goal_msg.use_current_state = True
        if start_state is not None:
            goal_msg.use_current_state = False
            goal_msg.start_state.position = start_state.position.cpu().flatten().to_list()
            goal_msg.start_state.name = start_state.joint_names
        goal_msg.use_planning_scene = update_planning_scene
        goal_msg.hold_partial_pose = False
        goal_msg.time_dilation_factor = time_dilation_factor
        # Fill the six new parameters in here
        goal_msg.grasp_partial_pose_vec_weight = grasp_approach_path_constraint
        grasp_offset_pose = Pose()
        # Keeping the orientation identity
        grasp_offset_pose.position.x = grasp_approach_offset_distance[0]
        grasp_offset_pose.position.y = grasp_approach_offset_distance[1]
        grasp_offset_pose.position.z = grasp_approach_offset_distance[2]
        goal_msg.grasp_offset_pose = grasp_offset_pose
        goal_msg.grasp_approach_constraint_in_goal_frame = grasp_approach_constraint_in_goal_frame

        goal_msg.retract_partial_pose_vec_weight = retract_path_constraint
        retract_offset_pose = Pose()
        # Keeping the orientation identity
        retract_offset_pose.position.x = retract_offset_distance[0]
        retract_offset_pose.position.y = retract_offset_distance[1]
        retract_offset_pose.position.z = retract_offset_distance[2]
        goal_msg.retract_offset_pose = retract_offset_pose
        goal_msg.retract_constraint_in_goal_frame = retract_constraint_in_goal_frame

        goal_msg.disable_collision_links = disable_collision_links
        goal_msg.plan_grasp = True
        goal_msg.plan_approach_to_grasp = plan_approach_to_grasp
        goal_msg.plan_grasp_to_retract = plan_grasp_to_retract
        goal_msg.grasp_offset_pose = grasp_offset_pose
        result = self.send_plan_goal(goal_msg, visualize_trajectory)
        if execute:
            if result.success:
                self.execute_plan(result.planned_trajectory[0])
        return result

    def move_joint(
        self,
        goal_state: CuJointState,
        start_state: Optional[CuJointState] = None,
        plan_config: Optional[MotionGenPlanConfig] = None,
    ):
        # Reserving for future use
        pass

    def execute_plan(self, robot_trajectory, wait_until_complete: bool = True):
        # check if robot's current state is within start state of plan:
        self.node.get_logger().info('Executing plan')
        if self.__robot_executing:
            self.node.get_logger().error('Robot is still executing previous command')
            return False, None
        self.node.get_logger().info('waiting for js')
        self.__js_buffer = None
        while self.__js_buffer is None:
            time.sleep(0.001)
        self.node.get_logger().info('received js')

        current_js = copy.deepcopy(self.__js_buffer)
        self.__js_buffer = None
        start_point = robot_trajectory.joint_trajectory.points[0]
        start_point_names = robot_trajectory.joint_trajectory.joint_names
        current_names = current_js['joint_names']
        joint_order = []
        for j in start_point_names:
            joint_order.append(current_names.index(j))

        current_position = [current_js['position'][i] for i in joint_order]
        current_velocity = [current_js['velocity'][i] for i in joint_order]

        position_error = np.linalg.norm(
            np.ravel(current_position) - np.ravel(start_point.positions))

        velocity_error = np.linalg.norm(
            np.ravel(current_velocity) - np.ravel(start_point.velocities))

        if not self.__use_sim_time:
            if position_error > 0.05:
                self.node.get_logger().error('Start joint position has large error from current \
                                            robot state, l2 error is ' + str(position_error))
                return False, None

            if velocity_error > 0.05:
                self.node.get_logger().error('Start joint velocity has large error from current \
                                            robot state, l2 error is ' + str(velocity_error))
                return False, None

        self.node.get_logger().info('executing goal')
        self.result = None

        self.execute_plan_client.wait_for_server()
        self.node.get_logger().info('Found action server')

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = robot_trajectory
        goal_msg.trajectory.joint_trajectory.header = Header()
        goal_msg.trajectory.multi_dof_joint_trajectory.header = Header()

        self.send_goal_future = self.execute_plan_client.send_goal_async(
            goal_msg, feedback_callback=self.trajectory_execution_feedback_cb
        )

        self.__robot_executing = True
        self.send_goal_future.add_done_callback(self.goal_response_callback)
        if wait_until_complete:
            self.wait_until_complete()
        return True, self.result

    def trajectory_execution_feedback_cb(self, msg):
        self.node.get_logger().info('Feedback: ' + msg.feedback.state)

    def wait_until_complete(self):
        if self.__robot_executing:
            while self.result is None:
                time.sleep(0.01)

        self.__robot_executing = False
