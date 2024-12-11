# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

from curobo.types.math import Pose
from curobo.types.state import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from curobo.wrap.reacher.motion_gen import PoseCostMetric
from isaac_ros_cumotion.cumotion_planner import CumotionActionServer
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.msg import MoveItErrorCodes
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor


class CumotionGoalSetPlannerServer(CumotionActionServer):

    def __init__(self):

        super().__init__()
        self._goal_set_planner_server = ActionServer(
            self, MotionPlan, 'cumotion/motion_plan', self.motion_plan_execute_callback
        )

    def warmup(self):
        self.get_logger().info('warming up cuMotion, wait until ready')
        self.motion_gen.warmup(enable_graph=True, n_goalset=100, warmup_js_trajopt=True)
        self.get_logger().info('cuMotion is ready for planning queries!')

    def toggle_link_collision(self, collision_link_names: List[str],
                              enable_flag: bool):
        if len(collision_link_names) > 0:
            if enable_flag:
                for k in collision_link_names:
                    self.motion_gen.kinematics.kinematics_config.enable_link_spheres(k)
            else:
                for k in collision_link_names:
                    self.motion_gen.kinematics.kinematics_config.disable_link_spheres(k)

    def get_cu_pose_from_ros_pose(self, ros_pose):
        cu_pose = Pose.from_list(
            [ros_pose.position.x,
             ros_pose.position.y,
             ros_pose.position.z,
             ros_pose.orientation.w,
             ros_pose.orientation.x,
             ros_pose.orientation.y,
             ros_pose.orientation.z]
        )
        return cu_pose

    def get_goal_poses(self, plan_req: MotionPlan.Goal) -> Pose:
        if plan_req.goal_pose.header.frame_id != self.motion_gen.kinematics.base_link:
            self.get_logger().error(
                'Planning frame: '
                + plan_req.goal_pose.header.frame_id
                + ' is not same as motion gen frame: '
                + self.motion_gen.kinematics.base_link
            )
            return False, MoveItErrorCodes.INVALID_LINK_NAME, []
        poses = []
        for k in plan_req.goal_pose.poses:
            poses.append(
                [
                    k.position.x,
                    k.position.y,
                    k.position.z,
                    k.orientation.w,
                    k.orientation.x,
                    k.orientation.y,
                    k.orientation.z,
                ]
            )
        if len(poses) == 0:
            self.get_logger().error('No goal pose found')
            return False, MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS, poses
        goal_pose = Pose.from_batch_list(poses, self.motion_gen.tensor_args)
        goal_pose = Pose(
            position=goal_pose.position.view(1, -1, 3),
            quaternion=goal_pose.quaternion.view(1, -1, 4),
        )
        return True, MoveItErrorCodes.SUCCESS, goal_pose

    def motion_plan_execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        pose_cost_metric = None

        # check moveit scaling factors:
        time_dilation_factor = goal_handle.request.time_dilation_factor
        if time_dilation_factor == 0.0:
            time_dilation_factor = 0.1
            self.get_logger().warn('Cannot set time_dilation_factor = 0.0')
        self.get_logger().info('Planning with time_dilation_factor: ' + str(time_dilation_factor))

        goal_handle.succeed()
        self.motion_gen.reset(reset_seed=False)

        result = MotionPlan.Result()
        result.success = False
        if goal_handle.request.use_planning_scene:
            self.get_logger().info('Updating planning scene')
            scene = goal_handle.request.world
            world_objects = scene.collision_objects
            world_update_status = self.update_world_objects(world_objects)
            if not world_update_status:
                result.error_code.val = MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
                self.get_logger().error('World update failed.')
                return result

        start_state = None
        plan_req = goal_handle.request
        if plan_req.use_current_state:
            if self._CumotionActionServer__js_buffer is None:
                self.get_logger().error(
                    'joint_state was not received from '
                    + self._CumotionActionServer__joint_states_topic
                )
                return result
            # read joint state:
            state = CuJointState.from_position(
                position=self.tensor_args.to_device(
                    self._CumotionActionServer__js_buffer['position']
                ).unsqueeze(0),
                joint_names=self._CumotionActionServer__js_buffer['joint_names'],
            )
            state.velocity = self.tensor_args.to_device(
                self._CumotionActionServer__js_buffer['velocity']
            ).unsqueeze(0)
            start_state = self.motion_gen.get_active_js(state)
            self._CumotionActionServer__js_buffer = None
        elif len(plan_req.start_state.position) > 0:
            start_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(plan_req.start_state.position).unsqueeze(
                        0
                    ),
                    joint_names=plan_req.start_state.name,
                )
            )
        else:
            self.get_logger().error('joint state in start state was empty')
            return result

        if plan_req.plan_grasp:
            self.get_logger().info('Planning to Grasp Object with stop at offset distance')
            success, error_code, poses = self.get_goal_poses(plan_req)
            self.get_logger().info(f'Success, Error Code): {success}, {error_code}!')
            if not success:
                result.error_code.val = error_code
                return result
            grasp_vec_weight = None
            if len(plan_req.grasp_partial_pose_vec_weight) == 6:
                grasp_vec_weight = [plan_req.grasp_partial_pose_vec_weight[i] for i in range(6)]

            retract_vec_weight = None
            if len(plan_req.retract_partial_pose_vec_weight) == 6:
                retract_vec_weight = [plan_req.retract_partial_pose_vec_weight[i]
                                      for i in range(6)]

            grasp_constraint_in_goal_frame = plan_req.grasp_approach_constraint_in_goal_frame
            retract_constraint_in_goal_frame = plan_req.retract_constraint_in_goal_frame
            grasp_plan_result = self.motion_gen.plan_grasp(
                start_state,
                poses,
                MotionGenPlanConfig(
                    max_attempts=self._CumotionActionServer__max_attempts,
                    enable_graph_attempt=1,
                    time_dilation_factor=time_dilation_factor,
                ),
                grasp_approach_offset=self.get_cu_pose_from_ros_pose(plan_req.grasp_offset_pose),
                grasp_approach_path_constraint=grasp_vec_weight,
                retract_offset=self.get_cu_pose_from_ros_pose(plan_req.retract_offset_pose),
                retract_path_constraint=retract_vec_weight,
                disable_collision_links=plan_req.disable_collision_links,
                plan_approach_to_grasp=plan_req.plan_approach_to_grasp,
                plan_grasp_to_retract=plan_req.plan_grasp_to_retract,
                grasp_approach_constraint_in_goal_frame=grasp_constraint_in_goal_frame,
                retract_constraint_in_goal_frame=retract_constraint_in_goal_frame,
            )
            if grasp_plan_result.success.item():
                traj = self.get_joint_trajectory(
                    grasp_plan_result.grasp_trajectory, grasp_plan_result.grasp_trajectory_dt,
                )
                result.planning_time = grasp_plan_result.planning_time
                result.planned_trajectory.append(traj)
                if plan_req.plan_grasp_to_retract:
                    traj = self.get_joint_trajectory(
                        grasp_plan_result.retract_trajectory,
                        grasp_plan_result.retract_trajectory_dt,
                    )
                    result.planned_trajectory.append(traj)
                result.success = True
                result.goal_index = grasp_plan_result.goalset_index.item()
            else:
                result.success = False
                result.message = grasp_plan_result.status
        else:
            if plan_req.plan_cspace:
                self.get_logger().info('Planning CSpace target')
                if len(plan_req.goal_state.position) <= 0:
                    self.get_logger().error('goal state is empty')
                    return result
                goal_state = self.motion_gen.get_active_js(
                    CuJointState.from_position(
                        position=self.tensor_args.to_device(
                            plan_req.goal_state.position).unsqueeze(0),
                        joint_names=plan_req.goal_state.name,
                    )
                )
                self.toggle_link_collision(plan_req.disable_collision_links, False)

                motion_gen_result = self.motion_gen.plan_single_js(
                    start_state,
                    goal_state,
                    MotionGenPlanConfig(
                        max_attempts=self._CumotionActionServer__max_attempts,
                        enable_graph_attempt=1,
                        time_dilation_factor=time_dilation_factor,
                    ),
                )
                self.toggle_link_collision(plan_req.disable_collision_links, True)

            elif plan_req.plan_pose:
                self.get_logger().info('Planning Pose target')
                if plan_req.hold_partial_pose:
                    if len(plan_req.grasp_partial_pose_vec_weight) < 6:
                        self.get_logger().error('Partial pose vec weight should be of length 6')
                        return result

                    grasp_vec_weight = [plan_req.grasp_partial_pose_vec_weight[i]
                                        for i in range(6)]
                    pose_cost_metric = PoseCostMetric(
                        hold_partial_pose=True,
                        grasp_vec_weight=self.motion_gen.tensor_args.to_device(grasp_vec_weight),
                    )

                # read goal poses:
                success, error_code, poses = self.get_goal_poses(plan_req)
                if not success:
                    result.error_code.val = error_code
                    return result

                self.toggle_link_collision(plan_req.disable_collision_links, False)
                if poses.shape[1] == 1:
                    poses.position = poses.position.view(-1, 3)
                    poses.quaternion = poses.quaternion.view(-1, 4)
                    motion_gen_result = self.motion_gen.plan_single(
                        start_state,
                        poses,
                        MotionGenPlanConfig(
                            max_attempts=self._CumotionActionServer__max_attempts,
                            enable_graph_attempt=1,
                            time_dilation_factor=time_dilation_factor,
                            pose_cost_metric=pose_cost_metric,
                        ),
                    )
                else:
                    motion_gen_result = self.motion_gen.plan_goalset(
                        start_state,
                        poses,
                        MotionGenPlanConfig(
                            max_attempts=self._CumotionActionServer__max_attempts,
                            enable_graph_attempt=1,
                            time_dilation_factor=time_dilation_factor,
                            pose_cost_metric=pose_cost_metric,
                        ),
                    )
                self.toggle_link_collision(plan_req.disable_collision_links, True)

            if motion_gen_result.success.item():
                result.error_code.val = MoveItErrorCodes.SUCCESS
                traj = self.get_joint_trajectory(
                    motion_gen_result.optimized_plan, motion_gen_result.optimized_dt.item()
                )
                result.planning_time = motion_gen_result.total_time
                result.planned_trajectory.append(traj)
                result.success = True
                result.goal_index = motion_gen_result.goalset_index.item()
            elif not motion_gen_result.valid_query:
                self.get_logger().error(f'Invalid planning query: {motion_gen_result.status}')
                if motion_gen_result.status == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS:
                    result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
                if motion_gen_result.status in [
                    MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                    MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
                ]:
                    result.error_code.val = MoveItErrorCodes.START_STATE_IN_COLLISION
            else:
                self.get_logger().error(
                    f'Motion planning failed wih status: {motion_gen_result.status}'
                )
                if motion_gen_result.status == MotionGenStatus.IK_FAIL:
                    result.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION

            self.get_logger().info(
                'returned planning result (query, success, failure_status): '
                + str(self._CumotionActionServer__query_count)
                + ' '
                + str(motion_gen_result.success.item())
                + ' '
                + str(motion_gen_result.status)
            )

        self._CumotionActionServer__query_count += 1

        return result


def main(args=None):
    rclpy.init(args=args)
    cumotion_action_server = CumotionGoalSetPlannerServer()
    executor = MultiThreadedExecutor()
    executor.add_node(cumotion_action_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        cumotion_action_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    cumotion_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
