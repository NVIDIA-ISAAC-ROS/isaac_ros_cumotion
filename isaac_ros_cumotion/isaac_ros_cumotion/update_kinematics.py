# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from os import path
import time
from typing import Dict, List, Optional

from ament_index_python.packages import get_package_share_directory
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.cuda_robot_model.util import load_robot_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
from curobo.types.state import JointState as CuJointState
from curobo.util.logger import log_error
from curobo.util.xrdf_utils import return_value_if_exists
from curobo.util_file import get_robot_configs_path
from curobo.util_file import join_path
from curobo.util_file import load_yaml
from isaac_ros_cumotion.util import get_spheres_marker
from isaac_ros_cumotion_interfaces.action import UpdateLinkSpheres
import numpy as np
from rclpy.action import ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.impl.rcutils_logger import RcutilsLogger
from rclpy.node import Node
import torch
from visualization_msgs.msg import MarkerArray


def get_robot_config(robot_file: str,
                     urdf_file_path: str,
                     logger: RcutilsLogger,
                     object_link_name: Optional[str] = None) -> Dict:

    if robot_file.lower().endswith('.yml'):
        logger.warn(
            'YML files will be deprecated soon. Switch to XRDF files for future compatibility.')

        # Construct the YAML file path and convert it to a dictionary.
        robot_file_path = join_path(get_robot_configs_path(), robot_file)
        robot_config = load_yaml(robot_file_path)

        # If a ROS param provides the URDF path, override the YAML file's path.
        if urdf_file_path is not None:
            robot_config['robot_cfg']['kinematics']['urdf_path'] = urdf_file_path

    elif robot_file.lower().endswith('.xrdf'):

        # If no URDF file path is provided, stop
        if urdf_file_path is None:
            logger.fatal('urdf_path is required to load robot from XRDF file')
            raise SystemExit

        # Construct the XRDF file path and convert it to a dictionary.
        xrdf_dir_path = path.join(
            get_package_share_directory('isaac_ros_cumotion_robot_description'), 'xrdf')
        xrdf_file_path = join_path(xrdf_dir_path, robot_file)
        content_path = ContentPath(robot_xrdf_absolute_path=xrdf_file_path,
                                   robot_urdf_absolute_path=urdf_file_path)
        robot_config = load_robot_yaml(content_path)

        update_collision_sphere_buffer(
            robot_yaml=robot_config,
            link_name='attached_object',
            num_spheres=100,
        )

        # Add the object link for forward kinematics if specified
        if object_link_name is not None:

            robot_config = append_tool_frames(
                robot_yaml=robot_config,
                tool_frames=[object_link_name]
            )

    else:
        logger.fatal('Invalid robot file; only XRDF or YML files accepted. Halting.')
        raise SystemExit

    return robot_config


def update_collision_sphere_buffer(
    robot_yaml: Dict,
    link_name: str,
    num_spheres: int,
) -> Dict:

    updt_sphere_dict = return_value_if_exists(
        robot_yaml['robot_cfg']['kinematics'],
        'extra_collision_spheres',
        raise_error=False
    )

    if updt_sphere_dict is None:
        updt_sphere_dict = {link_name: num_spheres}
    else:
        updt_sphere_dict[link_name] = num_spheres

    robot_yaml['robot_cfg']['kinematics']['extra_collision_spheres'] = updt_sphere_dict
    return robot_yaml


def append_tool_frames(
    robot_yaml: Dict,
    tool_frames: List[str]
) -> Dict:

    robot_dict = robot_yaml['robot_cfg']['kinematics']
    link_names = return_value_if_exists(robot_dict, 'link_names', raise_error=False)
    if link_names is None:
        link_names = []
    link_names.extend(tool_frames)
    link_names = list(set(link_names))  # Ensure link names are unique
    robot_dict['link_names'] = link_names
    robot_yaml = {'robot_cfg': {'kinematics': robot_dict}}
    return robot_yaml


class UpdateLinkSpheresServer:

    def __init__(self, server_node: Node, action_name: str, robot_kinematics: CudaRobotModel,
                 robot_base_frame: str) -> None:

        self.__server_node = server_node
        self.__action_name = action_name
        self.__robot_kinematics = robot_kinematics
        self.__robot_base_frame = robot_base_frame

        # Initialize a mutually exclusive callback group
        self.__callback_group = MutuallyExclusiveCallbackGroup()

        # Initialize the action server
        self.__action_server = ActionServer(self.__server_node,
                                            UpdateLinkSpheres,
                                            f'{self.__action_name}',
                                            goal_callback=self.goal_callback,
                                            execute_callback=self.execute_callback,
                                            callback_group=self.__callback_group)

        # Initialize the robot sphere publisher
        debug_robot_topic = f'viz_all_spheres/{self.__action_name}'
        self.__debug_robot_publisher = self.__server_node.create_publisher(
            MarkerArray, debug_robot_topic, 10)

    def goal_callback(self, goal_request: UpdateLinkSpheres.Goal) -> GoalResponse:

        if not self.validate_goal(goal_request):
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def validate_goal(self, goal_request: UpdateLinkSpheres.Goal) -> bool:

        if not isinstance(goal_request.attach_object, bool):
            return False

        if not isinstance(goal_request.object_link_name, str):
            return False

        flattened_sphere_arr = goal_request.flattened_sphere_arr

        # Check if all elements are floats
        if not all(isinstance(item, float) for item in flattened_sphere_arr):
            return False

        # Check if the number of elements is a multiple of 4
        if len(flattened_sphere_arr) % 4 != 0:
            return False

        return True

    def execute_callback(self, goal_handle: ServerGoalHandle) -> UpdateLinkSpheres.Result:

        feedback_msg = UpdateLinkSpheres.Feedback()
        result = UpdateLinkSpheres.Result()

        # Extract information from the goal sent by the action client
        attach_object = goal_handle.request.attach_object

        try:
            if attach_object:

                self.handle_attachment(goal_handle, feedback_msg)

            else:
                self.handle_detachment(goal_handle, feedback_msg)

            goal_handle.succeed()
            result.outcome = 'Link Sphere updated'

        except Exception as e:

            feedback_msg.status = f'Error occured while update link spheres: {e}'
            goal_handle.publish_feedback(feedback_msg)

            goal_handle.abort()
            result.outcome = 'Failed to update Link Sphere'

        return result

    def handle_attachment(self, goal_handle: ServerGoalHandle,
                          feedback_msg: UpdateLinkSpheres.Feedback) -> None:

        time_start = time.time()

        feedback_msg.status = f'Attaching object for {self.__action_name}...'
        goal_handle.publish_feedback(feedback_msg)

        flattened_sphere_arr = np.array(goal_handle.request.flattened_sphere_arr)

        sphere_tensor = torch.tensor(flattened_sphere_arr, device='cuda:0', dtype=torch.float32)

        # Reshape the tensor to (N, 4), where N is the number of spheres
        sphere_tensor = sphere_tensor.view(-1, 4)

        self.__robot_kinematics.kinematics_config.update_link_spheres(
            link_name=goal_handle.request.object_link_name,
            sphere_position_radius=sphere_tensor,
            start_sph_idx=0,
        )

        time_total = time.time() - time_start
        feedback_msg.status = f'Attached object for {self.__action_name} in {time_total}s'
        goal_handle.publish_feedback(feedback_msg)

    def handle_detachment(self, goal_handle: ServerGoalHandle,
                          feedback_msg: UpdateLinkSpheres.Feedback) -> None:
        self.__robot_kinematics.kinematics_config.detach_object(
            link_name=goal_handle.request.object_link_name)
        feedback_msg.status = f'Detached Attached object for {self.__action_name}'
        goal_handle.publish_feedback(feedback_msg)

    def publish_all_active_spheres(self,
                                   robot_joint_states: np.ndarray,
                                   robot_joint_names: List[str],
                                   tensor_args: TensorDeviceType,
                                   rgb: List[float] = [0.0, 1.0, 0.0, 1.0]) -> None:

        if self.__debug_robot_publisher.get_subscription_count() == 0:
            return

        q = CuJointState.from_numpy(position=robot_joint_states,
                                    joint_names=robot_joint_names,
                                    tensor_args=tensor_args).unsqueeze(0)

        active_jnames = self.__robot_kinematics.joint_names
        q = q.get_ordered_joint_state(active_jnames)

        if len(q.position.shape) == 1:
            log_error('q should be of shape [b, dof]')
        kin_state = self.__robot_kinematics.get_state(q.position)
        spheres = kin_state.link_spheres_tensor.cpu().numpy()

        current_time = self.__server_node.get_clock().now().to_msg()

        m_arr = get_spheres_marker(
            spheres[0],
            self.__robot_base_frame,
            current_time,
            rgb,
        )

        self.__debug_robot_publisher.publish(m_arr)
