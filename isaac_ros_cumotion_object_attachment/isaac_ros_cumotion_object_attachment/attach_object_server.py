# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
from enum import Enum
import os
import struct
import threading
import time
import traceback
from typing import Dict, List, Tuple, Union

from action_msgs.msg import GoalStatus
import cupy as cp
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Cuboid as CuCuboid
from curobo.geom.types import Mesh as CuMesh
from curobo.geom.types import Obstacle as CuObstacle
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose as CuPose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState as CuJointState
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from geometry_msgs.msg import Pose, Vector3
from hdbscan import HDBSCAN as cpu_HDBSCAN
from isaac_ros_common.qos import add_qos_parameter
from isaac_ros_cumotion.update_kinematics import get_robot_config
from isaac_ros_cumotion.util import get_spheres_marker
from isaac_ros_cumotion_interfaces.action import AttachObject
from isaac_ros_cumotion_interfaces.action import UpdateLinkSpheres
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.action import ActionClient, ActionServer, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.task import Future
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import (CameraInfo, Image, JointState, PointCloud2,
                             PointField)
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import torch
import trimesh
from visualization_msgs.msg import Marker, MarkerArray


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ObjectState(Enum):
    """
    Enum representing the state of an object.

    Values:
        ATTACHED (str): The object is currently attached.
        DETACHED (str): The object is currently detached.
        ATTACHED_FALLBACK (str): The object has been approximated by a fallback sphere.
    """

    ATTACHED = 'attached'
    DETACHED = 'detached'
    ATTACHED_FALLBACK = 'attached (fallback)'


class AttachObjectServer(Node):
    """
    Node that handles attaching or detaching an object based on a given action goal.

    It processes a depth image, with the robot segmented out, and the joint state. The goal is to
    extract the object's point cloud, approximate it with a mesh, and generate object spheres using
    forward kinematics. These spheres are then sent to other nodes to update their robot model.
    """

    def __init__(self):

        # Constructor of the parent class
        super().__init__('object_attachment_node')

        # Initialize parameters

        self.declare_parameter('robot', 'ur5e_robotiq_2f_140.xrdf')
        self.declare_parameter('urdf_path', rclpy.Parameter.Type.STRING)

        self.declare_parameter('cuda_device', 0)
        self.declare_parameter('time_sync_slop', 0.1)
        self.declare_parameter('filter_depth_buffer_time', 0.1)
        self.declare_parameter('tf_lookup_duration', 5.0)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('depth_image_topics', [
                               '/cumotion/camera_1/world_depth'])
        self.declare_parameter('depth_camera_infos', [
                               '/camera_1/aligned_depth_to_color/camera_info'])
        self.declare_parameter('object_link_name', 'attached_object')
        self.declare_parameter('object_attachment_gripper_frame_name', 'grasp_frame')
        self.declare_parameter(
            'action_names', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('search_radius', 0.2)
        self.declare_parameter('surface_sphere_radius', 0.01)

        # Declare clustering parameters
        self.declare_parameter('clustering_bypass_clustering', False)
        self.declare_parameter('clustering_hdbscan_min_samples', 20)
        self.declare_parameter('clustering_hdbscan_min_cluster_size', 30)
        self.declare_parameter(
            'clustering_hdbscan_cluster_selection_epsilon', 0.5)
        self.declare_parameter('clustering_num_top_clusters_to_select', 3)
        self.declare_parameter('clustering_group_clusters', False)
        self.declare_parameter('clustering_min_points', 100)

        # Get frame information from URDF that object attachment adds the spheres w.r.t to
        self.declare_parameter('object_attachment_grasp_frame_name', 'grasp_frame')
        # Number of spheres to add for CUBOID or MESH approach
        self.declare_parameter('object_attachment_n_spheres', 100)

        # Nvblox parameters for object clearing
        # update_esdf_on_request determines if the esdf should be updated upon service request
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('trigger_aabb_object_clearing', False)

        # object_esdf_clearing_padding adds padding to the dims of the AABB / radius of the sphere
        self.declare_parameter('object_esdf_clearing_padding', [0.05, 0.05, 0.05])

        # QOS settings
        depth_qos = add_qos_parameter(self, 'DEFAULT', 'depth_qos')
        depth_info_qos = add_qos_parameter(self, 'DEFAULT', 'depth_info_qos')

        # Retrieve parameters
        self.__robot_file = self.get_parameter(
            'robot').get_parameter_value().string_value

        try:
            self.__urdf_path = self.get_parameter('urdf_path')
            self.__urdf_path = self.__urdf_path.get_parameter_value().string_value
            if self.__urdf_path == '':
                self.__urdf_path = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__urdf_path = None

        self.__cuda_device_id = self.get_parameter(
            'cuda_device').get_parameter_value().integer_value
        self.__time_sync_slop = self.get_parameter(
            'time_sync_slop').get_parameter_value().double_value
        self.__filter_depth_buffer_time = self.get_parameter(
            'filter_depth_buffer_time').get_parameter_value().double_value
        self.__tf_lookup_duration = self.get_parameter(
            'tf_lookup_duration').get_parameter_value().double_value
        self.__joint_states_topic = self.get_parameter(
            'joint_states_topic').get_parameter_value().string_value
        self.__depth_image_topics = self.get_parameter(
            'depth_image_topics').get_parameter_value().string_array_value
        self.__depth_camera_infos = self.get_parameter(
            'depth_camera_infos').get_parameter_value().string_array_value
        self.__object_link_name = self.get_parameter(
            'object_link_name').get_parameter_value().string_value
        self.__gripper_frame_name = self.get_parameter(
            'object_attachment_gripper_frame_name').get_parameter_value().string_value
        self.__action_names = list(self.get_parameter(
            'action_names').get_parameter_value().string_array_value)
        self.__search_radius = self.get_parameter(
            'search_radius').get_parameter_value().double_value
        self.__surface_sphere_radius = self.get_parameter(
            'surface_sphere_radius').get_parameter_value().double_value
        self.__object_attachment_n_spheres = self.get_parameter(
            'object_attachment_n_spheres').get_parameter_value().integer_value

        # Retrieve clustering parameters
        self.__bypass_clustering = self.get_parameter(
            'clustering_bypass_clustering').get_parameter_value().bool_value
        self.__hdbscan_min_samples = self.get_parameter(
            'clustering_hdbscan_min_samples').get_parameter_value().integer_value
        self.__hdbscan_min_cluster_size = self.get_parameter(
            'clustering_hdbscan_min_cluster_size').get_parameter_value().integer_value
        self.__hdbscan_cluster_selection_epsilon = self.get_parameter(
            'clustering_hdbscan_cluster_selection_epsilon').get_parameter_value().double_value
        self.__num_top_clusters_to_select = self.get_parameter(
            'clustering_num_top_clusters_to_select').get_parameter_value().integer_value
        self.__group_clusters = self.get_parameter(
            'clustering_group_clusters').get_parameter_value().bool_value
        self.__min_points = self.get_parameter(
            'clustering_min_points').get_parameter_value().integer_value

        # Nvblox object clearing parameters
        self.__trigger_aabb_object_clearing = (
            self.get_parameter('trigger_aabb_object_clearing').get_parameter_value().bool_value
        )
        self.__update_esdf_on_request = (
            self.get_parameter('update_esdf_on_request').get_parameter_value().bool_value
        )
        self._object_esdf_clearing_padding = np.asarray(self.get_parameter(
            'object_esdf_clearing_padding').get_parameter_value().double_array_value)

        # Validate topic lengths
        self.__num_cameras = len(self.__depth_image_topics)
        if len(self.__depth_camera_infos) != self.__num_cameras:
            self.get_logger().error(
                'Number of topics in depth_camera_infos does not match depth_image_topics')

        self.__tensor_args = TensorDeviceType(
            device=torch.device('cuda', self.__cuda_device_id))

        # Create subscribers for depth image and robot joint state:
        subscribers = [Subscriber(self, Image, topic, qos_profile=depth_qos)
                       for topic in self.__depth_image_topics]
        subscribers.append(Subscriber(
            self, JointState, self.__joint_states_topic))

        # Subscribe to topics with sync:
        self.__approx_time_sync = ApproximateTimeSynchronizer(
            tuple(subscribers), queue_size=100, slop=self.__time_sync_slop)
        self.__approx_time_sync.registerCallback(
            self.process_depth_and_joint_state)

        # Create subscriber for camera info
        self.__info_subscribers = []
        for idx in range(self.__num_cameras):
            self.__info_subscribers.append(
                self.create_subscription(
                    CameraInfo, self.__depth_camera_infos[idx],
                    lambda msg, index=idx: self.camera_info_cb(msg, index), depth_info_qos)
            )

        # Create publishers:
        self.__object_origin_publisher = self.create_publisher(
            PointStamped, 'object_origin_frame', 10)
        self.__nearby_points_publisher = self.create_publisher(
            PointCloud2, 'nearby_point_cloud', 100)
        self.__clustered_points_publisher = self.create_publisher(
            PointCloud2, 'clustered_point_cloud', 100)
        self.__object_cloud_publisher = self.create_publisher(
            PointCloud2, 'object_point_cloud', 100)
        self.__robot_sphere_markers_publisher = self.create_publisher(
            MarkerArray, 'robot_sphere_markers', 10)

        self.__tf_buffer = Buffer(
            cache_time=rclpy.duration.Duration(seconds=60.0))
        self.__tf_listener = TransformListener(self.__tf_buffer, self)
        self.__br = CvBridge()

        # Create buffers to store data:
        self.__depth_buffers = None
        self.__depth_intrinsics = [None for x in range(self.__num_cameras)]
        self.__robot_pose_camera = [None for x in range(self.__num_cameras)]
        self.__depth_encoding = None

        self.__js_buffer = None
        self.__timestamp = None
        self.__camera_headers = []
        self.__lock = threading.Lock()

        self.__robot_pose_cameras = None
        self.__object_spheres = None
        self.__object_mesh = None

        robot_config = get_robot_config(
            robot_file=self.__robot_file,
            urdf_file_path=self.__urdf_path,
            logger=self.get_logger(),
            object_link_name=self.__object_link_name,
        )

        # Extracting robot's base link name
        self.__cfg_base_link = robot_config['robot_cfg']['kinematics']['base_link']

        # Creating an instance of robot kinematics using config file:
        robot_cfg = RobotConfig.from_dict(
            robot_config, self.__tensor_args)
        self.__kin_model = CudaRobotModel(robot_cfg.kinematics)

        # Maintain the state of the object attachment
        self.__object_state = ObjectState.DETACHED

        # Defining action server
        action_server_cb_grp = MutuallyExclusiveCallbackGroup()
        self.__action_server = ActionServer(
            self,
            AttachObject,
            'attach_object',
            self.attach_object_server_execute_callback,
            goal_callback=self.attach_object_server_goal_callback,
            callback_group=action_server_cb_grp)

        self.__cpu_hdbscan = cpu_HDBSCAN(
            min_samples=self.__hdbscan_min_samples,
            min_cluster_size=self.__hdbscan_min_cluster_size,
            cluster_selection_epsilon=self.__hdbscan_cluster_selection_epsilon
        )

        self.__att_obj_srv_fb_msg = AttachObject.Feedback()
        self.__att_obj_result = AttachObject.Result()

        # Dictionary to store action clients and their callback groups
        self.__action_clients = {}
        self.__callback_groups = {}
        for action_name in self.__action_names:
            callback_group = MutuallyExclusiveCallbackGroup()
            self.__callback_groups[action_name] = callback_group
            self.__action_clients[action_name] = ActionClient(
                self,
                UpdateLinkSpheres,
                action_name,
                callback_group=callback_group
            )

        # Create esdf action client
        if self.__trigger_aabb_object_clearing:
            esdf_service_name = 'nvblox_node/get_esdf_and_gradient'
            esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
            self.__esdf_client = self.create_client(
                EsdfAndGradients, esdf_service_name, callback_group=esdf_service_cb_group
            )
            while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f'Service({esdf_service_name}) not available, waiting again...'
                )
            self.__esdf_req = EsdfAndGradients.Request()

        self.get_logger().info(
            f'Node initialized with {self.__num_cameras} cameras')

    def attach_object_server_goal_callback(
            self,
            attachment_goal: AttachObject.Goal
    ) -> GoalResponse:
        """
        Validate the incoming goal to attach or detach an object.

        Ensure goal data types match the action definition. Reject if the object is already in
        the requested state (attached or detached). Accept the goal if valid, otherwise reject.
        """
        if not self.validate_goal(attachment_goal):
            return GoalResponse.REJECT

        if not self.check_object_state(attachment_goal.attach_object):
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def validate_goal(
            self,
            attachment_goal: AttachObject.Goal
    ) -> bool:
        """
        Validate if the goal type matches the action definition.

        Args
        ----
            attachment_goal (AttachObject.Goal): The goal sent by the action client.

        Returns
        -------
            bool: True if the goal is valid; otherwise, False.

        """
        # Validate the 'attach_object' field
        if not isinstance(attachment_goal.attach_object, bool):
            self.get_logger().error(
                'Value of attach_object sent over action call is invalid. Expecting bool value.'
            )
            return False

        # Validate the 'fallback_radius' field
        if not isinstance(attachment_goal.fallback_radius, float):
            self.get_logger().error(
                'Value of fallback_radius sent over action call is invalid. Expecting float value.'
            )
            return False

        # Validate the 'object_config' field
        if not isinstance(attachment_goal.object_config, Marker):
            self.get_logger().error(
                'Object configuration is invalid. Expecting Marker type.'
            )
            return False

        # Validate the shape of the object (SPHERE, CUBE, MESH_RESOURCE)
        if attachment_goal.object_config.type not in [
                Marker.SPHERE, Marker.CUBE, Marker.MESH_RESOURCE]:
            self.get_logger().error(
                'Object type is invalid. Expected one of SPHERE, CUBE, or MESH_RESOURCE.'
            )
            return False

        # For CUBE and MESH_RESOURCE types, validate pose and scale
        if attachment_goal.object_config.type in [Marker.CUBE, Marker.MESH_RESOURCE]:
            # Validate the object pose (Pose type)
            if not isinstance(attachment_goal.object_config.pose, Pose):
                self.get_logger().error(
                    'Object pose is invalid. Expecting Pose type for CUBE or MESH_RESOURCE.'
                )
                return False

            # Validate the object scale (Vector3 type)
            if not isinstance(attachment_goal.object_config.scale, Vector3):
                self.get_logger().error(
                    'Object scale is invalid. Expecting Vector3 type for CUBE or MESH_RESOURCE.'
                )
                return False

        # Additional validation for custom meshes: check if mesh_resource is set and is a string
        if attachment_goal.object_config.type == Marker.MESH_RESOURCE:
            if not isinstance(attachment_goal.object_config.mesh_resource, str):
                self.get_logger().error(
                    'Mesh resource is invalid. Expecting a string value for MESH_RESOURCE.'
                )
                return False
            if not attachment_goal.object_config.mesh_resource:
                self.get_logger().error(
                    'Mesh resource is missing for custom mesh object.'
                )
                return False
            if not os.path.isfile(attachment_goal.object_config.mesh_resource):
                self.get_logger().error(f'Mesh resource file does not exist: '
                                        f'{attachment_goal.object_config.mesh_resource}')
                return False

        # All validations passed
        return True

    def check_object_state(
            self,
            attach_object: bool
    ) -> bool:
        """
        Prevent redundant attachment or detachment.

        Args
        ----
            attach_object (bool): True to attach, False to detach.

        Returns
        -------
            bool: True if the operation can proceed; otherwise, False.

        """
        if self.__object_state == ObjectState.DETACHED:
            if attach_object:
                return True
            else:
                self.get_logger().error(
                    'Goal to detach object received, but the object is already detached.')
                return False

        elif self.__object_state in [ObjectState.ATTACHED, ObjectState.ATTACHED_FALLBACK]:
            if attach_object:
                self.get_logger().error(
                    'Detach the current object before attempting to attach the new one.')
                return False
            else:
                return True

        else:
            self.get_logger().error(
                f'Unexpected object state: {self.__object_state}')
            return False

    def attach_object_server_execute_callback(
            self,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> AttachObject.Result:
        """
        On goal acceptance, attach or detach the object as commanded.

        If attaching: Obtain object point cloud, generate spheres, and send to other nodes.
        If this fails, use a fallback sphere. Report success or failure.

        If detaching: Clear sphere buffers linked to the object and notify other nodes.
        Report success or abort based on the outcome.

        Args
        ----
            att_obj_srv_goal_handle (ServerGoalHandle): The goal handle that manages the goal.

        Returns
        -------
            AttachObject.Result: The result indicating success or failure of the operation.

        """
        # Check whether to attach or detach object from goal
        attach_object = att_obj_srv_goal_handle.request.attach_object

        # Check whether to attach or detach object from goal
        self.__attached_object_config = att_obj_srv_goal_handle.request.object_config

        if attach_object:
            self.__att_obj_srv_fb_msg.status = 'Executing action call to attach object'
        else:
            self.__att_obj_srv_fb_msg.status = 'Executing action call to detach object'
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        # Initialize timing variables
        start_time = time.time()

        try:
            if attach_object:
                self.handle_attachment(att_obj_srv_goal_handle)
                if self.__object_state != ObjectState.ATTACHED:
                    raise RuntimeError('Failed to attach the object.')
                self.__att_obj_result.outcome = 'Object attached'
            else:
                self.handle_detachment(att_obj_srv_goal_handle)
                if self.__object_state != ObjectState.DETACHED:
                    raise RuntimeError('Failed to detach the object.')
                self.__att_obj_result.outcome = 'Object Detached'

            att_obj_srv_goal_handle.succeed()

        except Exception as e:

            self.__att_obj_srv_fb_msg.status = f'Error during goal execution: {str(e)}'
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

            # Handling fallback if attachment fails
            if attach_object:
                self.__att_obj_srv_fb_msg.status = (
                    'Attempting to attach sphere of radius=fallback_radius'
                )
                att_obj_srv_goal_handle.publish_feedback(
                    self.__att_obj_srv_fb_msg)
                self.handle_fallback(
                    att_obj_srv_goal_handle)
                if self.__object_state == ObjectState.ATTACHED_FALLBACK:
                    self.__att_obj_result.outcome = 'Fallback object attached'
                    att_obj_srv_goal_handle.succeed()
                else:
                    self.__att_obj_result.outcome = 'Fallback object attachment also failed!'
                    att_obj_srv_goal_handle.abort()
            else:
                self.__att_obj_result.outcome = 'Detachment failed'
                att_obj_srv_goal_handle.abort()

        finally:
            # Calculate total execution time
            self.__att_obj_srv_fb_msg.status = (
                f'Total node time: {time.time() - start_time}')
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        return self.__att_obj_result

    def handle_attachment(
            self,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Attempt to attach the object.

        If depth image or joint state data is unavailable, retry a few times. If still unavailable,
        exit without updating the object state to attached.

        If data is available, generate object spheres from the point cloud. On error, keep the
        object detached. If successful, attempt to sync link spheres with other nodes.
        If sync succeeds, update the object state to attached.
        """
        max_retries = 5  # Maximum number of retries
        retries = 0
        success = False  # Initialize success before the loop

        # Keep trying to attach in case of data unavailability
        while retries < max_retries:

            if retries > 0:
                self.__att_obj_srv_fb_msg.status = (
                    f'Attempt #{retries + 1} to attach object...')
                att_obj_srv_goal_handle.publish_feedback(
                    self.__att_obj_srv_fb_msg)

            success, error = self.attach_object(att_obj_srv_goal_handle)

            if success and self.__object_spheres is not None:
                break  # Exit loop if object attachment successfull
            elif error is not None:
                self.__att_obj_srv_fb_msg.status = (
                    f'Attachment failed due to an error: {error}')
                att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)
                break  # Exit loop if error in sphere generation
            else:
                self.__att_obj_srv_fb_msg.status = (
                    'Subscriber(s) didnt recieve data on topics (cam, joint, tf)...')
                att_obj_srv_goal_handle.publish_feedback(
                    self.__att_obj_srv_fb_msg)
                retries += 1
                time.sleep(0.5)  # Wait 0.5s before retrying

        if not success:  # Only check if success is False, which covers both cases
            self.__att_obj_srv_fb_msg.status = (
                'Attachment failed after maximum retries or due to a critical error.')
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)
        else:
            sync_success = self.sync_object_link_spheres_across_nodes(
                att_obj_srv_goal_handle.request.attach_object, att_obj_srv_goal_handle)
            if sync_success:
                self.__object_state = ObjectState.ATTACHED
                self.update_esdf_voxel_grid_for_aabb_clearing()

    def update_esdf_voxel_grid_for_aabb_clearing(self) -> None:
        """
        Update the ESDF voxel by computing the AABBs to clear.

        It sends a request to clear them.
        """
        if not self.__trigger_aabb_object_clearing:
            return

        self.get_logger().info('Calling ESDF service')

        objects_to_clear = self.calculate_aabbs_and_spheres_to_clear()
        aabbs_to_clear_min_m = objects_to_clear[0]
        aabbs_to_clear_size_m = objects_to_clear[1]
        spheres_to_clear_center_m = objects_to_clear[2]
        spheres_to_clear_radius_m = objects_to_clear[3]

        esdf_future = self.send_esdf_request(
            aabbs_to_clear_min_m, aabbs_to_clear_size_m,
            spheres_to_clear_center_m, spheres_to_clear_radius_m)
        while not esdf_future.done():
            time.sleep(0.001)
        response = esdf_future.result()
        if not response.success:
            self.get_logger().error('ESDF request failed! Not clearing object.')

    def calculate_aabbs_and_spheres_to_clear(self) -> List[List]:
        """
        Compute the AABBs to clear by getting the local object frame to world transform.

        Then calculating the vertices of the object. Finally, it transforms the vertices into
        world frame and computes the AABB with padding. If fallback was reached, it instead
        clears the fallback sphere with padding.
        """
        world_T_object = self.calculate_world_T_object()
        if world_T_object is None:
            return [], [], [], []

        # Handle fallback case; simply clear the radius inside the fallback sphere
        if self.__object_state == ObjectState.ATTACHED_FALLBACK:
            centre = self.__object_spheres[:3]
            homogeneous_centre = np.append(centre, 1)
            world_centre = np.matmul(world_T_object, homogeneous_centre)[:3]
            radius = self.__object_spheres[3] + self._object_esdf_clearing_padding[0]
            world_centre = [Point(x=world_centre[0], y=world_centre[1], z=world_centre[2])]
            return [], [], world_centre, [radius]

        # Calculate the vertices in world frame depending on the marker type
        if self.__attached_object_config.type == Marker.CUBE:
            vertices = self.calculate_cuboid_vertices()
            homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
            world_vertices = np.matmul(world_T_object, homogeneous_vertices.T).T[:, :3]
        elif self.__attached_object_config.type == Marker.MESH_RESOURCE:
            mesh = trimesh.load_mesh(self.__attached_object_config.mesh_resource)
            world_vertices = trimesh.transform_points(mesh.vertices, world_T_object)
        elif self.__attached_object_config.type == Marker.SPHERE:
            mesh = self.__object_mesh.get_trimesh_mesh()
            world_vertices = mesh.vertices

        min_corner = np.min(world_vertices, axis=0) - self._object_esdf_clearing_padding / 2
        max_corner = np.max(world_vertices, axis=0) + self._object_esdf_clearing_padding / 2

        aabb_size = np.abs(max_corner - min_corner).tolist()
        min_corner = min_corner.tolist()
        aabb_min = [Point(x=min_corner[0], y=min_corner[1], z=min_corner[2])]
        aabb_size = [Vector3(x=aabb_size[0], y=aabb_size[1], z=aabb_size[2])]
        return aabb_min, aabb_size, [], []

    def calculate_world_T_object(self) -> np.ndarray | None:
        """Compute the transform to go from local object frame to world frame."""
        grasp_pose_object = self.__attached_object_config.pose
        try:
            world_pose_grasp = self.__tf_buffer.lookup_transform(
                'world',
                'grasp_frame',
                rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger.error(f'Could not transform world to grasp_frame: {ex}')
            return None

        grasp_T_object = np.eye(4)
        grasp_T_object[:3, :3] = R.from_quat([
            grasp_pose_object.orientation.x,
            grasp_pose_object.orientation.y,
            grasp_pose_object.orientation.z,
            grasp_pose_object.orientation.w,
        ]).as_matrix()
        grasp_T_object[:3, 3] = np.asarray([
            grasp_pose_object.position.x,
            grasp_pose_object.position.y,
            grasp_pose_object.position.z,
        ])
        world_T_grasp = np.eye(4)
        world_T_grasp[:3, :3] = R.from_quat([
            world_pose_grasp.transform.rotation.x,
            world_pose_grasp.transform.rotation.y,
            world_pose_grasp.transform.rotation.z,
            world_pose_grasp.transform.rotation.w,
        ]).as_matrix()
        world_T_grasp[:3, 3] = np.asarray([
            world_pose_grasp.transform.translation.x,
            world_pose_grasp.transform.translation.y,
            world_pose_grasp.transform.translation.z,
        ])
        world_T_object = np.matmul(world_T_grasp, grasp_T_object)
        return world_T_object

    def calculate_cuboid_vertices(self) -> np.ndarray:
        """
        Calculate the vertices required for the cuboid approach.

        Manually computes them and then offseting them so that it's centered
        at (0, 0, 0).
        """
        width = self.__attached_object_config.scale.x
        height = self.__attached_object_config.scale.y
        depth = self.__attached_object_config.scale.z
        # Manually create the vertices with the lower left corner
        # being (0, 0, 0)
        vertices = np.array([
            [0, 0, 0],
            [0, 0, depth],
            [0, height, 0],
            [0, height, depth],
            [width, 0, 0],
            [width, 0, depth],
            [width, height, 0],
            [width, height, depth],
        ])
        # Offset the cuboid so that centre is (0, 0, 0)
        for i in range(len(vertices)):
            vertices[i][0] -= width / 2
            vertices[i][1] -= height / 2
            vertices[i][2] -= depth / 2
        return vertices

    def send_esdf_request(self, aabbs_to_clear_min_m: List[Point],
                          aabbs_to_clear_size_m: List[Vector3],
                          spheres_to_clear_center_m: List[Point],
                          spheres_to_clear_radius_m: List[float]) -> Future:
        """
        Send the AABBs and spheres to clear to the ESDF client.

        Args
        ----
            aabbs_to_clear_min_m (List[Point]): List of aabb minimum corners to clear
            aabbs_to_clear_size_m (List[Vector3]): List of aabb sizes to clear
            spheres_to_clear_center_m (List[Point]): List of sphere centers to clear
            spheres_to_clear_radius_m (List[float]): List of sphere radii to clear

        Returns
        -------
            Future: Future of the request

        """
        self.__esdf_req.visualize_esdf = False
        self.__esdf_req.update_esdf = self.__update_esdf_on_request
        self.__esdf_req.frame_id = self.__cfg_base_link

        self.__esdf_req.aabbs_to_clear_min_m = aabbs_to_clear_min_m
        self.__esdf_req.aabbs_to_clear_size_m = aabbs_to_clear_size_m
        self.__esdf_req.spheres_to_clear_center_m = spheres_to_clear_center_m
        self.__esdf_req.spheres_to_clear_radius_m = spheres_to_clear_radius_m

        self.get_logger().info(
            '[Object Attachment]: Sending ESDF Update Request for AABB clearing'
        )

        esdf_future = self.__esdf_client.call_async(self.__esdf_req)

        return esdf_future

    def handle_fallback(
            self,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Use a fallback radius sphere for the object.

        Attempt to sync link spheres with other nodes. If sync succeeds, mark as attached;
        otherwise, keep detached.
        """
        self.__att_obj_srv_fb_msg.status = (
            'Attaching pre-defined collision sphere...')
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        fallback_radius = att_obj_srv_goal_handle.request.fallback_radius

        # Initialize a sphere at the object frame's origin with fallback_radius
        self.__object_spheres = np.array([0.0, 0.0, 0.0, fallback_radius])

        sync_success = self.sync_object_link_spheres_across_nodes(
            True, att_obj_srv_goal_handle)
        if sync_success:
            self.__object_state = ObjectState.ATTACHED_FALLBACK
            self.update_esdf_voxel_grid_for_aabb_clearing()

    def handle_detachment(
            self,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Clear sphere buffers linked to the object.

        Attempt to sync link spheres with other nodes. If sync succeeds, mark as detached;
        otherwise, keep attached.
        """
        detachment_time = time.time()
        try:
            self.__kin_model.kinematics_config.detach_object(
                link_name=self.__object_link_name)

            self.__att_obj_srv_fb_msg.status = (
                f'Object detached in {time.time()-detachment_time}s.')
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

            sync_success = self.sync_object_link_spheres_across_nodes(
                False, att_obj_srv_goal_handle)
            if sync_success:
                self.__object_state = ObjectState.DETACHED

        except Exception as e:
            self.__att_obj_srv_fb_msg.status = (
                f'Error in detachment: {str(e)}')
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

    def sync_object_link_spheres_across_nodes(
            self,
            attach_object: bool,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> bool:
        """
        Synchronize object link spheres across nodes via action calls.

        For attachment:
            - Send a flattened list of spheres and the associated link to other nodes.

        For detachment:
            - Notify nodes to clear the sphere buffer for the link.

        Use an action client to send the update goal and wait for completion.
        Return True if all nodes successfully update; otherwise, return False.

        Args
        ----
            attach_object (bool): True to attach, False to detach.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the goal
                sent to the server.

        Returns
        -------
            bool: True if synchronization succeeds; False otherwise.

        """
        update_spheres_goal_msg = UpdateLinkSpheres.Goal()
        update_spheres_goal_msg.attach_object = attach_object
        update_spheres_goal_msg.object_link_name = self.__object_link_name

        if attach_object:
            update_spheres_goal_msg.flattened_sphere_arr = self.__object_spheres.flatten().tolist()

        update_spheres_goal_futures = []
        for action_name, action_client in self.__action_clients.items():
            update_spheres_future = self.send_goal_to_update_spheres(
                action_client, update_spheres_goal_msg, action_name, att_obj_srv_goal_handle)
            update_spheres_goal_futures.append(
                (action_name, update_spheres_future))

        # Wait for the completion of goal at all dependent action servers
        all_succeeded = True
        while update_spheres_goal_futures:
            for action_name, update_spheres_future in update_spheres_goal_futures[:]:
                if update_spheres_future.done():
                    result = update_spheres_future.result()
                    if not result.accepted:
                        all_succeeded = False
                    if result.status == GoalStatus.STATUS_ABORTED:
                        all_succeeded = False
                    update_spheres_goal_futures.remove(
                        (action_name, update_spheres_future))

            time.sleep(0.01)

        return all_succeeded

    def send_goal_to_update_spheres(
            self,
            action_client: ActionClient,
            update_spheres_goal_msg: UpdateLinkSpheres.Goal,
            action_name: str,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> Future:
        """
        Send a goal to update link spheres when the server is ready.

        Set up feedback and response callbacks to track progress and the outcome of the goal.

        Args
        ----
            action_client (ActionClient): Client used to communicate with the action server.
            update_spheres_goal_msg (UpdateLinkSpheres.Goal): Goal message to update link spheres.
            action_name (str): Name of the action being executed.
            att_obj_srv_goal_handle (ServerGoalHandle): Represents the ongoing goal to attach or
                detach the object, managing its state separately from the update link spheres
                action.

        Returns
        -------
            Future: Represents the asynchronous result of the action, including whether the goal
            to update spheres was accepted, rejected, or completed successfully.

        """
        action_client.wait_for_server()
        send_goal_future = action_client.send_goal_async(
            update_spheres_goal_msg,
            feedback_callback=(
                lambda updt_spheres_fb_msg: self.update_spheres_feedback_callback(
                    updt_spheres_fb_msg, action_name, att_obj_srv_goal_handle
                ))
        )
        send_goal_future.add_done_callback(
            lambda update_spheres_future: self.update_spheres_response_callback(
                update_spheres_future, action_name, att_obj_srv_goal_handle)
        )

        return send_goal_future

    def update_spheres_feedback_callback(
            self,
            updt_spheres_fb_msg: UpdateLinkSpheres.Feedback,
            action_name: str,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Handle feedback from the update spheres action.

        Updates the status message based on the received feedback and publishes it
        to the original goal (attach/detach) client.

        Args
        ----
            updt_spheres_fb_msg (UpdateLinkSpheres.Feedback): Feedback message from the
                update spheres action.
            action_name (str): Name of the action sending the feedback.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            None

        """
        self.__att_obj_srv_fb_msg.status = (
            f'Feedback for {action_name}: {updt_spheres_fb_msg.feedback.status}')
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

    def update_spheres_response_callback(
            self,
            update_spheres_future: Future,
            action_name: str,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Handle the response from the update spheres action.

        Checks if the goal was accepted or rejected and publishes feedback accordingly.
        If accepted, sets up a callback to handle the final result of the update action.

        Args
        ----
            update_spheres_future (Future): Future object holding the result of the goal request.
            action_name (str): Name of the action sending the response.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the
                original attach/detach goal, ensuring the correct goal receives feedback.

        Returns
        -------
            None

        """
        update_spheres_goal_handle = update_spheres_future.result()
        if not update_spheres_goal_handle.accepted:
            self.__att_obj_srv_fb_msg.status = (
                f'Goal for {action_name} rejected.')
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)
            return

        self.__att_obj_srv_fb_msg.status = (
            f'Goal for {action_name} accepted.')
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        self.__get_result_future = update_spheres_goal_handle.get_result_async()
        self.__get_result_future.add_done_callback(
            lambda update_spheres_future: self.update_spheres_result_callback(
                update_spheres_future, action_name, att_obj_srv_goal_handle)
        )

    def update_spheres_result_callback(
            self,
            update_spheres_future: Future,
            action_name: str,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Handle the final result of the update spheres action.

        Extracts the result and publishes a status message indicating success or failure.

        Args
        ----
            update_spheres_future (Future): Future containing the result of the update
                spheres action.
            action_name (str): Name of the action for which the result is being handled.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, ensuring feedback is sent to the correct goal.

        Returns
        -------
            None

        """
        result = update_spheres_future.result().result
        self.__att_obj_srv_fb_msg.status = (
            f'Final result for {action_name} action: Success: {result.outcome}')
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

    def filter_depth_buffers(self) -> None:
        """
        Filter the depth buffers by as much as time slop.

        This makes sure object attachment always filters based on latest depth information.
        """
        current_time = self.get_clock().now()  # Get current time
        time_threshold = current_time - rclpy.duration.Duration(
            seconds=self.__filter_depth_buffer_time)
        time_threshold_msg = time_threshold.to_msg()

        filtered_depth_buffers = []
        filtered_depth_timestamps = []

        for i, timestamp in enumerate(self.__depth_timestamps):
            time_in_seconds_one = timestamp.sec + timestamp.nanosec * 1e-9
            time_in_seconds_two = time_threshold_msg.sec + time_threshold_msg.nanosec * 1e-9
            if time_in_seconds_one > time_in_seconds_two:
                filtered_depth_buffers.append(self.__depth_buffers[i])
                filtered_depth_timestamps.append(timestamp)

        self.__depth_buffers = filtered_depth_buffers
        self.__depth_timestamps = filtered_depth_timestamps

    def process_depth_and_joint_state(
            self,
            *msgs: Tuple[Union[Image, JointState]]
    ) -> None:
        """
        Process depth and joint state messages.

        Extracts and stores depth buffers, encoding, and camera headers from Image messages.
        For JointState messages, stores joint names, positions, and the timestamp.

        Args
        ----
            *msgs (Tuple[Union[Image, JointState]]): Variable-length tuple of Image and JointState
                messages to process.

        Returns
        -------
            None

        """
        self.__depth_buffers = []
        self.__depth_encoding = []
        self.__camera_headers = []
        self.__depth_timestamps = []
        for msg in msgs:
            if (isinstance(msg, Image)):
                img = self.__br.imgmsg_to_cv2(msg)
                if msg.encoding == '32FC1':
                    img = 1000.0 * img
                self.__depth_buffers.append(img)
                self.__camera_headers.append(msg.header)
                self.__depth_encoding.append(msg.encoding)
                self.__depth_timestamps.append(msg.header.stamp)  # Store timestamp
            if (isinstance(msg, JointState)):
                self.__js_buffer = {'joint_names': msg.name,
                                    'position': msg.position}
                self.__timestamp = msg.header.stamp

    def camera_info_cb(
            self,
            msg: CameraInfo,
            idx: int
    ) -> None:
        """
        Handle incoming CameraInfo messages.

        Updates the depth intrinsics for the specified camera index.

        Args
        ----
            msg (CameraInfo): The CameraInfo message containing intrinsic parameters.
            idx (int): Index of the camera to update.

        Returns
        -------
            None

        """
        self.__depth_intrinsics[idx] = msg.k

    def attach_object(
            self,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> Tuple[bool, Union[str, None]]:
        """
        Attach an object to the robot based on the object's geometry type.

        The method ensures that valid data from sensors (subscribers) is available, retrieves
        camera transforms, and processes object-specific data for proper attachment. Depending
        on the shape of the object, the function performs different computations, such as handling
        cuboids, custom meshes, and spheres using depth images and joint state data. The final
        outcome is the creation and publication of robot sphere markers and collision spheres.

        Args
        ----
            att_obj_srv_goal_handle (ServerGoalHandle): The goal handle containing necessary data
                for attaching the object and managing the feedback communication with the client.

        Returns
        -------
            Tuple[bool, Union[str, None]]: A tuple containing:
                - bool: True if the object was successfully attached; otherwise, False.
                - Union[str, None]: Error message if an exception occurred during the process,
                    otherwise None.

        """
        # Validate necessary data from subscribers
        if not self.has_valid_subscriber_data():
            return False, None

        # Handle camera transforms
        if not self.retrieve_camera_transforms(att_obj_srv_goal_handle):
            return False, None

        try:
            attached_object_shape = self.__attached_object_config.type
            # Lock to prevent concurrent data modification during read and copy.
            with self.__lock:
                error_msg = 'No depth images in the buffer, please check the time sync slop ' \
                            'parameter. It could be that joint states and images could not ' \
                            'be synced together.'
                if attached_object_shape == Marker.SPHERE:
                    self.filter_depth_buffers()
                    error_msg += f' We filter the depth to only get depth images from the past ' \
                                 f'{self.__filter_depth_buffer_time} seconds'

                if len(self.__depth_buffers) == 0:
                    self.__att_obj_srv_fb_msg.status = error_msg
                    self.get_logger().error(self.__att_obj_srv_fb_msg.status)
                    return False, None

                depth_image = np.copy(np.stack((self.__depth_buffers)))
                intrinsics = np.copy(np.stack(self.__depth_intrinsics))
                js = np.copy(self.__js_buffer['position'])
                j_names = deepcopy(self.__js_buffer['joint_names'])

                # Reset the timestamp and camera headers for the next cycle.
                self.__timestamp = None
                self.__camera_headers = []

            object_frame_origin, joint_states = self.forward_kinematics_computations(
                joint_positions=js,
                joint_names=j_names,
                att_obj_srv_goal_handle=att_obj_srv_goal_handle
            )

            if attached_object_shape in (Marker.CUBE, Marker.MESH_RESOURCE):
                attached_object_frame_sphere_tensor = self.get_spheres_in_attached_object_frame()
                self.attach_object_collision_spheres(
                    spheres_in_attached_object_frame=attached_object_frame_sphere_tensor
                )
            elif attached_object_shape == Marker.SPHERE:
                self.attach_object_collision_spheres_from_point_cloud(
                    att_obj_srv_goal_handle=att_obj_srv_goal_handle,
                    depth_image=depth_image,
                    intrinsics=intrinsics,
                    object_frame_origin=object_frame_origin,
                    joint_states=joint_states)

            if self.__robot_sphere_markers_publisher.get_subscription_count() > 0:
                self.publish_robot_spheres(
                    joint_states=joint_states,
                    att_obj_srv_goal_handle=att_obj_srv_goal_handle
                )

            return True, None  # Indicate success

        except Exception as e:

            self.__att_obj_srv_fb_msg.status = f'{traceback.format_exc()}'
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

            return False, str(e)

    def attach_object_collision_spheres_from_point_cloud(
            self, att_obj_srv_goal_handle: ServerGoalHandle,
            depth_image: np.ndarray,
            intrinsics: CameraInfo,
            object_frame_origin: Pose,
            joint_states: JointState
    ):
        """
        Generate spheres for object by processing the depth image.

        It takes as input the depth image, intrinsics and joint states as input to construct the
        collision spheres which are attached to the object_frame on the robot.
        """
        points_wrt_robot_base = self.get_point_cloud_from_depth(
            depth_image=depth_image,
            intrinsics=intrinsics,
            att_obj_srv_goal_handle=att_obj_srv_goal_handle
        )

        points_near_object_frame_origin = self.get_points_around_center(
            all_points=points_wrt_robot_base,
            center=object_frame_origin,
            att_obj_srv_goal_handle=att_obj_srv_goal_handle
        )

        object_point_cloud = self.get_object_point_cloud(
            point_cloud=points_near_object_frame_origin,
            object_frame_origin=object_frame_origin,
            bypass_clustering=self.__bypass_clustering,
            num_top_clusters_to_select=self.__num_top_clusters_to_select,
            group_clusters=self.__group_clusters,
            min_points=self.__min_points,
            att_obj_srv_goal_handle=att_obj_srv_goal_handle
        )

        object_mesh = self.get_object_mesh(
            object_point_cloud=object_point_cloud,
            att_obj_srv_goal_handle=att_obj_srv_goal_handle
        )
        self.__object_mesh = object_mesh
        self.generate_object_collision_spheres_from_mesh(
            joint_states=joint_states,
            object_mesh=object_mesh,
            att_obj_srv_goal_handle=att_obj_srv_goal_handle
        )

    def has_valid_subscriber_data(
            self
    ) -> bool:
        """
        Check if the necessary subscriber data is available.

        Verifies that all depth intrinsics are valid, camera headers are present,
        and a timestamp exists.

        Returns
        -------
            bool: True if all necessary subscriber data is available; otherwise, False.

        """
        if (
            not all(isinstance(intrinsic, np.ndarray)
                    for intrinsic in self.__depth_intrinsics)
            or len(self.__camera_headers) == 0
            or self.__timestamp is None
        ):
            self.get_logger().error('Could not find valid subscriber data to attach object')
            return False
        return True

    def retrieve_camera_transforms(
            self,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> bool:
        """
        Read and process camera transforms.

        If camera transforms are not yet available, attempts to retrieve them. Publishes feedback
        based on success or failure.

        Args
        ----
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, ensuring feedback is sent correctly.

        Returns
        -------
            bool: True if transforms were successfully retrieved for all cameras; otherwise, False.

        """
        if self.__robot_pose_cameras is None:
            with self.__lock:
                camera_headers = deepcopy(self.__camera_headers)

            for i in range(self.__num_cameras):
                if self.__robot_pose_camera[i] is None:
                    if not self.lookup_camera_transform(i, camera_headers[i],
                                                        att_obj_srv_goal_handle):
                        continue

            if None not in self.__robot_pose_camera:
                self.__robot_pose_cameras = CuPose.cat(
                    self.__robot_pose_camera)
                self.__att_obj_srv_fb_msg.status = 'Received TF from cameras to robot'
                att_obj_srv_goal_handle.publish_feedback(
                    self.__att_obj_srv_fb_msg)
            else:
                self.__att_obj_srv_fb_msg.status = 'Failed to retrieve transform for any camera'
                self.get_logger().error(self.__att_obj_srv_fb_msg.status)
                att_obj_srv_goal_handle.publish_feedback(
                    self.__att_obj_srv_fb_msg)
                return False

        return True

    def lookup_camera_transform(
            self,
            index: int,
            camera_header: Header,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> bool:
        """
        Attempt to look up and store a camera transform.

        Tries to retrieve the transform between the camera and the robot base link. On success,
        stores the transform; on failure, logs an error and publishes feedback.

        Args
        ----
            index (int): Index of the camera for which the transform is being looked up.
            camera_header (Header): The header of the camera containing the frame ID.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, ensuring feedback is sent correctly.

        Returns
        -------
            bool: True if the transform was successfully retrieved; otherwise, False.

        """
        try:
            t = self.__tf_buffer.lookup_transform(
                self.__cfg_base_link,
                camera_header.frame_id,
                self.__timestamp,
                rclpy.duration.Duration(seconds=self.__tf_lookup_duration),
            )
            self.__robot_pose_camera[index] = CuPose.from_list(
                [
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                    t.transform.rotation.w,
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                ]
            )
            return True
        except TransformException as ex:
            self.__att_obj_srv_fb_msg.status = (
                f'Could not transform {camera_header.frame_id} to {self.__cfg_base_link}: {ex}'
            )
            self.get_logger().error(self.__att_obj_srv_fb_msg.status)
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)
            return False

    def get_spheres_in_attached_object_frame(
            self,
    ) -> torch.Tensor:
        """
        Generate and return the collision spheres in the attached object's frame of reference.

        Calculates the position of collision spheres relative to the attached object's
        frame, taking into account the transformations between the detected object, gripper, and
        attached object frames. The function also concatenates the computed sphere positions and
        their radii into a tensor for further processing.

        Args
        ----
            None

        Returns
        -------
            torch.Tensor: A tensor containing the sphere centers and radii, with the sphere
            positions transformed to the attached object's frame.

        """
        grasp_pose = self.__attached_object_config.pose
        gripper_frame_pose_object_frame = [grasp_pose.position.x,
                                           grasp_pose.position.y,
                                           grasp_pose.position.z,
                                           grasp_pose.orientation.w,
                                           grasp_pose.orientation.x,
                                           grasp_pose.orientation.y,
                                           grasp_pose.orientation.z]

        attached_object_frame_pose_gripper_frame = (
            self.get_gripper_to_attached_object_frame_transform()
        )

        detected_object_frame_position_sphere_centers, radii_spheres = (
            self.get_collision_spheres_in_detected_object_frame(
                pose=gripper_frame_pose_object_frame
            )
        )

        attached_object_frame_position_sphere_centers = (
            attached_object_frame_pose_gripper_frame.transform_points(
                points=detected_object_frame_position_sphere_centers
            )
        )

        attached_object_frame_sphere_tensor = torch.cat(
            (attached_object_frame_position_sphere_centers, radii_spheres),
            dim=1
        )

        return attached_object_frame_sphere_tensor

    def get_gripper_to_attached_object_frame_transform(
            self
    ) -> CuPose:
        """
        Compute the transform from the gripper frame to the attached object's frame.

        Retrieves the poses of the gripper and attached object frames relative to the
        robot base frame using the robot's kinematic model based on default joint positions.
        The relative pose of the gripper frame relative to the attached object frame is then
        calculated and returned.

        Args
        ----
            None

        Returns
        -------
            CuPose: The transformation matrix that defines the pose of the gripper frame relative
            to the attached object frame.

        """
        object_frame = self.__object_link_name
        gripper_frame = self.__gripper_frame_name

        # Type hint | self.__kin_model: CudaRobotModel
        default_joint_positions = self.__kin_model.retract_config

        # Type hint | robot_state: CudaRobotModelState
        robot_state = self.__kin_model.compute_kinematics_from_joint_position(
            joint_position=default_joint_positions
        )

        link_poses = robot_state.link_poses
        robot_base_frame_pose_object_frame = link_poses[object_frame]
        robot_base_frame_pose_gripper_frame = link_poses[gripper_frame]

        object_frame_pose_robot_base_frame = robot_base_frame_pose_object_frame.inverse()

        object_frame_pose_gripper_frame = object_frame_pose_robot_base_frame.multiply(
            robot_base_frame_pose_gripper_frame
        )

        return object_frame_pose_gripper_frame

    def get_detected_object_to_gripper_frame_transform(
            self
    ) -> CuPose:
        """
        Compute the transform from the detected object's frame to the gripper frame.

        Retrieves the grasp pose, which defines the pose of the detected object's frame
        relative to the gripper frame. The position and orientation are extracted from
        the grasp pose and used to construct a CuPose object.

        Args
        ----
            None

        Returns
        -------
            CuPose: The transformation matrix that defines the pose of the detected object frame
            relative to the gripper frame.

        """
        grasp_pose = self.__attached_object_config.pose

        gripper_frame_pose_detected_object_frame = CuPose(
            position=torch.tensor(
                [grasp_pose.position.x,
                    grasp_pose.position.y,
                    grasp_pose.position.z],
                device=torch.device('cuda', self.__cuda_device_id)),
            quaternion=torch.tensor(
                [grasp_pose.orientation.x,
                    grasp_pose.orientation.y,
                    grasp_pose.orientation.z,
                    grasp_pose.orientation.w],
                device=torch.device('cuda', self.__cuda_device_id))
        )

        return gripper_frame_pose_detected_object_frame

    def get_collision_spheres_in_detected_object_frame(
            self,
            pose: List[float]
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute collision spheres in the detected object's frame.

        Depending on the shape of the attached object (cuboid or custom mesh), creates
        an appropriate object representation using the provided pose. Then generates
        collision spheres within the object's frame by invoking
        `generate_spheres_in_object_frame`. Returns the positions of the spheres' centers
        and their radii.

        Args
        ----
            pose (List[float]): The pose of the detected object.

        Returns
        -------
            Tuple[torch.Tensor, List[float]]: A tuple containing the positions of the spheres'
            centers in the object frame and the radii of the spheres.

        """
        attached_object_shape = self.__attached_object_config.type

        if attached_object_shape == Marker.CUBE:

            dims_array = [self.__attached_object_config.scale.x,
                          self.__attached_object_config.scale.y,
                          self.__attached_object_config.scale.z]

            attached_object = CuCuboid(
                name='blind_cuboid',
                pose=pose,
                dims=dims_array
            )

        if attached_object_shape == Marker.MESH_RESOURCE:

            attached_object = CuMesh(
                name='mesh_object_for_object_attachment',
                pose=pose,
                file_path=self.__attached_object_config.mesh_resource
            )

        object_frame_position_sphere_centers, radii_spheres = (
            self.generate_spheres_in_object_frame(
                attached_object=attached_object
            )
        )

        return object_frame_position_sphere_centers, radii_spheres

    def generate_spheres_in_object_frame(
            self,
            attached_object: CuObstacle
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collision spheres within the object's frame.

        Uses the provided attached object to generate collision spheres. The positions of the
        spheres' centers and their radii are collected and returned as tensors.

        Args
        ----
            attached_object (CuObstacle): The object for which to generate bounding spheres.

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - torch.Tensor: Tensor of shape (n_spheres, 3) representing the positions of the
                    spheres' centers in the object's frame.
                - torch.Tensor: Tensor of shape (n_spheres, 1) representing the radii of spheres.

        """
        cu_spheres = attached_object.get_bounding_spheres(
            n_spheres=self.__object_attachment_n_spheres,
            surface_sphere_radius=self.__surface_sphere_radius
        )

        object_frame_position_sphere_centers = torch.tensor(
            [cu_sphere.pose[:3] for cu_sphere in cu_spheres],
            dtype=torch.float32,
            device=torch.device('cuda', self.__cuda_device_id)
        )
        radii_spheres = torch.tensor(
            [cu_sphere.radius for cu_sphere in cu_spheres],
            dtype=torch.float32,
            device=torch.device('cuda', self.__cuda_device_id)
        )

        radii_spheres = radii_spheres.view(-1, 1)

        return object_frame_position_sphere_centers, radii_spheres

    def attach_object_collision_spheres(
            self,
            spheres_in_attached_object_frame: torch.Tensor
    ) -> None:
        """
        Update the kinematic model with collision spheres in the attached object's frame.

        This method takes the provided collision spheres' positions and radii in the attached
        object's frame and updates the kinematic configuration to reflect these spheres.

        Args
        ----
            spheres_in_attached_object_frame (torch.Tensor): A tensor containing the positions
                and radii of the collision spheres in the attached object's frame.

        Returns
        -------
            None

        """
        self.__kin_model.kinematics_config.update_link_spheres(
            link_name=self.__object_link_name,
            sphere_position_radius=spheres_in_attached_object_frame
        )

        link_spheres_object_frame = self.__kin_model.kinematics_config.get_link_spheres(
            link_name=self.__object_link_name)

        self.__object_spheres = link_spheres_object_frame.cpu().numpy()

    def get_point_cloud_from_depth(
            self,
            depth_image: torch.Tensor,
            intrinsics: torch.Tensor,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> torch.Tensor:
        """
        Generate a 3D point cloud from the segmented depth image using camera intrinsics.

        Converts the depth image into a point cloud in the robot's base frame by leveraging the
        camera intrinsics & robot's pose. The point cloud is extracted using the camera observation
        model and transformed into the robot base frame.

        Args
        ----
            depth_image (torch.Tensor): A tensor representing the segmented depth image.
            intrinsics (torch.Tensor): Camera intrinsics used for projecting the depth image.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            torch.Tensor: A 3D point cloud in the robot's base frame.

        """
        time_start = time.time()

        depth_image = self.__tensor_args.to_device(
            depth_image.astype(np.float32))
        depth_image = depth_image.view(
            self.__num_cameras, depth_image.shape[-2], depth_image.shape[-1])

        intrinsics = (
            self.__tensor_args.to_device(
                intrinsics).view(self.__num_cameras, 3, 3)
        )

        cam_obs = CameraObservation(
            depth_image=depth_image,
            pose=self.__robot_pose_cameras,
            intrinsics=intrinsics
        )

        point_cloud = cam_obs.get_pointcloud()

        # robot_base_points: point wrt robot base frame
        points_wrt_robot_base = (
            self.__robot_pose_cameras.batch_transform_points(point_cloud)
        )

        points_wrt_robot_base = points_wrt_robot_base.contiguous().view(-1, 3)

        self.__att_obj_srv_fb_msg.status = (
            f'Extracted point cloud from segmented depth image in {time.time() - time_start}s.'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        return points_wrt_robot_base

    def forward_kinematics_computations(
        self,
        joint_positions: np.ndarray,
        joint_names: list[str],
        att_obj_srv_goal_handle: ServerGoalHandle
    ) -> tuple[torch.Tensor, JointState]:
        """
        Compute the forward kinematics for the manipulator and obtain the object frame origin.

        Uses the joint positions and joint names to compute the forward kinematics of the
        manipulator and determine the position of the object frame origin. Optionally publishes
        the object position if a subscriber is active.

        Args
        ----
            joint_positions (np.ndarray): The angular positions of the manipulator's joints.
            joint_names (list): The names of the joints corresponding to the positions.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            tuple[torch.Tensor, JointState]: A tuple containing the object's frame origin as a
                tensor and the current joint states.

        """
        start_time = time.time()

        joint_states = CuJointState.from_numpy(
            position=joint_positions,
            joint_names=joint_names,
            tensor_args=self.__tensor_args).unsqueeze(0)

        active_jnames = self.__kin_model.joint_names
        joint_states = joint_states.get_ordered_joint_state(active_jnames)
        out = self.__kin_model.get_state(joint_states.position)
        object_frame_origin = out.link_poses[self.__object_link_name].position

        self.__att_obj_srv_fb_msg.status = (
            f'Obtained object position using forward kinematics in {time.time() - start_time}s.'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        if self.__object_origin_publisher.get_subscription_count() > 0:
            self.publish_point(object_frame_origin)

        return object_frame_origin, joint_states

    def get_points_around_center(
            self,
            all_points: torch.Tensor,
            center: torch.Tensor,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> torch.Tensor:
        """
        Extract points within a specified radius around the object center.

        Computes the Euclidean distance between each point and the object center and selects points
        within a defined search radius. Optionally publishes the nearby points if a subscriber
        is active.

        Args
        ----
            all_points (torch.Tensor): The entire point cloud from which points are filtered.
            center (torch.Tensor): The object frame's origin used to calculate distances.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            torch.Tensor: A tensor of points located near the object center.

        """
        start_time = time.time()

        distances = torch.norm(all_points[:, :] - center, dim=1)
        indices = distances <= self.__search_radius
        nearby_points = all_points[indices]

        self.__att_obj_srv_fb_msg.status = (
            f'Obtained points near object center in {time.time()-start_time}s'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        if self.__nearby_points_publisher.get_subscription_count() > 0:
            self.publish_point_cloud(
                nearby_points,
                self.__nearby_points_publisher
            )

        return nearby_points

    def get_object_point_cloud(
            self,
            point_cloud: torch.Tensor,
            object_frame_origin: torch.Tensor,
            bypass_clustering: bool,
            num_top_clusters_to_select: int,
            group_clusters: bool,
            min_points: int,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> torch.Tensor:
        """
        Obtain the object point cloud either by clustering or bypassing clustering.

        If clustering is enabled, function applies clustering algorithm to the input point cloud to
        detect clusters. The most relevant cluster(s) are selected using a heuristic. If clustering
        is bypassed, the input point cloud is used directly. The object point cloud can optionally
        be published if a subscriber is active.

        Args
        ----
            point_cloud (torch.Tensor): The point cloud representing the scene expressed with
                respect to robot base frame.
            bypass_clustering (bool): Whether to bypass clustering & use the point cloud directly.
            min_samples (int): Minimum samples for HDBSCAN clustering.
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            cluster_selection_epsilon (float): Epsilon value for cluster selection in HDBSCAN.
            object_frame_origin (torch.Tensor): The reference point (object frame origin) for
                cluster selection.
            num_top_clusters_to_select (int): The number of top/largest clusters to select.
            group_clusters (bool): Whether to group multiple clusters into one.
            min_points (int): Minimum number of points required for a cluster to be valid.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            torch.Tensor: The filtered point cloud representing the object.

        """
        if bypass_clustering:
            object_point_cloud = point_cloud
        else:
            start_time = time.time()

            # The CPU implementation can be switched to a GPU-based cuML approach once Jetson's
            # binary support is restored. Since cuML uses CuPy arrays (which are GPU-accelerated),
            # downstream calculations remain in CuPy for efficiency. This avoids replacing CuPy
            # with torch.tensor and prevents future rework when cuML is reintroduced.
            points_cupy = cp.asarray(point_cloud)
            points_cpu = point_cloud.cpu().numpy()
            labels_cpu = self.__cpu_hdbscan.fit_predict(points_cpu)
            labels_cupy = cp.asarray(labels_cpu)

            self.__att_obj_srv_fb_msg.status = (
                f'CPU HDBSCAN clustering completed in {time.time()-start_time}s')
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

            object_point_cloud = self.cluster_selection_heuristic(
                points=points_cupy,
                labels=labels_cupy,
                ref_point=object_frame_origin,
                num_top_clusters_to_select=num_top_clusters_to_select,
                group_clusters=group_clusters,
                min_points=min_points,
                att_obj_srv_goal_handle=att_obj_srv_goal_handle
            )

        if self.__object_cloud_publisher.get_subscription_count() > 0:
            self.publish_point_cloud(
                object_point_cloud, self.__object_cloud_publisher)

        return object_point_cloud

    def cluster_selection_heuristic(
            self,
            points: cp.ndarray,
            labels: cp.ndarray,
            ref_point: torch.Tensor,
            num_top_clusters_to_select: int,
            group_clusters: bool,
            min_points: int,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> torch.Tensor:
        """
        Apply a heuristic to filter, select, and group clusters based on distance and size.

        Filters the clusters based on a minimum number of points, calculates their distance from
        a reference point, and selects the largest clusters. Optionally groups the selected
        clusters into one point cloud.

        Args
        ----
            points (cp.ndarray): An array of points (XYZ coordinates).
            labels (cp.ndarray): An array of cluster labels corresponding to the points.
            ref_point (torch.Tensor): A tensor representing the reference point.
            num_top_clusters_to_select (int): The number of top largest clusters to select.
            group_clusters (bool): Whether to group the selected clusters into a single cluster.
            min_points (int): The minimum number of points required for a cluster to be valid.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            torch.Tensor: A tensor representing the selected or grouped clusters.

        """
        total_start_time = time.time()
        start_time = time.time()

        filtered_clusters = self.get_filtered_labeled_clusters(
            points, labels, min_points)

        self.__att_obj_srv_fb_msg.status = (
            f' Heuristic Step-1 | Got Filtered labeled clusters in {time.time()-start_time}s'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        # Publish the filtered clusters as a point cloud
        if self.__clustered_points_publisher.get_subscription_count() > 0:
            publish_time = time.time()
            self.publish_all_clusters(filtered_clusters)
            self.__att_obj_srv_fb_msg.status = (
                f' Published all clusters in {time.time()-publish_time}s'
            )
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        if not filtered_clusters:
            self.__att_obj_srv_fb_msg.status = 'No clusters found after filtering'
            att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)
            return None

        start_time = time.time()

        cluster_distances = self.calculate_cluster_distances(
            filtered_clusters, ref_point)

        self.__att_obj_srv_fb_msg.status = (
            ' Heuristic Step-2 | Calculated cluster distanced from object frame'
            f'in {time.time()-start_time}s'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        start_time = time.time()

        selected_clusters = self.sort_and_select_clusters(
            cluster_distances, filtered_clusters, num_top_clusters_to_select)

        self.__att_obj_srv_fb_msg.status = (
            f' Heuristic Step-3 | Sort and select clusters in {time.time()-start_time}s'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        # If group_clusters is True, combine the top clusters into one
        if group_clusters:
            object_cluster = cp.concatenate(selected_clusters, axis=0)
        # Otherwise, return the largest cluster
        else:
            object_cluster = max(
                selected_clusters, key=lambda cluster: cluster.shape[0])

        object_point_cloud = torch.as_tensor(object_cluster, device='cuda:0')

        self.__att_obj_srv_fb_msg.status = (
            'Total Heuristic time | Selected cluster as per heuristic'
            f'in {time.time()-total_start_time}s'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        return object_point_cloud

    def get_filtered_labeled_clusters(
            self,
            points: cp.ndarray,
            labels: cp.ndarray,
            min_points: int
    ) -> Dict[int, cp.ndarray]:
        """
        Filter clusters based on a minimum number of points.

        Filters out clusters that have fewer points than the specified threshold.

        Args
        ----
            points (cp.ndarray): An array of points (XYZ coordinates).
            labels (cp.ndarray): An array of cluster labels corresponding to the points.
            min_points (int): The minimum number of points required for a cluster to be valid.

        Returns
        -------
            Dict[int, cp.ndarray]: A dictionary mapping valid cluster labels (int) to their
                respective points (XYZ coordinates).

        """
        unique_labels = cp.unique(labels)

        clusters = {}
        for label in unique_labels:
            label = int(label)  # Ensure label is a scalar integer
            if label != -1:  # Ignore noise points
                cluster_points = points[labels == label]
                if cluster_points.shape[0] > min_points:
                    clusters[label] = cluster_points

        return clusters

    def calculate_cluster_distances(
            self,
            clusters: Dict[int, cp.ndarray],
            ref_point: torch.Tensor
    ) -> Dict[int, float]:
        """
        Calculate the distance from a reference point to each cluster.

        The distance is calculated by finding the Euclidean distance between the reference point
        and the mean of each cluster.

        Args
        ----
            clusters (Dict[int, cp.ndarray]): A dictionary mapping cluster labels (int) to arrays
                of points (XYZ coordinates).
            ref_point (torch.Tensor): A tensor representing the reference point in space.

        Returns
        -------
            Dict[int, float]: A dictionary mapping each cluster label (int) to its distance from
                the reference point.

        """
        ref_point_gpu = cp.asarray(ref_point.flatten())
        cluster_distances = {}
        for label, cluster in clusters.items():
            cluster_mean = cp.mean(cluster, axis=0)
            distance = cp.linalg.norm(cluster_mean - ref_point_gpu)
            cluster_distances[label] = distance

        return cluster_distances

    def sort_and_select_clusters(
            self,
            cluster_distances: Dict[int, float],
            clusters: Dict[int, cp.ndarray],
            num_top_clusters: int
    ) -> List[cp.ndarray]:
        """
        Sort clusters by their distance from a reference point and select the largest N clusters.

        The clusters are first sorted based on their distance to the reference point, and then the
        top N clusters in terms of number of points are selected.

        Args
        ----
            cluster_distances (Dict[int, float]): A dictionary mapping cluster labels (int) to
                their calculated distance from a reference point.
            clusters (Dict[int, cp.ndarray]): A dictionary mapping each cluster label (int) to an
                array of points (XYZ coordinates).
            num_top_clusters (int): The number of top clusters to select based on distance.

        Returns
        -------
            List[cp.ndarray]: A list of the top N clusters, where each cluster is represented by an
                array of points (XYZ coordinates).

        """
        sorted_cluster_distances = sorted(
            cluster_distances.items(), key=lambda item: item[1])
        selected_top_clusters = sorted_cluster_distances[:num_top_clusters]
        selected_clusters = [clusters[label]
                             for label, _ in selected_top_clusters]

        return selected_clusters

    def publish_all_clusters(
            self,
            filtered_clusters: Dict[int, cp.ndarray]
    ) -> None:
        """
        Publish colored point clouds for each cluster using a ROS PointCloud2 message.

        Initializes the header with the current timestamp and sets the frame ID to the base link
        of the manipulator. The function sets up the point fields for XYZ and RGB values, where
        the XYZ values represent the positions of points in each cluster and RGB values represent
        the color assigned to each cluster.

        The colors are assigned based on the order in which clusters are processed.

        Args
        ----
            filtered_clusters (Dict[int, cp.ndarray]): A dictionary mapping each cluster label
                (int) to an array of XYZ coordinates representing the points in the cluster.

        Returns
        -------
            None

        """
        if not filtered_clusters:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.__cfg_base_link

        fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12,
                       datatype=PointField.UINT32, count=1),
        ]

        unique_labels = list(filtered_clusters.keys())
        color_map = self.get_cluster_colors(len(unique_labels))
        cluster_colors = {label: color_map[i]
                          for i, label in enumerate(unique_labels)}

        points_with_color = []

        # Convert all cupy arrays to numpy arrays and then to lists outside the loop
        filtered_clusters_list = {label: points.get().tolist()
                                  for label, points in filtered_clusters.items()}

        for label, points_list in filtered_clusters_list.items():
            rgb = cluster_colors[label]
            r, g, b = rgb
            rgb_int = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            for point in points_list:
                x, y, z = point
                points_with_color.append([x, y, z, rgb_int])

        cloud_data = point_cloud2.create_cloud(
            header, fields, points_with_color)
        self.__clustered_points_publisher.publish(cloud_data)

    def get_cluster_colors(
            self,
            num_clusters: int
    ) -> Dict[int, tuple]:
        """
        Generate a color map for clusters, assigning each cluster a unique RGB color.

        Creates a dictionary where each cluster (represented by an integer) is mapped to an RGB
        tuple. Predefined colors (red, green, blue, yellow, etc.) are used for the first few
        clusters, and random colors are generated for additional clusters if the number of clusters
        exceeds the predefined colors list.

        Args
        ----
            num_clusters (int): The number of clusters to be colored. If more than the available
                predefined colors, random colors are generated for the additional clusters.

        Returns
        -------
            Dict[int, tuple]: A dictionary mapping each cluster to an RGB color.

        """
        fixed_colors = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255)  # Cyan
        ]

        color_map = {}
        for i in range(min(num_clusters, len(fixed_colors))):
            color_map[i] = fixed_colors[i]

        np.random.seed(42)  # For reproducibility
        used_colors = set(fixed_colors)
        for i in range(len(fixed_colors), num_clusters):
            while True:
                random_color = (np.random.randint(0, 255), np.random.randint(
                    0, 255), np.random.randint(0, 255))
                if random_color not in used_colors:
                    color_map[i] = random_color
                    used_colors.add(random_color)
                    break

        return color_map

    def publish_point(
            self,
            object_frame_origin_tensor: torch.Tensor
    ) -> None:
        """
        Publish the origin point of the object frame using a ROS PointStamped message.

        Takes the object's origin point, expressed in the robot's base link frame, and publishes it
        using a ROS `PointStamped` message. The point is initialized with the necessary headers
        (including the timestamp) and then published.

        Args
        ----
            object_frame_origin_tensor (torch.Tensor): A 3D point representing the origin of the
                object frame, expressed in the robot's base link frame.

        Returns
        -------
            None

        """
        self.point = object_frame_origin_tensor.cpu().numpy().flatten()

        point_msg = PointStamped()
        point_msg.header = Header()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = self.__cfg_base_link
        point_msg.point.x = float(self.point[0])
        point_msg.point.y = float(self.point[1])
        point_msg.point.z = float(self.point[2])

        self.__object_origin_publisher.publish(point_msg)

    def publish_point_cloud(
            self,
            point_cloud: torch.Tensor,
            publisher: Publisher
    ) -> None:
        """
        Publish a point cloud using a ROS PointCloud2 message.

        Converts the input point cloud (a tensor containing x, y, z coordinates) into a
        ROS `PointCloud2` message. It attaches a timestamp and sets the frame ID to the
        robot's base link. The point cloud data is then published via the provided publisher.

        Args
        ----
            point_cloud (torch.Tensor): A tensor representing the 3D point cloud with x, y, z
                coordinates.
            publisher (Publisher): A ROS publisher object used to publish `PointCloud2` message.

        Returns
        -------
            None

        """
        # Initialize PointCloud2 message
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = self.__cfg_base_link
        point_cloud_msg.height = 1
        point_cloud_msg.width = point_cloud.shape[0]

        # Define PointField layout
        point_cloud_msg.fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 12  # 3 fields * 4 bytes per field
        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
        point_cloud_msg.is_dense = False

        # Serialize point cloud data
        point_cloud_msg.data = b''.join(
            struct.pack('fff', *point) for point in point_cloud.cpu().numpy()
        )

        # Publish the point cloud
        publisher.publish(point_cloud_msg)

    def get_object_mesh(
            self,
            object_point_cloud: torch.Tensor,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> CuMesh:
        """
        Generate a 3D mesh from the object's point cloud using the CuMesh class.

        Converts the input point cloud (a torch tensor) into a numpy array and uses the
        `from_point_cloud` method of the `CuMesh` class from `curobo.geom.types` to generate a
        3D mesh. The mesh is created using the marching cubes algorithm, which approximates the
        object's surface by forming a convex hull around the point cloud.

        Args
        ----
            object_point_cloud (torch.Tensor): A tensor representing the 3D object point cloud,
                expressed with respect to robot base frame
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            CuMesh: A mesh object of type `CuMesh`, representing the object's surface generated
                from the point cloud data.

        """
        start_time = time.time()

        object_point_cloud_np = object_point_cloud.cpu().numpy()

        object_mesh = CuMesh.from_pointcloud(pointcloud=object_point_cloud_np)

        self.__att_obj_srv_fb_msg.status = f'Generated object mesh in {time.time()-start_time}s'
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

        return object_mesh

    def generate_object_collision_spheres_from_mesh(
            self,
            joint_states: JointState,
            object_mesh: CuMesh,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Attach an external object to the robot and generate collision spheres.

        Uses the robot's forward kinematics model to attach the external object (in the form of a
        mesh) to a specified link. Then retrieves the collision sphere information for the link,
        which includes the coordinates (x, y, z) and radii for each sphere in the object frame.

        Args
        ----
            joint_states (JointState): Contains the state information (e.g., positions, velocities,
                accelerations) for each joint in the configuration space of the manipulator.
            object_mesh (CuMesh): The mesh representing the external object to be attached. All
                coordinates of the mesh vertices are expressed with respect to robot base frame.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            None

        """
        start_time = time.time()

        self.__kin_model.attach_external_objects_to_robot(
            joint_state=joint_states,
            external_objects=[object_mesh],
            surface_sphere_radius=self.__surface_sphere_radius,
            link_name=self.__object_link_name
        )

        link_spheres_object_frame = self.__kin_model.kinematics_config.get_link_spheres(
            link_name=self.__object_link_name)

        self.__object_spheres = link_spheres_object_frame.cpu().numpy()

        self.__att_obj_srv_fb_msg.status = (
            f'Generated object collision spheres in {time.time()-start_time}s'
        )
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)

    def publish_robot_spheres(
            self,
            joint_states: JointState,
            att_obj_srv_goal_handle: ServerGoalHandle
    ) -> None:
        """
        Publish the robot's collision spheres based on joint state information.

        Uses the forward kinematics model and the state information from `JointState` (e.g.,
        positions, velocities, accelerations) to obtain robot's collision spheres, approximating
        its geometry in the robot base frame. These spheres are published using an API as a marker
        array, which can be visualized in RViz.

        Args
        ----
            joint_states (JointState): Contains the state information (e.g., positions, velocities,
                accelerations) for each joint in the manipulator.
            att_obj_srv_goal_handle (ServerGoalHandle): Manages the state of the original
                attach/detach goal, to which the feedback is published.

        Returns
        -------
            None

        """
        start_time = time.time()

        link_pos, link_quat, _, _, _, _, robot_spheres_robot_base_frame = (
            self.__kin_model.forward(
                q=joint_states.position,
                link_name=self.__object_link_name
            )
        )

        m_arr = get_spheres_marker(
            robot_spheres=robot_spheres_robot_base_frame.cpu().numpy().squeeze(0),
            base_frame=self.__cfg_base_link,
            time=self.get_clock().now().to_msg(),
            rgb=[1.0, 0.0, 0.0, 1.0],
        )

        self.__robot_sphere_markers_publisher.publish(m_arr)

        self.__att_obj_srv_fb_msg.status = f'Published object spheres in {time.time()-start_time}s'
        att_obj_srv_goal_handle.publish_feedback(self.__att_obj_srv_fb_msg)


def main(args=None):

    rclpy.init(args=args)
    attach_object_server = AttachObjectServer()
    executor = MultiThreadedExecutor()
    executor.add_node(attach_object_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        attach_object_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    attach_object_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
