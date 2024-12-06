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

from ament_index_python.packages import get_package_share_directory
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid
from curobo.geom.types import Cylinder
from curobo.geom.types import Mesh
from curobo.geom.types import Sphere
from curobo.geom.types import VoxelGrid as CuVoxelGrid
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState as CuJointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path
from curobo.util_file import join_path
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.wrap.reacher.motion_gen import MotionGenConfig
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from isaac_ros_cumotion.xrdf_utils import convert_xrdf_to_curobo
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import RobotTrajectory
import numpy as np
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
import torch
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker


class CumotionActionServer(Node):

    def __init__(self):
        super().__init__("cumotion_action_server")
        self.tensor_args = TensorDeviceType()
        self.declare_parameter("robot", "ur5e.yml")
        self.declare_parameter("time_dilation_factor", 0.5)
        self.declare_parameter("collision_cache_mesh", 20)
        self.declare_parameter("collision_cache_cuboid", 20)
        self.declare_parameter("interpolation_dt", 0.02)
        self.declare_parameter("voxel_dims", [2.0, 2.0, 2.0])
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter("read_esdf_world", False)
        self.declare_parameter("publish_curobo_world_as_voxels", False)
        self.declare_parameter("add_ground_plane", False)
        self.declare_parameter("publish_voxel_size", 0.05)
        self.declare_parameter("max_publish_voxels", 50000)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("tool_frame", rclpy.Parameter.Type.STRING)

        self.declare_parameter("grid_position", [0.0, 0.0, 0.0])

        self.declare_parameter(
            "esdf_service_name", "/nvblox_node/get_esdf_and_gradient"
        )
        self.declare_parameter("urdf_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("enable_curobo_debug_mode", False)
        self.declare_parameter("override_moveit_scaling_factors", False)
        debug_mode = (
            self.get_parameter("enable_curobo_debug_mode")
            .get_parameter_value()
            .bool_value
        )
        if debug_mode:
            setup_curobo_logger("info")
        else:
            setup_curobo_logger("warning")

        self.__voxel_pub = self.create_publisher(Marker, "/curobo/voxels", 10)
        self._action_server = ActionServer(
            self, MoveGroup, "cumotion/move_group", self.execute_callback
        )

        try:
            self.__urdf_path = (
                self.get_parameter("urdf_path").get_parameter_value().string_value
            )
            if self.__urdf_path == "":
                self.__urdf_path = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__urdf_path = None
        self.__urdf_path = path.join(
            get_package_share_directory("isaac_ros_cumotion_robot_description"),
            "urdf",
            self.__urdf_path,
        )

        try:
            self.__tool_frame = self.get_parameter("tool_frame")
            self.__tool_frame = self.__tool_frame.get_parameter_value().string_value
            if self.__tool_frame == "":
                self.__tool_frame = None
        except rclpy.exceptions.ParameterUninitializedException:
            self.__tool_frame = None

        self.__xrdf_path = path.join(
            get_package_share_directory("isaac_ros_cumotion_robot_description"), "xrdf"
        )
        self.__joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        self.__add_ground_plane = (
            self.get_parameter("add_ground_plane").get_parameter_value().bool_value
        )
        self.__override_moveit_scaling_factors = (
            self.get_parameter("override_moveit_scaling_factors")
            .get_parameter_value()
            .bool_value
        )
        # ESDF service

        self.__read_esdf_grid = (
            self.get_parameter("read_esdf_world").get_parameter_value().bool_value
        )
        self.__publish_curobo_world_as_voxels = (
            self.get_parameter("publish_curobo_world_as_voxels")
            .get_parameter_value()
            .bool_value
        )
        self.__grid_position = (
            self.get_parameter("grid_position").get_parameter_value().double_array_value
        )
        self.__max_publish_voxels = (
            self.get_parameter("max_publish_voxels").get_parameter_value().integer_value
        )
        self.__voxel_dims = (
            self.get_parameter("voxel_dims").get_parameter_value().double_array_value
        )
        self.__publish_voxel_size = (
            self.get_parameter("publish_voxel_size").get_parameter_value().double_value
        )
        self.__voxel_size = (
            self.get_parameter("voxel_size").get_parameter_value().double_value
        )
        self.__esdf_client = None
        self.__esdf_req = None
        if self.__read_esdf_grid:
            esdf_service_name = (
                self.get_parameter("esdf_service_name")
                .get_parameter_value()
                .string_value
            )

            esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
            self.__esdf_client = self.create_client(
                EsdfAndGradients,
                esdf_service_name,
                callback_group=esdf_service_cb_group,
            )
            while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(
                    f"Service({esdf_service_name}) not available, waiting again..."
                )
            self.__esdf_req = EsdfAndGradients.Request()
        self.warmup()
        self.__query_count = 0
        self.__tensor_args = self.motion_gen.tensor_args
        self.subscription = self.create_subscription(
            JointState, self.__joint_states_topic, self.js_callback, 10
        )
        self.__js_buffer = None

    def js_callback(self, msg):
        self.__js_buffer = {
            "joint_names": msg.name,
            "position": msg.position,
            "velocity": msg.velocity,
        }

    def warmup(self):
        robot_file = self.get_parameter("robot").get_parameter_value().string_value
        if robot_file == "":
            self.get_logger().fatal("Received empty robot file")
            raise SystemExit

        collision_cache_cuboid = (
            self.get_parameter("collision_cache_cuboid")
            .get_parameter_value()
            .integer_value
        )
        collision_cache_mesh = (
            self.get_parameter("collision_cache_mesh")
            .get_parameter_value()
            .integer_value
        )
        interpolation_dt = (
            self.get_parameter("interpolation_dt").get_parameter_value().double_value
        )

        self.get_logger().info("Loaded robot file name: " + robot_file)
        self.get_logger().info("warming up cuMotion, wait until ready")

        tensor_args = self.tensor_args
        world_cfg = WorldConfig()

        if robot_file.lower().endswith(".xrdf"):
            if self.__urdf_path is None:
                self.get_logger().fatal(
                    "urdf_path is required to load robot from .xrdf"
                )
                raise SystemExit
            robot_dict = load_yaml(join_path(self.__xrdf_path, robot_file))
            robot_dict = convert_xrdf_to_curobo(
                self.__urdf_path, robot_dict, self.get_logger()
            )
        else:
            robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))

        if self.__urdf_path is not None:
            robot_dict["robot_cfg"]["kinematics"]["urdf_path"] = self.__urdf_path

        robot_dict = robot_dict["robot_cfg"]
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_dict,
            world_cfg,
            tensor_args,
            interpolation_dt=interpolation_dt,
            collision_cache={
                "obb": collision_cache_cuboid,
                "mesh": collision_cache_mesh,
            },
            collision_checker_type=CollisionCheckerType.MESH,
            ee_link_name=self.__tool_frame,
        )

        motion_gen = MotionGen(motion_gen_config)
        self.__robot_base_frame = motion_gen.kinematics.base_link

        motion_gen.warmup(enable_graph=True)

        self.__world_collision = motion_gen.world_coll_checker
        if not self.__add_ground_plane:
            motion_gen.clear_world_cache()
        self.motion_gen = motion_gen
        self.get_logger().info("cuMotion is ready for planning queries!")

    def update_voxel_grid(self):
        self.get_logger().info("Calling ESDF service")
        # This is half of x,y and z dims
        aabb_min = Point()
        aabb_min.x = -1 * self.__voxel_dims[0] / 2
        aabb_min.y = -1 * self.__voxel_dims[1] / 2
        aabb_min.z = -1 * self.__voxel_dims[2] / 2
        # This is a voxel size.
        voxel_dims = Vector3()
        voxel_dims.x = self.__voxel_dims[0]
        voxel_dims.y = self.__voxel_dims[1]
        voxel_dims.z = self.__voxel_dims[2]
        esdf_future = self.send_request(aabb_min, voxel_dims)
        while not esdf_future.done():
            time.sleep(0.001)
        response = esdf_future.result()
        esdf_grid = self.get_esdf_voxel_grid(response)
        if torch.max(esdf_grid.feature_tensor) <= (
            -1000.0 + 0.5 * self.__voxel_size + 1e-5
        ):
            self.get_logger().error("ESDF data is empty, try again after few seconds.")
            return False
        self.__world_collision.update_voxel_data(esdf_grid)
        self.get_logger().info("Updated ESDF grid")
        return True

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        self.get_logger().info(
            f"ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}"
        )
        esdf_future = self.__esdf_client.call_async(self.__esdf_req)

        return esdf_future

    def get_esdf_voxel_grid(self, esdf_data):
        esdf_array = esdf_data.esdf_and_gradients
        array_shape = [
            esdf_array.layout.dim[0].size,
            esdf_array.layout.dim[1].size,
            esdf_array.layout.dim[2].size,
        ]
        array_data = np.array(esdf_array.data)

        array_data = self.__tensor_args.to_device(array_data)

        # Array data is reshaped to x y z channels
        array_data = array_data.view(
            array_shape[0], array_shape[1], array_shape[2]
        ).contiguous()

        # Array is squeezed to 1 dimension
        array_data = array_data.reshape(-1, 1)

        # nvblox uses negative distance inside obstacles, cuRobo needs the opposite:
        array_data = -1 * array_data

        # nvblox assigns a value of -1000.0 for unobserved voxels, making
        array_data[array_data >= 1000.0] = -1000.0

        # nvblox distance are at origin of each voxel, cuRobo's esdf needs it to be at faces
        array_data = array_data + 0.5 * self.__voxel_size

        esdf_grid = CuVoxelGrid(
            name="world_voxel",
            dims=self.__voxel_dims,
            pose=[
                self.__grid_position[0],
                self.__grid_position[1],
                self.__grid_position[2],
                1,
                0.0,
                0.0,
                0,
            ],  # x, y, z, qw, qx, qy, qz
            voxel_size=self.__voxel_size,
            feature_dtype=torch.float32,
            feature_tensor=array_data,
        )

        return esdf_grid

    def get_cumotion_collision_object(self, mv_object: CollisionObject):
        objs = []
        pose = mv_object.pose

        world_pose = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]
        world_pose = Pose.from_list(world_pose)
        supported_objects = True
        if len(mv_object.primitives) > 0:
            for k in range(len(mv_object.primitives)):
                pose = mv_object.primitive_poses[k]
                primitive_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
                object_pose = world_pose.multiply(
                    Pose.from_list(primitive_pose)
                ).tolist()

                if mv_object.primitives[k].type == SolidPrimitive.BOX:
                    # cuboid:
                    dims = mv_object.primitives[k].dimensions
                    obj = Cuboid(
                        name=str(mv_object.id) + "_" + str(k) + "_cuboid",
                        pose=object_pose,
                        dims=dims,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.SPHERE:
                    # sphere:
                    radius = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].SPHERE_RADIUS
                    ]
                    obj = Sphere(
                        name=str(mv_object.id) + "_" + str(k) + "_sphere",
                        pose=object_pose,
                        radius=radius,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.CYLINDER:
                    # cylinder:
                    cyl_height = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].CYLINDER_HEIGHT
                    ]
                    cyl_radius = mv_object.primitives[k].dimensions[
                        mv_object.primitives[k].CYLINDER_RADIUS
                    ]
                    obj = Cylinder(
                        name=str(mv_object.id) + "_" + str(k) + "_cylinder",
                        pose=object_pose,
                        height=cyl_height,
                        radius=cyl_radius,
                    )
                    objs.append(obj)
                elif mv_object.primitives[k].type == SolidPrimitive.CONE:
                    self.get_logger().error("Cone primitive is not supported")
                    supported_objects = False
                else:
                    self.get_logger().error("Unknown primitive type")
                    supported_objects = False
        if len(mv_object.meshes) > 0:
            for k in range(len(mv_object.meshes)):
                pose = mv_object.mesh_poses[k]
                mesh_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
                object_pose = world_pose.multiply(Pose.from_list(mesh_pose)).tolist()
                verts = mv_object.meshes[k].vertices
                verts = [[v.x, v.y, v.z] for v in verts]
                tris = [
                    [v.vertex_indices[0], v.vertex_indices[1], v.vertex_indices[2]]
                    for v in mv_object.meshes[k].triangles
                ]

                obj = Mesh(
                    name=str(mv_object.id) + "_" + str(len(objs)) + "_mesh",
                    pose=object_pose,
                    vertices=verts,
                    faces=tris,
                )
                objs.append(obj)
        return objs, supported_objects

    def get_joint_trajectory(self, js: CuJointState, dt: float):
        traj = RobotTrajectory()
        cmd_traj = JointTrajectory()
        q_traj = js.position.cpu().view(-1, js.position.shape[-1]).numpy()
        vel = js.velocity.cpu().view(-1, js.position.shape[-1]).numpy()
        acc = js.acceleration.view(-1, js.position.shape[-1]).cpu().numpy()
        for i in range(len(q_traj)):
            traj_pt = JointTrajectoryPoint()
            traj_pt.positions = q_traj[i].tolist()
            if js is not None and i < len(vel):
                traj_pt.velocities = vel[i].tolist()
            if js is not None and i < len(acc):
                traj_pt.accelerations = acc[i].tolist()
            time_d = rclpy.time.Duration(seconds=i * dt).to_msg()
            traj_pt.time_from_start = time_d
            cmd_traj.points.append(traj_pt)
        cmd_traj.joint_names = js.joint_names
        cmd_traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_trajectory = cmd_traj
        return traj

    def update_world_objects(self, moveit_objects):
        world_update_status = True
        if len(moveit_objects) > 0:
            cuboid_list = []
            sphere_list = []
            cylinder_list = []
            mesh_list = []
            for i, obj in enumerate(moveit_objects):
                cumotion_objects, world_update_status = (
                    self.get_cumotion_collision_object(obj)
                )
                for cumotion_object in cumotion_objects:
                    if isinstance(cumotion_object, Cuboid):
                        cuboid_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Cylinder):
                        cylinder_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Sphere):
                        sphere_list.append(cumotion_object)
                    elif isinstance(cumotion_object, Mesh):
                        mesh_list.append(cumotion_object)

            world_model = WorldConfig(
                cuboid=cuboid_list,
                cylinder=cylinder_list,
                sphere=sphere_list,
                mesh=mesh_list,
            ).get_collision_check_world()
            self.motion_gen.update_world(world_model)
        if self.__read_esdf_grid:
            world_update_status = self.update_voxel_grid()
        if self.__publish_curobo_world_as_voxels:
            voxels = self.__world_collision.get_esdf_in_bounding_box(
                Cuboid(
                    name="test",
                    pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                    dims=self.__voxel_dims,
                ),
                voxel_size=self.__publish_voxel_size,
            )
            xyzr_tensor = voxels.xyzr_tensor.clone()
            xyzr_tensor[..., 3] = voxels.feature_tensor
            self.publish_voxels(xyzr_tensor)
        return world_update_status

    def execute_callback(self, goal_handle):
        self.get_logger().info("Executing goal...")
        # check moveit scaling factors:
        min_scaling_factor = min(
            goal_handle.request.request.max_velocity_scaling_factor,
            goal_handle.request.request.max_acceleration_scaling_factor,
        )
        time_dilation_factor = min(1.0, min_scaling_factor)

        if time_dilation_factor <= 0.0 or self.__override_moveit_scaling_factors:
            time_dilation_factor = (
                self.get_parameter("time_dilation_factor")
                .get_parameter_value()
                .double_value
            )
        self.get_logger().info(
            "Planning with time_dilation_factor: " + str(time_dilation_factor)
        )
        plan_req = goal_handle.request.request
        goal_handle.succeed()

        scene = goal_handle.request.planning_options.planning_scene_diff

        world_objects = scene.world.collision_objects
        world_update_status = self.update_world_objects(world_objects)
        result = MoveGroup.Result()

        if not world_update_status:
            result.error_code.val = MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
            self.get_logger().error("World update failed.")
            return result
        start_state = None
        if len(plan_req.start_state.joint_state.position) > 0:
            start_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(
                        plan_req.start_state.joint_state.position
                    ).unsqueeze(0),
                    joint_names=plan_req.start_state.joint_state.name,
                )
            )
        else:
            self.get_logger().info(
                "PlanRequest start state was empty, reading current joint state"
            )
        if start_state is None or plan_req.start_state.is_diff:
            if self.__js_buffer is None:
                self.get_logger().error(
                    "joint_state was not received from " + self.__joint_states_topic
                )
                return result

            # read joint state:
            state = CuJointState.from_position(
                position=self.tensor_args.to_device(
                    self.__js_buffer["position"]
                ).unsqueeze(0),
                joint_names=self.__js_buffer["joint_names"],
            )
            state.velocity = self.tensor_args.to_device(
                self.__js_buffer["velocity"]
            ).unsqueeze(0)
            current_joint_state = self.motion_gen.get_active_js(state)
            if start_state is not None and plan_req.start_state.is_diff:
                start_state.position += current_joint_state.position
                start_state.velocity += current_joint_state.velocity
            else:
                start_state = current_joint_state

        if len(plan_req.goal_constraints[0].joint_constraints) > 0:
            self.get_logger().info("Calculating goal pose from Joint target")
            goal_config = [
                plan_req.goal_constraints[0].joint_constraints[x].position
                for x in range(len(plan_req.goal_constraints[0].joint_constraints))
            ]
            goal_jnames = [
                plan_req.goal_constraints[0].joint_constraints[x].joint_name
                for x in range(len(plan_req.goal_constraints[0].joint_constraints))
            ]
            goal_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(goal_config).view(1, -1),
                    joint_names=goal_jnames,
                )
            )
            kinematic_model_state = self.motion_gen.compute_kinematics(goal_state)
            goal_pose = kinematic_model_state.ee_pose.clone()
            link_poses = kinematic_model_state.link_poses
            for key in list(link_poses.keys()):
                link_poses[key] = link_poses[key].clone()
        elif (
            len(plan_req.goal_constraints[0].position_constraints) > 0
            and len(plan_req.goal_constraints[0].orientation_constraints) > 0
        ):
            self.get_logger().info("Using goal from Pose")

            position = (
                plan_req.goal_constraints[0]
                .position_constraints[0]
                .constraint_region.primitive_poses[0]
                .position
            )
            position = [position.x, position.y, position.z]
            orientation = (
                plan_req.goal_constraints[0].orientation_constraints[0].orientation
            )
            orientation = [orientation.w, orientation.x, orientation.y, orientation.z]
            pose_list = position + orientation
            goal_pose = Pose.from_list(pose_list, tensor_args=self.tensor_args)

            # Check if link names match:
            position_link_name = (
                plan_req.goal_constraints[0].position_constraints[0].link_name
            )
            orientation_link_name = (
                plan_req.goal_constraints[0].orientation_constraints[0].link_name
            )
            plan_link_name = self.motion_gen.kinematics.ee_link
            if position_link_name != orientation_link_name:
                self.get_logger().error(
                    'Link name for Target Position "'
                    + position_link_name
                    + '" and Target Orientation "'
                    + orientation_link_name
                    + '" do not match'
                )
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return result
            if position_link_name != plan_link_name:
                self.get_logger().error(
                    'Link name for Target Pose "'
                    + position_link_name
                    + '" and Planning frame "'
                    + plan_link_name
                    + '" do not match, relaunch node with tool_frame = '
                    + position_link_name
                )
                result.error_code.val = MoveItErrorCodes.INVALID_LINK_NAME
                return result
        else:
            self.get_logger().error("Goal constraints not supported")

        self.motion_gen.reset(reset_seed=False)
        motion_gen_result = self.motion_gen.plan_single(
            start_state,
            goal_pose,
            MotionGenPlanConfig(
                max_attempts=5,
                enable_graph_attempt=1,
                time_dilation_factor=time_dilation_factor,
            ),
            link_poses,
        )

        result = MoveGroup.Result()
        if motion_gen_result.success.item():
            result.error_code.val = MoveItErrorCodes.SUCCESS
            result.trajectory_start = plan_req.start_state
            traj = self.get_joint_trajectory(
                motion_gen_result.optimized_plan, motion_gen_result.optimized_dt.item()
            )
            result.planning_time = motion_gen_result.total_time
            result.planned_trajectory = traj
        elif not motion_gen_result.valid_query:
            self.get_logger().error(
                f"Invalid planning query: {motion_gen_result.status}"
            )
            if (
                motion_gen_result.status
                == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS
            ):
                result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
            if motion_gen_result.status in [
                MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
            ]:

                result.error_code.val = MoveItErrorCodes.START_STATE_IN_COLLISION
        else:
            self.get_logger().error(
                f"Motion planning failed wih status: {motion_gen_result.status}"
            )
            if motion_gen_result.status == MotionGenStatus.IK_FAIL:
                result.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION

        self.get_logger().info(
            "returned planning result (query, success, failure_status): "
            + str(self.__query_count)
            + " "
            + str(motion_gen_result.success.item())
            + " "
            + str(motion_gen_result.status)
        )
        self.__query_count += 1
        return result

    def publish_voxels(self, voxels):
        vox_size = self.__publish_voxel_size

        # create marker:
        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.id = 0
        marker.type = 6  # cube list
        marker.ns = "curobo_world"
        marker.action = 0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=1000.0).to_msg()
        marker.frame_locked = False
        marker.scale.x = vox_size
        marker.scale.y = vox_size
        marker.scale.z = vox_size

        # get only voxels that are inside surfaces:

        voxels = voxels[voxels[:, 3] >= 0.0]
        vox = voxels.view(-1, 4).cpu().numpy()
        marker.points = []

        for i in range(min(len(vox), self.__max_publish_voxels)):

            pt = Point()
            pt.x = float(vox[i, 0])
            pt.y = float(vox[i, 1])
            pt.z = float(vox[i, 2])
            color = ColorRGBA()
            d = vox[i, 3]

            rgba = [min(1.0, 1.0 - float(d)), 0.0, 0.0, 1.0]

            color.r = rgba[0]
            color.g = rgba[1]
            color.b = rgba[2]
            color.a = rgba[3]
            marker.colors.append(color)
            marker.points.append(pt)
        # publish voxels:
        marker.header.stamp = self.get_clock().now().to_msg()

        self.__voxel_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    cumotion_action_server = CumotionActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(cumotion_action_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        cumotion_action_server.get_logger().info("KeyboardInterrupt, shutting down.\n")
    cumotion_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
