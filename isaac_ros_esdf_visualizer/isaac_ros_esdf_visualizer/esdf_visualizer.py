# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sdf.world import WorldCollisionConfig
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.types import Cuboid as CuCuboid
from curobo.geom.types import VoxelGrid as CuVoxelGrid
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from isaac_ros_cumotion_python_utils.utils import \
    get_grid_center, get_grid_min_corner, get_grid_size, is_grid_valid, \
    load_grid_corners_from_workspace_file
import numpy as np
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
import torch
from visualization_msgs.msg import Marker


class ESDFVisualizer(Node):

    def __init__(self):
        super().__init__('esdf_visualizer')

        # The grid_center_m and grid_size_m parameters are loaded from the workspace file
        # if the workspace_file_path is set and valid.
        self.declare_parameter('workspace_file_path', '')
        self.declare_parameter('grid_center_m', [0.0, 0.0, 0.0])
        self.declare_parameter('grid_size_m', [2.0, 2.0, 2.0])
        self.declare_parameter('update_esdf_on_request', True)
        self.declare_parameter('use_aabb_on_request', True)

        # Parameters for clearing shapes in the map.
        self.declare_parameter('clear_shapes_on_request', False)
        self.declare_parameter('clear_shapes_subsampling_factor', 2)
        self.declare_parameter('aabbs_to_clear_min_m', [0.7, 0.6, -0.1])
        self.declare_parameter('aabbs_to_clear_size_m', [0.2, 0.2, 0.9])
        self.declare_parameter('spheres_to_clear_center_m', [1.5, 0.5, 0.0])
        self.declare_parameter('spheres_to_clear_radius_m', 0.2)

        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('publish_voxel_size', 0.01)
        self.declare_parameter('max_publish_voxels', 500000)

        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('esdf_service_call_period_secs', 0.01)
        self.__esdf_future = None
        # Voxel publisher
        self.__voxel_pub = self.create_publisher(Marker, '/curobo/voxels', 10)
        # ESDF service
        esdf_service_name = (
            self.get_parameter('esdf_service_name').get_parameter_value().string_value
        )
        esdf_service_cb_group = MutuallyExclusiveCallbackGroup()
        self.__esdf_client = self.create_client(
            EsdfAndGradients, esdf_service_name, callback_group=esdf_service_cb_group
        )
        while not self.__esdf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Service({esdf_service_name}) not available, waiting again...')
        self.__esdf_req = EsdfAndGradients.Request()

        # Timer for calling the ESDF service
        timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            self.get_parameter('esdf_service_call_period_secs').get_parameter_value().double_value,
            self.timer_callback, callback_group=timer_cb_group
        )

        self.__workspace_file_path = (
            self.get_parameter('workspace_file_path').get_parameter_value().string_value
        )
        self.__grid_size_m = (
            self.get_parameter('grid_size_m').get_parameter_value().double_array_value
        )
        self.__update_esdf_on_request = (
            self.get_parameter('update_esdf_on_request').get_parameter_value().bool_value
        )
        self.__use_aabb_on_request = (
            self.get_parameter('use_aabb_on_request').get_parameter_value().bool_value
        )
        self.__grid_center_m = (
            self.get_parameter('grid_center_m').get_parameter_value().double_array_value
        )
        self.__voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.__publish_voxel_size = (
            self.get_parameter('publish_voxel_size').get_parameter_value().double_value
        )
        self.__max_publish_voxels = (
            self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        )
        self.__clear_shapes_on_request = (
            self.get_parameter('clear_shapes_on_request').get_parameter_value().bool_value)
        self.__clear_shapes_subsampling_factor = (
            self.get_parameter(
                'clear_shapes_subsampling_factor').get_parameter_value().integer_value)
        self.__aabbs_to_clear_min_m = (
            self.get_parameter('aabbs_to_clear_min_m').get_parameter_value().double_array_value)
        self.__aabbs_to_clear_size_m = (
            self.get_parameter('aabbs_to_clear_size_m').get_parameter_value().double_array_value)
        self.__spheres_to_clear_center_m = (
            self.get_parameter(
                'spheres_to_clear_center_m').get_parameter_value().double_array_value)
        self.__spheres_to_clear_radius_m = (
            self.get_parameter('spheres_to_clear_radius_m').get_parameter_value().double_value)

        # Setup the grid position and dimension.
        if os.path.exists(self.__workspace_file_path):
            self.get_logger().info(
                f'Loading grid center and dims from workspace file: {self.__workspace_file_path}.')
            min_corner, max_corner = load_grid_corners_from_workspace_file(
                self.__workspace_file_path)
            self.__grid_size_m = get_grid_size(min_corner, max_corner, self.__voxel_size)
            self.__grid_center_m = get_grid_center(min_corner, self.__grid_size_m)
        else:
            self.get_logger().info(
                'Loading grid position and dims from grid_center_m and grid_size_m parameters.')

        if is_grid_valid(self.__grid_size_m, self.__voxel_size):
            self.get_logger().fatal('Number of voxels should be at least 1 in every dimension.')
            raise SystemExit

        # Init WorldVoxelCollision
        world_cfg = WorldConfig.from_dict(
            {
                'voxel': {
                    'world_voxel': {
                        'dims': self.__grid_size_m,
                        'pose': [
                            self.__grid_center_m[0],
                            self.__grid_center_m[1],
                            self.__grid_center_m[2],
                            1,
                            0,
                            0,
                            0,
                        ],  # x, y, z, qw, qx, qy, qz
                        'voxel_size': self.__voxel_size,
                        'feature_dtype': torch.float32,
                    },
                },
            },
        )
        tensor_args = TensorDeviceType()
        world_collision_config = WorldCollisionConfig.load_from_dict(
            {
                'checker_type': CollisionCheckerType.VOXEL,
                'max_distance': 10.0,
                'n_envs': 1,
            },
            world_cfg,
            tensor_args,
        )
        self.__world_collision = WorldVoxelCollision(world_collision_config)

        self.__robot_base_frame = (
            self.get_parameter('robot_base_frame').get_parameter_value().string_value
        )
        self.__tensor_args = tensor_args
        self.__clear_shapes_counter = 0
        self.__cumotion_grid_shape = self.__world_collision.get_voxel_grid(
            'world_voxel').get_grid_shape()[0]

    def timer_callback(self):
        if self.__esdf_future is None:
            self.get_logger().info('Calling ESDF service')

            # Get the AABB
            min_corner = get_grid_min_corner(self.__grid_center_m, self.__grid_size_m)
            aabb_min = Point()
            aabb_min.x = min_corner[0]
            aabb_min.y = min_corner[1]
            aabb_min.z = min_corner[2]
            aabb_size = Vector3()
            aabb_size.x = self.__grid_size_m[0]
            aabb_size.y = self.__grid_size_m[1]
            aabb_size.z = self.__grid_size_m[2]

            # Request the esdf grid
            self.__esdf_future = self.send_request(aabb_min, aabb_size)
        if self.__esdf_future is not None and self.__esdf_future.done():
            response = self.__esdf_future.result()
            if response.success:
                self.fill_marker(response)
            else:
                self.get_logger().info('ESDF request failed. Not updating the grid.')
            self.__esdf_future = None

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.visualize_esdf = True
        self.__esdf_req.update_esdf = self.__update_esdf_on_request
        self.__esdf_req.use_aabb = self.__use_aabb_on_request
        self.__esdf_req.frame_id = self.__robot_base_frame
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        # Optionally clear an AABB and a sphere in the map.
        if self.__clear_shapes_on_request and \
                self.__clear_shapes_counter % self.__clear_shapes_subsampling_factor == 0:
            aabbs_to_clear_min_m = Point(
                x=self.__aabbs_to_clear_min_m[0],
                y=self.__aabbs_to_clear_min_m[1],
                z=self.__aabbs_to_clear_min_m[2])
            aabbs_to_clear_size_m = Vector3(
                x=self.__aabbs_to_clear_size_m[0],
                y=self.__aabbs_to_clear_size_m[1],
                z=self.__aabbs_to_clear_size_m[2])
            spheres_to_clear_center_m = Point(
                x=self.__spheres_to_clear_center_m[0],
                y=self.__spheres_to_clear_center_m[1],
                z=self.__spheres_to_clear_center_m[2])
            self.__esdf_req.aabbs_to_clear_min_m = [aabbs_to_clear_min_m]
            self.__esdf_req.aabbs_to_clear_size_m = [aabbs_to_clear_size_m]
            self.__esdf_req.spheres_to_clear_center_m = [spheres_to_clear_center_m]
            self.__esdf_req.spheres_to_clear_radius_m = [self.__spheres_to_clear_radius_m]
        else:
            self.__esdf_req.aabbs_to_clear_min_m = []
            self.__esdf_req.aabbs_to_clear_size_m = []
            self.__esdf_req.spheres_to_clear_center_m = []
            self.__esdf_req.spheres_to_clear_radius_m = []
        self.__clear_shapes_counter += 1

        self.get_logger().info(
            f'ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}'
        )
        esdf_future = self.__esdf_client.call_async(self.__esdf_req)
        return esdf_future

    def get_esdf_voxel_grid(self, esdf_data):
        esdf_voxel_size = esdf_data.voxel_size_m
        if abs(esdf_voxel_size - self.__voxel_size) > 1e-4:
            self.get_logger().fatal(
                'Voxel size of esdf array is not equal to requested voxel_size, '
                f'{esdf_voxel_size} vs. {self.__voxel_size}')
            raise SystemExit

        # Get the esdf and gradient data
        esdf_array = esdf_data.esdf_and_gradients
        array_shape = [
            esdf_array.layout.dim[0].size,
            esdf_array.layout.dim[1].size,
            esdf_array.layout.dim[2].size,
        ]
        array_data = np.array(esdf_array.data, dtype=np.float32)
        array_data = torch.as_tensor(array_data)

        # Verify the grid shape
        if array_shape != self.__cumotion_grid_shape:
            self.get_logger().fatal(
                'Shape of received esdf voxel grid does not match the cumotion grid shape, '
                f'{array_shape} vs. {self.__cumotion_grid_shape}')
            raise SystemExit

        # Get the origin of the grid
        grid_origin = [
            esdf_data.origin_m.x,
            esdf_data.origin_m.y,
            esdf_data.origin_m.z,
        ]
        # The grid position is defined as the center point of the grid.
        grid_center_m = get_grid_center(grid_origin, self.__grid_size_m)

        # Array data is reshaped to x y z channels
        array_data = array_data.view(array_shape[0], array_shape[1], array_shape[2]).contiguous()

        # Array is squeezed to 1 dimension
        array_data = array_data.reshape(-1, 1)

        # nvblox assigns a value of -1000.0 for unobserved voxels, making it positive
        array_data[array_data < -999.9] = 1000.0

        # nvblox uses negative distance inside obstacles, cuRobo needs the opposite:
        array_data = -1.0 * array_data

        # nvblox treats surface voxels as distance = 0.0, while cuRobo treats
        # distance = 0.0 as not in collision. Adding an offset.
        array_data += 0.5 * self.__voxel_size

        esdf_grid = CuVoxelGrid(
            name='world_voxel',
            dims=self.__grid_size_m,
            pose=grid_center_m + [1, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
            voxel_size=self.__voxel_size,
            feature_dtype=torch.float32,
            feature_tensor=array_data,
        )

        return esdf_grid

    def fill_marker(self, esdf_data):
        esdf_grid = self.get_esdf_voxel_grid(esdf_data)
        self.__world_collision.update_voxel_data(esdf_grid)
        vox_size = self.__publish_voxel_size
        voxels = self.__world_collision.get_occupancy_in_bounding_box(
            CuCuboid(
                name='test',
                pose=[
                    self.__grid_center_m[0],
                    self.__grid_center_m[1],
                    self.__grid_center_m[2],
                    1, 0, 0, 0
                ],  # x, y, z, qw, qx, qy, qz
                dims=self.__grid_size_m,
            ),
            voxel_size=vox_size,
        )
        xyzr_tensor = voxels.xyzr_tensor.clone()
        xyzr_tensor[..., 3] = voxels.feature_tensor
        self.publish_voxels(xyzr_tensor)

    def publish_voxels(self, voxels):
        if self.__voxel_pub.get_subscription_count() == 0:
            # Nobody is listening, no need to create message and publish.
            return

        vox_size = self.__publish_voxel_size

        # create marker:
        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.id = 0
        marker.type = 6  # cube list
        marker.ns = 'curobo_world'
        marker.action = 0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()
        marker.frame_locked = False
        marker.scale.x = vox_size
        marker.scale.y = vox_size
        marker.scale.z = vox_size

        # get only voxels that are inside surfaces:

        voxels = voxels[voxels[:, 3] > 0.0]
        vox = voxels.view(-1, 4).cpu().numpy()
        marker.points = []
        number_of_voxels_to_publish = len(vox)
        if len(vox) > self.__max_publish_voxels:
            self.get_logger().warn(
                f'Number of voxels to publish bigger than max_publish_voxels, '
                f'{len(vox)} > {self.__max_publish_voxels}'
            )
            number_of_voxels_to_publish = self.__max_publish_voxels
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        vox = vox.astype(np.float64)
        for i in range(number_of_voxels_to_publish):
            # Publish the markers at the center of the voxels:
            pt = Point()
            pt.x = vox[i, 0]
            pt.y = vox[i, 1]
            pt.z = vox[i, 2]
            marker.points.append(pt)

        # publish voxels:
        marker.header.stamp = self.get_clock().now().to_msg()

        self.__voxel_pub.publish(marker)


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    esdf_client = ESDFVisualizer()

    # Spin the node so the callback function is called.
    try:
        esdf_client.get_logger().info('Starting ESDFVisualizer node')
        rclpy.spin(esdf_client)

    except KeyboardInterrupt:
        esdf_client.get_logger().info('Destroying ESDFVisualizer node')

    except Exception as e:
        esdf_client.get_logger().info(f'Shutting down due to exception of type {type(e)}: {e}')

    # Destroy the node explicitly
    esdf_client.destroy_node()

    # Shutdown the ROS client library for Python
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
