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

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sdf.world import WorldCollisionConfig
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.types import Cuboid as CuCuboid
from curobo.geom.types import VoxelGrid as CuVoxelGrid
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
import numpy as np
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
import torch
from visualization_msgs.msg import Marker


class ESDFVisualizer(Node):

    def __init__(self):
        super().__init__('esdf_visualizer')

        self.declare_parameter('voxel_dims', [1.25, 1.8, 1.8])
        self.declare_parameter('grid_position', [0.0, 0.0, 0.0])

        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('publish_voxel_size', 0.025)
        self.declare_parameter('max_publish_voxels', 50000)

        self.declare_parameter('esdf_service_name', '/nvblox_node/get_esdf_and_gradient')
        self.declare_parameter('robot_base_frame', 'base_link')
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
        timer_period = 0.01
        timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(
            timer_period, self.timer_callback, callback_group=timer_cb_group
        )

        self.__voxel_dims = (
            self.get_parameter('voxel_dims').get_parameter_value().double_array_value
        )

        self.__grid_position = (
            self.get_parameter('grid_position').get_parameter_value().double_array_value
        )
        self.__voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.__publish_voxel_size = (
            self.get_parameter('publish_voxel_size').get_parameter_value().double_value
        )
        self.__max_publish_voxels = (
            self.get_parameter('max_publish_voxels').get_parameter_value().integer_value
        )
        # Init WorldVoxelCollision
        world_cfg = WorldConfig.from_dict(
            {
                'voxel': {
                    'world_voxel': {
                        'dims': self.__voxel_dims,
                        'pose': [
                            self.__grid_position[0],
                            self.__grid_position[1],
                            self.__grid_position[2],
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

        esdf_grid = CuVoxelGrid(
            name='world_voxel',
            dims=self.__voxel_dims,
            pose=[
                self.__grid_position[0],
                self.__grid_position[1],
                self.__grid_position[2],
                1,
                0,
                0,
                0,
            ],
            voxel_size=self.__voxel_size,
            feature_dtype=torch.float32,
        )
        self.__grid_shape, _, _ = esdf_grid.get_grid_shape()

    def timer_callback(self):
        if self.__esdf_future is None:
            self.get_logger().info('Calling ESDF service')
            # This is half of x,y and z dims
            aabb_min = Point()
            aabb_min.x = (-0.5 * self.__voxel_dims[0]) + self.__grid_position[0]
            aabb_min.y = (-0.5 * self.__voxel_dims[1]) + self.__grid_position[1]
            aabb_min.z = (-0.5 * self.__voxel_dims[2]) + self.__grid_position[2]
            # This is a voxel size.
            voxel_dims = Vector3()
            voxel_dims.x = self.__voxel_dims[0]
            voxel_dims.y = self.__voxel_dims[1]
            voxel_dims.z = self.__voxel_dims[2]

            self.__esdf_future = self.send_request(aabb_min, voxel_dims)
        if self.__esdf_future is not None and self.__esdf_future.done():
            response = self.__esdf_future.result()
            self.fill_marker(response)
            self.__esdf_future = None

    def send_request(self, aabb_min_m, aabb_size_m):
        self.__esdf_req.aabb_min_m = aabb_min_m
        self.__esdf_req.aabb_size_m = aabb_size_m
        self.get_logger().info(
            f'ESDF  req = {self.__esdf_req.aabb_min_m}, {self.__esdf_req.aabb_size_m}'
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
        array_data = array_data.view(array_shape[0], array_shape[1], array_shape[2]).contiguous()

        # Array is squeezed to 1 dimension
        array_data = array_data.reshape(-1, 1)

        # nvblox uses negative distance inside obstacles, cuRobo needs the opposite:
        array_data = -1 * array_data

        # nvblox assigns a value of -1000.0 for unobserved voxels, making
        array_data[array_data >= 1000.0] = -1000.0

        # nvblox distance are at origin of each voxel, cuRobo's esdf needs it to be at faces
        array_data = array_data + 0.5 * self.__voxel_size

        esdf_grid = CuVoxelGrid(
            name='world_voxel',
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

    def fill_marker(self, esdf_data):
        esdf_grid = self.get_esdf_voxel_grid(esdf_data)
        self.__world_collision.update_voxel_data(esdf_grid)
        vox_size = self.__publish_voxel_size
        voxels = self.__world_collision.get_esdf_in_bounding_box(
            CuCuboid(
                name='test',
                pose=[0.0, 0.0, 0.0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
                dims=self.__voxel_dims,
            ),
            voxel_size=vox_size,
        )
        xyzr_tensor = voxels.xyzr_tensor.clone()
        xyzr_tensor[..., 3] = voxels.feature_tensor
        self.publish_voxels(xyzr_tensor)

    def publish_voxels(self, voxels):
        vox_size = 0.25 * self.__publish_voxel_size

        # create marker:
        marker = Marker()
        marker.header.frame_id = self.__robot_base_frame
        marker.id = 0
        marker.type = 6  # cube list
        marker.ns = 'curobo_world'
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

    # Destroy the node explicitly
    esdf_client.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()
