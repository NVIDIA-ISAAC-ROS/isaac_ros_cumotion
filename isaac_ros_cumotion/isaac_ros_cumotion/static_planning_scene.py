# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

from isaac_ros_cumotion_interfaces.srv import PublishStaticPlanningScene
from isaac_ros_cumotion_python_utils.moveit_scene_file_parser import MoveItSceneFileReader
from moveit_msgs.msg import PlanningScene
import rclpy
from rclpy.node import Node


class StaticPlanningSceneServer(Node):
    """
    ROS2 node that provides a service to parse and publish MoveIt collision objects scene files.

    This node reads a MoveIt scene file, parses the collision objects, and publishes them
    to the /planning_scene topic. It also provides a service that returns the parsed
    planning scene for other nodes to use.
    """

    def __init__(self):
        super().__init__('static_planning_scene_server')
        self.declare_parameter('moveit_collision_objects_scene_file', '')
        self.__moveit_collision_objects_scene_file = (
            self.get_parameter('moveit_collision_objects_scene_file')
            .get_parameter_value().string_value
        )

        # Create service and publisher
        self.srv = self.create_service(
            PublishStaticPlanningScene,
            'publish_static_planning_scene',
            self.publish_planning_scene_callback
        )
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        self.get_logger().info('Static Planning Scene Server initialized')

    def publish_planning_scene_callback(self, request, response):
        """
        Service callback to parse and publish the planning scene.

        Returns
        -------
            response: Service response with success status and planning scene

        """
        # Use the provided scene file path if available, otherwise use the default
        scene_file_path = (request.scene_file_path if request.scene_file_path
                           else self.__moveit_collision_objects_scene_file)

        if not scene_file_path:
            response.success = False
            response.message = 'No static planning scene file path provided'
            response.status = 1
            self.get_logger().info('No static planning scene file path provided')
            return response

        if os.path.exists(scene_file_path):
            self.get_logger().info(
                'Loading collision objects from scene file: '
                f'{scene_file_path}'
            )
            try:
                planning_scene = self.parse_moveit_collision_objects_scene_file(scene_file_path)
                self.planning_scene_pub.publish(planning_scene)
                self.get_logger().info(
                    f'Published {len(planning_scene.world.collision_objects)} '
                    'collision objects to /planning_scene'
                )
                response.planning_scene = planning_scene
                response.success = True
                response.message = 'Planning scene published successfully.'
                response.status = 0
            except Exception as e:
                self.get_logger().error(f'Failed to publish planning scene: {e}')
                response.success = False
                response.message = f'Failed to publish planning scene: {e}'
                response.status = 3
        else:
            response.success = False
            response.message = f'Scene file not found: {scene_file_path}'
            response.status = 2
        return response

    def parse_moveit_collision_objects_scene_file(self, scene_file_path: str) -> PlanningScene:
        """
        Parse a MoveIt scene file and return a PlanningScene message containing collision objects.

        Args
        ----
            scene_file_path (str): Path to the MoveIt scene file

        Returns
        -------
            PlanningScene: A PlanningScene message containing the parsed collision objects

        Raises
        ------
            FileNotFoundError: If the scene file does not exist
            ValueError: If the scene file is invalid or contains unsupported objects

        """
        try:
            scene_reader = MoveItSceneFileReader()
            return scene_reader.parse_scene_file(scene_file_path)
        except Exception as e:
            self.get_logger().error(
                'Unexpected error while parsing scene file: ' + str(e)
            )
            raise e


def main(args=None):
    """Initialize and run the static planning scene server."""
    rclpy.init(args=args)
    static_planning_scene_server = StaticPlanningSceneServer()

    try:
        rclpy.spin(static_planning_scene_server)
    except KeyboardInterrupt:
        static_planning_scene_server.get_logger().info('Destroying StaticPlanningSceneServer node')
    finally:
        static_planning_scene_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
