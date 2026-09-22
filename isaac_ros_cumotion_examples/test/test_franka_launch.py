# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Check the upstream demo compatibility patches without requiring ROS."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


DEMO = """
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config = config
    move_group_node = Node(
        package="moveit_ros_move_group", executable="move_group",
        parameters=[moveit_config.planning_pipelines])
    static_tf_node = Node(
        package="tf2_ros", executable="static_transform_publisher",
        arguments=TF_ARGUMENTS)
    ros2_controllers_path = "/upstream/config/ros2_controllers.lyrical.yaml"
    ros2_control_node = Node(
        package="controller_manager", executable="ros2_control_node",
        parameters=[ros2_controllers_path])
    joint_state_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager",
                   "/controller_manager"PARAM_ARGUMENTS])
    panda_arm_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"PARAM_ARGUMENTS])
    panda_hand_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"PARAM_ARGUMENTS])
    return [move_group_node, static_tf_node, ros2_control_node,
            joint_state_broadcaster_spawner, panda_arm_controller_spawner,
            panda_hand_controller_spawner]
"""


class TestFrankaLaunch(unittest.TestCase):
    """Exercise old and fixed upstream launch shapes through generated nodes."""

    def generate_nodes(self, tf_arguments, param_arguments):
        """Load the Isaac wrapper against an isolated upstream demo fixture."""
        with tempfile.TemporaryDirectory() as directory:
            share = Path(directory)
            (share / 'launch').mkdir()
            (share / 'config').mkdir()
            (share / 'launch/demo.launch.py').write_text(
                DEMO.replace('TF_ARGUMENTS', repr(tf_arguments)).replace(
                    'PARAM_ARGUMENTS', param_arguments))
            (share / 'config/isaac_ros_cumotion_planning.yaml').write_text('{}')
            packages = ModuleType('ament_index_python.packages')
            packages.get_package_share_directory = lambda package: str(share)
            actions = ModuleType('launch_ros.actions')
            actions.Node = lambda **kwargs: SimpleNamespace(**kwargs)
            yaml = ModuleType('yaml')
            yaml.safe_load = lambda stream: {'planning_plugin': 'test_cumotion'}
            modules = {
                'ament_index_python': ModuleType('ament_index_python'),
                'ament_index_python.packages': packages,
                'launch_ros': ModuleType('launch_ros'),
                'launch_ros.actions': actions,
                'yaml': yaml,
            }
            launch_file = Path(__file__).parents[1] / 'launch/franka.launch.py'
            spec = importlib.util.spec_from_file_location('franka_launch', launch_file)
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, modules):
                spec.loader.exec_module(module)
                module.config = SimpleNamespace(planning_pipelines={
                    'planning_pipelines': ['ompl'],
                    'default_planning_pipeline': 'ompl',
                })
                return module.generate_launch_description()

    def check_nodes(self, nodes, expected_tf, param_flag='--param-file'):
        """Assert controller parameters, transform semantics, and cuMotion survive."""
        move_group, static_tf, control, *spawners = nodes
        pipelines = move_group.parameters[0]
        self.assertEqual(pipelines['planning_pipelines'], ['ompl', 'isaac_ros_cumotion'])
        self.assertEqual(pipelines['default_planning_pipeline'], 'isaac_ros_cumotion')
        self.assertEqual(pipelines['isaac_ros_cumotion'], {'planning_plugin': 'test_cumotion'})
        self.assertEqual(static_tf.arguments, expected_tf)
        self.assertEqual(control.executable, 'ros2_control_node')
        self.assertEqual([node.arguments[0] for node in spawners], [
            'joint_state_broadcaster', 'panda_arm_controller', 'panda_hand_controller'])
        for node in spawners:
            self.assertEqual(len(node.arguments), 5)
            self.assertEqual(sum(node.arguments.count(flag) for flag in ('--param-file', '-p')), 1)
            self.assertEqual(node.arguments.count(param_flag), 1)
            self.assertEqual(node.arguments[node.arguments.index(param_flag) + 1],
                             control.parameters[0])

    def test_old_upstream(self):
        """Supply controller params and preserve positional yaw/pitch/roll semantics."""
        nodes = self.generate_nodes(
            ['1', '2', '3', '0.1', '0.2', '0.3', 'world', 'panda_link0'], '')
        self.check_nodes(nodes, [
            '--x', '1', '--y', '2', '--z', '3',
            '--yaw', '0.1', '--pitch', '0.2', '--roll', '0.3',
            '--frame-id', 'world', '--child-frame-id', 'panda_link0'])

    def test_fixed_upstream(self):
        """Retain already-correct upstream arguments without duplication."""
        arguments = ['--frame-id', 'world', '--child-frame-id', 'panda_link0']
        nodes = self.generate_nodes(arguments, ', "--param-file", ros2_controllers_path')
        self.check_nodes(nodes, arguments)

    def test_short_param_flag(self):
        """Retain the equivalent short controller parameter-file option."""
        arguments = ['--frame-id', 'world', '--child-frame-id', 'panda_link0']
        nodes = self.generate_nodes(arguments, ', "-p", ros2_controllers_path')
        self.check_nodes(nodes, arguments, param_flag='-p')


if __name__ == '__main__':
    unittest.main()
