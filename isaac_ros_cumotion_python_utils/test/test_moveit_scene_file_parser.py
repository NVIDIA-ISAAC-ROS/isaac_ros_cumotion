# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tempfile
from typing import List
import unittest

from isaac_ros_cumotion_python_utils.moveit_scene_file_parser import MoveItSceneFileReader
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


class TestMoveItSceneFileParser(unittest.TestCase):
    """Test cases for MoveItSceneFileParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = MoveItSceneFileReader()
        self.VALID_HEADER = '(noname)+\n'
        self.VALID_BOX = """* Box_0
-0.32 0.02 0
0 0 0 1
1
box
0.2 0.6 0.8
0 0 0
0 0 0 1
0 0 0 0
0
"""
        self.VALID_SPHERE = """* Sphere_0
0 0 0
0 0 0 1
1
sphere
0.1
0 0 0
0 0 0 1
0 0 0 0
0
"""
        self.VALID_CYLINDER = """* Cylinder_0
-0.8 0 0
0 0 0 1
1
cylinder
0.05 0.8
0 0 0
0 0 0 1
0 0 0 0
0
"""

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        scene_file = self._create_scene_file('')
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_missing_header(self):
        """Test parsing a file without header."""
        scene_file = self._create_scene_file('invalid_header\n')
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_invalid_object_id(self):
        """Test parsing a file with invalid object ID."""
        invalid_id = (
            self.VALID_HEADER + '*\n-0.32 0.02 0\n0 0 0 1\n1\nbox\n0.2 0.6 0.8\n'
            '0 0 0\n0 0 0 1\n0 0 0 0\n0\n'
        )
        scene_file = self._create_scene_file(invalid_id)
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_invalid_position(self):
        """Test parsing a file with invalid position values."""
        invalid_pos = (
            self.VALID_HEADER + '* Box_0\ninvalid\n0 0 0 1\n1\nbox\n0.2 0.6 0.8\n'
            '0 0 0\n0 0 0 1\n0 0 0 0\n0\n'
        )
        scene_file = self._create_scene_file(invalid_pos)
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_invalid_orientation(self):
        """Test parsing a file with invalid orientation values."""
        invalid_ori = (
            self.VALID_HEADER + '* Box_0\n-0.32 0.02 0\ninvalid\n1\nbox\n0.2 0.6 0.8\n'
            '0 0 0\n0 0 0 1\n0 0 0 0\n0\n'
        )
        scene_file = self._create_scene_file(invalid_ori)
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_invalid_dimensions(self):
        """Test parsing a file with invalid dimensions."""
        invalid_dims = (
            self.VALID_HEADER + '* Box_0\n-0.32 0.02 0\n0 0 0 1\n1\nbox\ninvalid\n'
            '0 0 0\n0 0 0 1\n0 0 0 0\n0\n'
        )
        scene_file = self._create_scene_file(invalid_dims)
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_unsupported_shape(self):
        """Test parsing a file with unsupported shape type."""
        invalid_shape = (
            self.VALID_HEADER + '* Box_0\n-0.32 0.02 0\n0 0 0 1\n1\ninvalid\n0.2 0.6 0.8\n'
            '0 0 0\n0 0 0 1\n0 0 0 0\n0\n'
        )
        scene_file = self._create_scene_file(invalid_shape)
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_duplicate_object_id(self):
        """Test parsing a file with duplicate object IDs."""
        scene_file = self._create_scene_file(
            self.VALID_HEADER + self.VALID_BOX + self.VALID_BOX
        )
        try:
            with self.assertRaises(ValueError):
                self.parser.parse_scene_file(scene_file)
        finally:
            os.unlink(scene_file)

    def test_parse_valid_scene_file(self):
        """Test parsing a valid scene file with multiple objects."""
        scene_file = self._create_scene_file(
            self.VALID_HEADER + self.VALID_BOX + self.VALID_SPHERE
        )

        try:
            scene = self.parser.parse_scene_file(scene_file)
            self.assertEqual(len(scene.world.collision_objects), 2)

            # Check box
            box = scene.world.collision_objects[0]
            self._assert_collision_object(
                box,
                'Box_0',
                [-0.32, 0.02, 0],
                [0, 0, 0, 1],
                SolidPrimitive.BOX,
                [0.2, 0.6, 0.8]
            )

            # Check sphere
            sphere = scene.world.collision_objects[1]
            self._assert_collision_object(
                sphere,
                'Sphere_0',
                [0, 0, 0],
                [0, 0, 0, 1],
                SolidPrimitive.SPHERE,
                [0.1]
            )
        finally:
            os.unlink(scene_file)

    def test_parse_complex_scene(self):
        """Test parsing a complex scene with multiple object types."""
        scene_file = self._create_scene_file(
            self.VALID_HEADER + self.VALID_BOX + self.VALID_SPHERE + self.VALID_CYLINDER
        )
        try:
            scene = self.parser.parse_scene_file(scene_file)
            self.assertEqual(len(scene.world.collision_objects), 3)

            # Check box
            box = scene.world.collision_objects[0]
            self._assert_collision_object(
                box,
                'Box_0',
                [-0.32, 0.02, 0],
                [0, 0, 0, 1],
                SolidPrimitive.BOX,
                [0.2, 0.6, 0.8]
            )

            # Check sphere
            sphere = scene.world.collision_objects[1]
            self._assert_collision_object(
                sphere,
                'Sphere_0',
                [0, 0, 0],
                [0, 0, 0, 1],
                SolidPrimitive.SPHERE,
                [0.1]
            )

            # Check cylinder
            cylinder = scene.world.collision_objects[2]
            self._assert_collision_object(
                cylinder,
                'Cylinder_0',
                [-0.8, 0, 0],
                [0, 0, 0, 1],
                SolidPrimitive.CYLINDER,
                [0.8, 0.05]
            )
        finally:
            os.unlink(scene_file)

    def _create_scene_file(self, content: str) -> str:
        """
        Create a temporary scene file with the given content.

        Args
        ----
            content: Content to write to the scene file.

        Returns
        -------
            str: Path to the created scene file.

        """
        fd, path = tempfile.mkstemp(suffix='.scene')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            return path
        except Exception:
            os.unlink(path)
            raise

    def _assert_collision_object(
        self,
        obj: CollisionObject,
        expected_id: str,
        expected_pos: List[float],
        expected_ori: List[float],
        expected_type: int,
        expected_dims: List[float]
    ):
        """
        Assert that a collision object matches the expected values.

        Args
        ----
            obj: The collision object to check against expected values.
            expected_id: The expected identifier for the collision object.
            expected_pos: The expected position [x, y, z] of the collision object.
            expected_ori: The expected orientation [x, y, z, w] of the collision object.
            expected_type: The expected primitive type (e.g., BOX, SPHERE, CYLINDER).
            expected_dims: The expected dimensions of the collision object.

        Returns
        -------
            None

        """
        self.assertEqual(obj.id, expected_id)
        self.assertEqual(obj.header.frame_id, 'world')
        self.assertEqual(len(obj.primitives), 1)
        self.assertEqual(len(obj.primitive_poses), 1)

        pose = obj.primitive_poses[0]
        self.assertEqual(pose.position.x, expected_pos[0])
        self.assertEqual(pose.position.y, expected_pos[1])
        self.assertEqual(pose.position.z, expected_pos[2])
        self.assertEqual(pose.orientation.x, expected_ori[0])
        self.assertEqual(pose.orientation.y, expected_ori[1])
        self.assertEqual(pose.orientation.z, expected_ori[2])
        self.assertEqual(pose.orientation.w, expected_ori[3])

        primitive = obj.primitives[0]
        self.assertEqual(primitive.type, expected_type)
        self.assertEqual(list(primitive.dimensions), expected_dims)


if __name__ == '__main__':
    unittest.main()
