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

from typing import List

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive


class MoveItSceneFileReader:
    """Class for parsing MoveIt scene files and creating collision objects."""

    def __init__(self):
        """Initialize the scene file reader."""
        pass

    def parse_scene_file(self, scene_file_path: str) -> PlanningScene:
        """
        Parse Moveit scene file and create PlanningScene containing collision objects.

        Args
        ----
            scene_file_path: Path to the scene file.

        Returns
        -------
            PlanningScene: A PlanningScene message containing the parsed objects.

        Raises
        ------
            FileNotFoundError: If the scene file does not exist.
            ValueError: If the scene file is invalid or contains unsupported objects.

        """
        try:
            with open(scene_file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise FileNotFoundError(f'Failed to read scene file: {str(e)}')

        if not lines:
            raise ValueError('Empty scene file')

        if not lines[0].endswith('+'):
            raise ValueError('Missing header')

        planning_scene = PlanningScene()
        planning_scene.world.collision_objects = []
        planning_scene.is_diff = True

        # Track object IDs to detect duplicates
        object_ids = set()

        i = 1
        while i < len(lines):
            if not lines[i].startswith('*'):
                i += 1
                continue

            if i + 9 >= len(lines):
                raise ValueError('Missing required parameters')

            try:
                obj_id = lines[i][2:].strip()
                if obj_id in object_ids:
                    raise ValueError(f'Duplicate object ID: {obj_id}')
                object_ids.add(obj_id)

                obj = self._create_collision_object(lines[i:i+6])
                planning_scene.world.collision_objects.append(obj)
                i += 1
            except ValueError as e:
                raise ValueError(f'Error parsing object starting at line {i+1}: {str(e)}')

        return planning_scene

    def _create_collision_object(self, lines: List[str]) -> CollisionObject:
        """
        Create a collision object from the given lines.

        Args
        ----
            lines: List of lines containing object data.

        Returns
        -------
            CollisionObject: The created collision object.

        Raises
        ------
            ValueError: If the object data is invalid.

        """
        obj = CollisionObject()
        obj.header.frame_id = 'world'

        obj.id = lines[0][2:].strip()
        if not obj.id:
            raise ValueError('Invalid object ID')

        try:
            pos = list(map(float, lines[1].split()))
            if len(pos) != 3:
                raise ValueError('Invalid position values')
        except Exception:
            raise ValueError('Invalid position values')

        try:
            ori = list(map(float, lines[2].split()))
            if len(ori) != 4:
                raise ValueError('Invalid orientation values')
        except Exception:
            raise ValueError('Invalid orientation values')

        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.x = ori[0]
        pose.orientation.y = ori[1]
        pose.orientation.z = ori[2]
        pose.orientation.w = ori[3]

        shape_type = lines[4].lower()
        try:
            dims = list(map(float, lines[5].split()))
        except Exception:
            raise ValueError('Invalid dimensions format')

        primitive = self._create_primitive(shape_type, dims)
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)

        return obj

    def _create_primitive(self, shape_type: str, dims: List[float]) -> SolidPrimitive:
        """
        Create a solid primitive based on shape type and dimensions.

        Args
        ----
            shape_type: Type of shape (box, sphere, cylinder).
            dims: List of dimensions for the shape.

        Returns
        -------
            SolidPrimitive: The created primitive, or None if shape type is unsupported.

        """
        primitive = SolidPrimitive()

        if shape_type == 'box':
            if len(dims) != 3:
                raise ValueError('Invalid dimensions for box')
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = dims
        elif shape_type == 'sphere':
            if len(dims) != 1:
                raise ValueError('Invalid dimensions for sphere')
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = dims
        elif shape_type == 'cylinder':
            if len(dims) != 2:
                raise ValueError('Invalid dimensions for cylinder')
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [dims[1], dims[0]]
        else:
            raise ValueError('Unsupported shape type')

        return primitive
