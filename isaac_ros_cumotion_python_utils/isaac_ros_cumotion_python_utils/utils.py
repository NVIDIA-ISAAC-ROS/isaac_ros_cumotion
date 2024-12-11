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

import math
import numpy as np
import numpy.typing as npt
from typing import List, Tuple
import yaml


def load_grid_corners_from_workspace_file(
        workspace_file_path: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """Load a workspace file and return the min / max corners specified.

    All voxels that are fully or partially contained in the bounding box
    defined by the min / max corners are part of the workspace.

    See the following file for an example:
       - isaac_manipulator_bringup/config/nvblox/workspace_bounds/zurich_test_bench.yaml

    Args:
        workspace_file_path (str): The absolute path to the workspace file.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: A tuple of two 3d vectors specifying the
                                         min and max corners of the workspace.
    """
    with open(workspace_file_path) as config_file:
        config_dict = yaml.safe_load(config_file)['/**']['ros__parameters']['static_mapper']
    min_corner = np.array([
        config_dict['workspace_bounds_min_corner_x_m'],
        config_dict['workspace_bounds_min_corner_y_m'],
        config_dict['workspace_bounds_min_height_m']
    ])
    max_corner = np.array([
        config_dict['workspace_bounds_max_corner_x_m'],
        config_dict['workspace_bounds_max_corner_y_m'],
        config_dict['workspace_bounds_max_height_m']
    ])
    return min_corner, max_corner


def get_grid_center(min_corner: List[float], grid_size: List[float]) -> List[float]:
    """Get the grid center from the minimum corner and grid size."""
    return (np.array(min_corner) + 0.5 * np.array(grid_size)).tolist()


def get_grid_min_corner(grid_center: List[float], grid_size: List[float]) -> List[float]:
    """Get the grid minimum corner from the grid center and the grid size."""
    return (np.array(grid_center) - 0.5 * np.array(grid_size)).tolist()


def get_grid_size(
        min_corner: List[float],
        max_corner: List[float],
        voxel_size: float,
        ) -> List[float]:
    """Get the grid size from corners and voxel size"""
    inv_voxel_size = 1.0 / voxel_size
    min_in_vox = [robust_floor(x * inv_voxel_size) for x in min_corner]
    max_in_vox = [robust_floor(x * inv_voxel_size) for x in max_corner]
    dims_in_vox = [max_in_vox[i] - min_in_vox[i] for i in range(len(min_in_vox))]

    dims_in_meters = [x * voxel_size for x in dims_in_vox]

    return dims_in_meters


def is_grid_valid(grid_size: List[float], voxel_size: float) -> bool:
    """Check if the grid is valid.

    Currently the only restriction is that there should be at least 1 voxel per dimension.

    Returns:
        bool: Whether the grid is valid.
    """
    num_voxels_per_dimension = np.array(grid_size) / voxel_size
    num_voxels_per_dim_is_zero = (num_voxels_per_dimension <= 0).any()
    return num_voxels_per_dim_is_zero


def robust_floor(x: float, threshold: float = 1e-04) -> int:
    """Floor float to integer when within threshold to account for floating point precision."""
    nearest_int = round(x)
    if (abs(x - nearest_int) < threshold):
        return nearest_int
    else:
        return int(math.floor(x))
