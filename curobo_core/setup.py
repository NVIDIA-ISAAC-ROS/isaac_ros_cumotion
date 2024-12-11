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

import importlib.util
from pathlib import Path
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from ament_index_python.packages import get_resource
from setuptools import find_namespace_packages, setup

ISAAC_ROS_COMMON_PATH = get_resource(
    'isaac_ros_common_scripts_path',
    'isaac_ros_common'
)[0]

ISAAC_ROS_COMMON_VERSION_INFO = Path(ISAAC_ROS_COMMON_PATH) / 'isaac_ros_common-version-info.py'

spec = importlib.util.spec_from_file_location(
    'isaac_ros_common_version_info',
    ISAAC_ROS_COMMON_VERSION_INFO
)

isaac_ros_common_version_info = importlib.util.module_from_spec(spec)
sys.modules['isaac_ros_common_version_info'] = isaac_ros_common_version_info
spec.loader.exec_module(isaac_ros_common_version_info)

from isaac_ros_common_version_info import GenerateVersionInfoCommand  # noqa: E402, I100

package_name = 'curobo_core'

extra_cuda_args = {
    'nvcc': [
        '--threads=8',
        '-O3',
        '--ftz=true',
        '--fmad=true',
        '--prec-div=false',
        '--prec-sqrt=false',
    ]
}
# create a list of modules to be compiled:
ext_modules = [
    CUDAExtension(
        'curobo.curobolib.lbfgs_step_cu',
        [
            'curobo/src/curobo/curobolib/cpp/lbfgs_step_cuda.cpp',
            'curobo/src/curobo/curobolib/cpp/lbfgs_step_kernel.cu',
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        'curobo.curobolib.kinematics_fused_cu',
        [
            'curobo/src/curobo/curobolib/cpp/kinematics_fused_cuda.cpp',
            'curobo/src/curobo/curobolib/cpp/kinematics_fused_kernel.cu',
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        'curobo.curobolib.geom_cu',
        [
            'curobo/src/curobo/curobolib/cpp/geom_cuda.cpp',
            'curobo/src/curobo/curobolib/cpp/sphere_obb_kernel.cu',
            'curobo/src/curobo/curobolib/cpp/pose_distance_kernel.cu',
            'curobo/src/curobo/curobolib/cpp/self_collision_kernel.cu',
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        'curobo.curobolib.line_search_cu',
        [
            'curobo/src/curobo/curobolib/cpp/line_search_cuda.cpp',
            'curobo/src/curobo/curobolib/cpp/line_search_kernel.cu',
            'curobo/src/curobo/curobolib/cpp/update_best_kernel.cu',
        ],
        extra_compile_args=extra_cuda_args,
    ),
    CUDAExtension(
        'curobo.curobolib.tensor_step_cu',
        [
            'curobo/src/curobo/curobolib/cpp/tensor_step_cuda.cpp',
            'curobo/src/curobo/curobolib/cpp/tensor_step_kernel.cu',
        ],
        extra_compile_args=extra_cuda_args,
    ),
]

setup(
    name=package_name,
    version='3.0.0',
    packages=find_namespace_packages(where='curobo/src'),
    package_dir={'': 'curobo/src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Isaac ROS Maintainers',
    maintainer_email='isaac-ros-maintainers@nvidia.com',
    description='This package wraps the cuRobo library as a ROS 2 package. '
                'cuRobo serves as the current backend for cuMotion.',
    license='NVIDIA Isaac ROS Software License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension,
        'build_py': GenerateVersionInfoCommand,
    },
    package_data={
        'curobo': ['**/*.*'],
    },
    include_package_data=True,
)
