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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_cumotion_robot_segmenter LANGUAGES CUDA CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Resolve cuMotion path from the vendor package.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_index_get_resource(CUMOTION_RELATIVE_PATH cumotion cumotion_vendor)
ament_index_get_prefix_path(AMENT_INDEX_PREFIX_PATHS)
foreach(PREFIX_PATH IN LISTS AMENT_INDEX_PREFIX_PATHS)
  set(CUMOTION_CONFIG_FILE_IN_PREFIX
    "${PREFIX_PATH}/${CUMOTION_RELATIVE_PATH}/lib/cmake/cumotion/cumotionConfig.cmake")
  if(EXISTS "${CUMOTION_CONFIG_FILE_IN_PREFIX}")
    set(CUMOTION_CONFIG_FILE "${CUMOTION_CONFIG_FILE_IN_PREFIX}")
    break()
  endif()
endforeach()

if(NOT CUMOTION_CONFIG_FILE)
  message(FATAL_ERROR
    "Could not find cuMotion CMake config from cumotion_vendor resource "
    "'${CUMOTION_RELATIVE_PATH}'.")
endif()

get_filename_component(cumotion_DIR "${CUMOTION_CONFIG_FILE}" DIRECTORY)
find_package(cumotion REQUIRED CONFIG)

# Additional dependencies (specified here as these are non-ament dependencies).
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

# NVTX
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
  add_definitions(-DUSE_NVTX)
  link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  link_libraries("CUDA::nvtx3")
endif()

# RobotSegmenterNode
ament_auto_add_library(robot_segmenter_node SHARED
  src/robot_segmenter.cpp
)
target_link_libraries(robot_segmenter_node
  CUDA::cudart
  Eigen3::Eigen
  cumotion::cumotion
  yaml-cpp
)
set_target_properties(robot_segmenter_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Register RobotSegmenter as a composable node.
rclcpp_components_register_nodes(
  robot_segmenter_node
  "nvidia::isaac_ros::manipulator::RobotSegmenter"
)

# Tests.
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_robot_segmenter_float16.py)
  add_launch_test(test/test_robot_segmenter_pol.py)
  add_launch_test(test/test_robot_segmenter_reload.py)
  add_launch_test(test/test_robot_segmenter_uint16.py)
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch test params)
