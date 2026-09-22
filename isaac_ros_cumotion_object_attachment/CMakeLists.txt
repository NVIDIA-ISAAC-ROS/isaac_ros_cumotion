# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_cumotion_object_attachment LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# cuMotion (resolve path from ament_index)
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

# object_attachment_node library
ament_auto_add_library(
  object_attachment_node SHARED
  src/object_attachment.cpp
)
target_link_libraries(object_attachment_node
  cumotion::cumotion
  assimp
  Eigen3::Eigen
  yaml-cpp
)
target_include_directories(object_attachment_node PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
)
rclcpp_components_register_nodes(object_attachment_node "nvidia::isaac_ros::cumotion::ObjectAttachmentNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::cumotion::ObjectAttachmentNode;$<TARGET_FILE:object_attachment_node>\n")

# Set RPATH for runtime library discovery
set_target_properties(object_attachment_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_cuboid_geometry
    test/test_cuboid_geometry.cpp
  )
  target_include_directories(test_cuboid_geometry PUBLIC
    ${CUMOTION_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
  )
  target_link_libraries(test_cuboid_geometry
    object_attachment_node
  )
  ament_target_dependencies(test_cuboid_geometry
    rclcpp
  )
  set_target_properties(test_cuboid_geometry PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  add_launch_test(test/isaac_ros_object_attachment_pol_test.py TIMEOUT "60"
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_launch_test(test/isaac_ros_object_attachment_esdf_test.py TIMEOUT "60"
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch params)
