# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_cumotion LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Resolve cuMotion path from isaac_ros_nitros
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_index_get_resource(CUMOTION_RELATIVE_PATH cumotion isaac_ros_nitros)
ament_index_get_prefix_path(AMENT_INDEX_PREFIX_PATHS)

# Resolve the cuMotion config exported by isaac_ros_nitros.
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
    "Could not find cuMotion CMake config from isaac_ros_nitros resource "
    "'${CUMOTION_RELATIVE_PATH}'.")
endif()

# Use cuMotion CMake package exports (cumotion::cumotion target).
get_filename_component(cumotion_DIR "${CUMOTION_CONFIG_FILE}" DIRECTORY)
find_package(cumotion REQUIRED CONFIG)

# Additional dependencies (specified here as these are non-ament dependencies).
find_package(assimp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories.
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Add utils as shared library.
ament_auto_add_library(utils SHARED
  src/impl/utils.cpp
)
target_link_libraries(utils
  cumotion::cumotion
  yaml-cpp
)
set_target_properties(utils PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
ament_target_dependencies(utils
  rclcpp
  geometry_msgs
  moveit_msgs
  sensor_msgs
  shape_msgs
  trajectory_msgs
  visualization_msgs
)

# Isaac ROS cuMotion implementation library.
ament_auto_add_library(cumotion_impl SHARED
  src/impl/cumotion_logger_bridge.cpp
  src/impl/world_manager_impl.cpp
  src/impl/robot_manager_impl.cpp
  src/impl/trajectory_optimizer_impl.cpp
  src/impl/ik_solver_impl.cpp
)
target_link_libraries(cumotion_impl
  utils
  cumotion::cumotion
  yaml-cpp
  assimp
)
set_target_properties(cumotion_impl PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add cuMotion planner as shared library.
ament_auto_add_library(cumotion_planner_lib SHARED
  src/cumotion_planner.cpp
)
target_link_libraries(cumotion_planner_lib
  cumotion_impl
)
set_target_properties(cumotion_planner_lib PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Register CumotionPlanner as a composable node.
rclcpp_components_register_nodes(
  cumotion_planner_lib
  "nvidia::isaac_ros::cumotion::CumotionPlanner"
)

# Static planning scene server component.
ament_auto_add_library(static_planning_scene_lib SHARED
  src/static_planning_scene.cpp
)
ament_target_dependencies(static_planning_scene_lib
  rclcpp
  rclcpp_components
  geometry_msgs
  moveit_msgs
  shape_msgs
  isaac_ros_cumotion_interfaces
)
target_link_libraries(static_planning_scene_lib
  utils
)

# Register StaticPlanningSceneServer as a composable node.
rclcpp_components_register_nodes(
  static_planning_scene_lib
  "nvidia::isaac_ros::cumotion::StaticPlanningSceneServer"
)

# Tests.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_cumotion_logger_bridge
    test/test_cumotion_logger_bridge.cpp
  )
  target_include_directories(test_cumotion_logger_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(test_cumotion_logger_bridge
    cumotion_impl
    cumotion::cumotion
  )
  ament_target_dependencies(test_cumotion_logger_bridge
    rclcpp
  )
  set_target_properties(test_cumotion_logger_bridge PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  ament_add_gtest(test_robot_manager_impl
    test/test_robot_manager_impl.cpp
  )
  target_include_directories(test_robot_manager_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(test_robot_manager_impl
    cumotion_impl
    utils
    cumotion::cumotion
  )
  ament_target_dependencies(test_robot_manager_impl
    rclcpp
  )
  set_target_properties(test_robot_manager_impl PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  ament_add_gtest(test_trajectory_optimizer_impl
    test/test_trajectory_optimizer_impl.cpp
  )
  target_include_directories(test_trajectory_optimizer_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(test_trajectory_optimizer_impl PRIVATE
    ISAAC_ROS_CUMOTION_ROBOT_DESCRIPTION_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/../isaac_ros_cumotion_robot_description"
  )
  target_link_libraries(test_trajectory_optimizer_impl
    cumotion_impl
    utils
    cumotion::cumotion
  )
  ament_target_dependencies(test_trajectory_optimizer_impl
    rclcpp
    ament_index_cpp # Added to be able to use ament_index_cpp::get_package_share_directory().
  )
  set_target_properties(test_trajectory_optimizer_impl PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  ament_add_gtest(test_utils_compose_pose
    test/test_utils_compose_pose.cpp
  )
  target_include_directories(test_utils_compose_pose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(test_utils_compose_pose
    utils
    cumotion::cumotion
  )
  set_target_properties(test_utils_compose_pose PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  ament_add_gtest(test_moveit_scene_file_parser
    test/test_moveit_scene_file_parser.cpp
  )
  target_link_libraries(test_moveit_scene_file_parser
    utils
  )
  set_target_properties(test_moveit_scene_file_parser PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  ament_add_gtest(test_world_manager_impl
    test/test_world_manager_impl.cpp
  )
  target_include_directories(test_world_manager_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(test_world_manager_impl
    cumotion_impl
    utils
    cumotion::cumotion
  )
  ament_target_dependencies(test_world_manager_impl
    rclcpp
    geometry_msgs
    nvblox_msgs
  )
  set_target_properties(test_world_manager_impl PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
endif()

# Python script to verify ESDF self-consistency.
install(
  PROGRAMS scripts/verify_esdf_self_consistent.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE launch params)
