cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_cumotion_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(moveit_common REQUIRED)
moveit_package()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  rclcpp_components
  moveit_core
)

include_directories(include)

set(MOVEIT_LIB_NAME isaac_ros_cumotion_moveit)

# cuMotion planning plugin
add_library(${MOVEIT_LIB_NAME}
  SHARED
  src/cumotion_planner_manager.cpp
  src/cumotion_interface.cpp
  src/cumotion_planning_context.cpp
  src/cumotion_move_group_client.cpp)

set_target_properties(${MOVEIT_LIB_NAME} PROPERTIES VERSION "${${PROJECT_NAME}_VERSION}")

ament_target_dependencies(${MOVEIT_LIB_NAME} ${THIS_PACKAGE_INCLUDE_DEPENDS})


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${MOVEIT_LIB_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install config files.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)
pluginlib_export_plugin_description_file(moveit_core cumotion_planner_plugin_description.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
