cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_cumotion_controllers)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers)
endif()

# Find ament dependencies declared in package.xml.
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Resolve cuMotion from the ament index (exported by isaac_ros_nitros).
ament_index_get_resource(CUMOTION_RELATIVE_PATH cumotion isaac_ros_nitros)
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
    "Could not find cuMotion CMake config from isaac_ros_nitros resource "
    "'${CUMOTION_RELATIVE_PATH}'.")
endif()

get_filename_component(cumotion_DIR "${CUMOTION_CONFIG_FILE}" DIRECTORY)
find_package(cumotion REQUIRED CONFIG)

# Controllers library. The inverse-dynamics solver (and its Pinocchio
# dependency) is pulled in via the ament <depend>isaac_ros_inverse_dynamics</depend>.
ament_auto_add_library(isaac_ros_cumotion_controllers SHARED
  src/ik_controller_base.cpp
  src/bimanual_ik_controller.cpp
  src/ik_controller.cpp
)
target_link_libraries(isaac_ros_cumotion_controllers
  cumotion::cumotion
)
set_target_properties(isaac_ros_cumotion_controllers PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Export controller plugin.
pluginlib_export_plugin_description_file(
  controller_interface isaac_ros_cumotion_controllers.xml)

# Install scripts.
install(PROGRAMS
  scripts/ik_controller_marker_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()
