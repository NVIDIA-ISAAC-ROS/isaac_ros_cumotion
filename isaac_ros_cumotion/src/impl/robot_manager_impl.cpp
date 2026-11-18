// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_cumotion/impl/robot_manager_impl.hpp"

#include <cumotion/cumotion.h>
#include <cumotion/kinematics.h>
#include <cumotion/robot_description.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/logging.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Constructor implementation.
RobotManagerImpl::RobotManagerImpl(
  const std::string & urdf_file_path,
  const std::string & xrdf_file_path,
  const rclcpp::Logger & logger)
: urdf_file_path_(urdf_file_path),
  xrdf_file_path_(xrdf_file_path),
  logger_(logger)
{
  RCLCPP_INFO(logger_, "Creating and initializing RobotManagerImpl");

  // Load robot description from URDF and XRDF files.
  RCLCPP_INFO(logger_, "Loading robot configuration from URDF and XRDF files");
  RCLCPP_INFO(logger_, "  URDF: %s", urdf_file_path_.c_str());
  RCLCPP_INFO(logger_, "  XRDF: %s", xrdf_file_path_.c_str());

  if (urdf_file_path_.empty() || xrdf_file_path_.empty()) {
    RCLCPP_FATAL(logger_, "URDF path or XRDF path is empty");
    throw std::runtime_error("Failed to load robot description: URDF or XRDF path is empty");
  }

  /**
   * Cache file contents so we can serve them over GetRobotDescription and later swap via
   * SetRobotDescription.
   */
  robot_urdf_ = ReadFileAsString(urdf_file_path_);
  robot_xrdf_ = ReadYamlFileAsString(xrdf_file_path_);

  // Load from memory so we can support runtime swapping via SetRobotDescription().
  robot_description_ = cumotion_lib::LoadRobotFromMemory(robot_xrdf_, robot_urdf_);

  if (!robot_description_) {
    RCLCPP_FATAL(logger_, "Failed to load robot description from files");
    throw std::runtime_error("Failed to load robot description");
  }

  RCLCPP_INFO(
    logger_, "Robot description loaded successfully.");

  // Get kinematics from robot description.
  kinematics_ = std::shared_ptr<cumotion_lib::Kinematics>(robot_description_->kinematics());
  if (!kinematics_) {
    RCLCPP_FATAL(logger_, "Failed to create kinematics solver");
    throw std::runtime_error("Failed to create kinematics solver");
  }

  // Get tool frame handle if specified.
  auto available_tool_frames = robot_description_->toolFrameNames();
  if (!tool_frame_.empty()) {
    // Validate that the specified tool frame exists in the robot description.
    bool frame_found = std::find(
      available_tool_frames.begin(), available_tool_frames.end(), tool_frame_) !=
      available_tool_frames.end();
    if (!frame_found) {
      RCLCPP_FATAL(
        logger_, "Specified tool frame '%s' not found in robot description",
        tool_frame_.c_str());
      throw std::runtime_error("Invalid tool frame: " + tool_frame_);
    }
    tool_frame_handle_ = kinematics_->frame(tool_frame_);
    RCLCPP_INFO(logger_, "Using specified tool frame: %s", tool_frame_.c_str());
  } else {
    // Use the first tool frame from the robot description.
    if (!available_tool_frames.empty()) {
      tool_frame_ = available_tool_frames[0];
      tool_frame_handle_ = kinematics_->frame(tool_frame_);
      RCLCPP_INFO(logger_, "Using default tool frame: %s", tool_frame_.c_str());
    } else {
      RCLCPP_FATAL(logger_, "No tool frames available in robot description");
      throw std::runtime_error("No tool frames available");
    }
  }

  // Get base frame.
  robot_base_frame_ = kinematics_->frameName(kinematics_->baseFrame());
  RCLCPP_INFO(logger_, "Robot base frame: %s", robot_base_frame_.c_str());

  // Build joint-name index map for c-space ordering.
  BuildJointNameToIndexMap();
}

void RobotManagerImpl::BuildJointNameToIndexMap()
{
  joint_name_to_index_.clear();
  if (!robot_description_) {
    return;
  }

  const int num_joints = robot_description_->numCSpaceCoords();
  for (int i = 0; i < num_joints; ++i) {
    const std::string name = robot_description_->cSpaceCoordName(i);
    joint_name_to_index_[name] = static_cast<std::size_t>(i);
  }
}

void RobotManagerImpl::UpdateJointState(
  const std::vector<std::string> & joint_names,
  const std::vector<double> & position,
  const std::vector<double> & velocity)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  /**
   * Lazily initialize the joint state buffer layout on first message, using the
   * robot's configuration-space joint ordering.
   */
  if (js_buffer_.joint_names.empty()) {
    const std::size_t num_joints = joint_name_to_index_.size();
    js_buffer_.joint_names.resize(num_joints);
    js_buffer_.position.resize(num_joints, 0.0);
    js_buffer_.velocity.resize(num_joints, 0.0);

    // Fill joint_names in index order.
    for (const auto & entry : joint_name_to_index_) {
      const std::size_t idx = entry.second;
      if (idx < js_buffer_.joint_names.size()) {
        js_buffer_.joint_names[idx] = entry.first;
      }
    }
  }

  /**
   * Update buffer entries in-place instead of growing the vectors.
   * Any joints in the incoming message that are not part of the robot's
   * configuration space (e.g., gripper joints) are ignored via the joint_name_to_index_ lookup.
   */
  for (std::size_t i = 0; i < joint_names.size(); ++i) {
    const auto & joint_name = joint_names[i];

    auto it = joint_name_to_index_.find(joint_name);
    if (it == joint_name_to_index_.end()) {
      continue;
    }

    const std::size_t idx = it->second;
    if (idx >= js_buffer_.position.size()) {
      continue;
    }

    if (i < position.size()) {
      js_buffer_.position[idx] = position[i];
    }
    if (i < velocity.size()) {
      js_buffer_.velocity[idx] = velocity[i];
    }
  }
}

// Robot description accessor.
std::shared_ptr<cumotion_lib::RobotDescription> RobotManagerImpl::GetRobotDescription() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return robot_description_;
}

bool RobotManagerImpl::SetRobotDescription(const std::string & urdf, const std::string & xrdf)
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);

  if (urdf.empty() || xrdf.empty()) {
    RCLCPP_ERROR(
      logger_,
      "Cannot set robot description: URDF or XRDF is empty (URDF: %zu bytes, XRDF: %zu bytes)",
      urdf.size(), xrdf.size());
    return false;
  }

  robot_urdf_ = urdf;
  robot_xrdf_ = xrdf;

  RCLCPP_INFO(
    logger_,
    "Loading robot description from memory (URDF: %zu bytes, XRDF: %zu bytes)",
    robot_urdf_.size(), robot_xrdf_.size());

  robot_description_ = cumotion_lib::LoadRobotFromMemory(robot_xrdf_, robot_urdf_);
  if (!robot_description_) {
    RCLCPP_FATAL(logger_, "Failed to load robot description from memory");
    return false;
  }

  kinematics_ = std::shared_ptr<cumotion_lib::Kinematics>(robot_description_->kinematics());
  if (!kinematics_) {
    RCLCPP_FATAL(logger_, "Failed to create kinematics solver");
    return false;
  }

  // Resolve tool frame handle again (keep the configured name if provided; otherwise pick default).
  auto available_tool_frames = robot_description_->toolFrameNames();
  if (!tool_frame_.empty()) {
    // Validate that the specified tool frame exists in the robot description.
    bool frame_found = std::find(
      available_tool_frames.begin(), available_tool_frames.end(), tool_frame_) !=
      available_tool_frames.end();
    if (!frame_found) {
      RCLCPP_FATAL(
        logger_, "Specified tool frame '%s' not found in robot description",
        tool_frame_.c_str());
      return false;
    }
    tool_frame_handle_ = kinematics_->frame(tool_frame_);
  } else {
    if (!available_tool_frames.empty()) {
      tool_frame_ = available_tool_frames[0];
      tool_frame_handle_ = kinematics_->frame(tool_frame_);
    } else {
      RCLCPP_FATAL(logger_, "No tool frames available in robot description");
      return false;
    }
  }

  robot_base_frame_ = kinematics_->frameName(kinematics_->baseFrame());

  /**
   * Refresh joint ordering map and force joint buffer re-init to new ordering.
   * Both joint_name_to_index_ and js_buffer_ must be updated atomically under state_mutex_
   * to prevent data races with UpdateJointState().
   */
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    joint_name_to_index_.clear();
    if (robot_description_) {
      const int num_joints = robot_description_->numCSpaceCoords();
      for (int i = 0; i < num_joints; ++i) {
        const std::string name = robot_description_->cSpaceCoordName(i);
        joint_name_to_index_[name] = static_cast<std::size_t>(i);
      }
    }
    js_buffer_.joint_names.clear();
    js_buffer_.position.clear();
    js_buffer_.velocity.clear();
  }

  return true;
}

std::string RobotManagerImpl::GetXRDF() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return robot_xrdf_;
}

std::string RobotManagerImpl::GetURDF() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return robot_urdf_;
}

// Kinematics accessor.
std::shared_ptr<cumotion_lib::Kinematics> RobotManagerImpl::GetKinematics() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return kinematics_;
}

// Tool frame handle accessor.
cumotion_lib::Kinematics::FrameHandle RobotManagerImpl::GetToolFrameHandle() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return tool_frame_handle_;
}

// Tool frame name accessor.
std::string RobotManagerImpl::GetToolFrame() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return tool_frame_;
}

// Base frame name accessor.
std::string RobotManagerImpl::GetBaseFrame() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  return robot_base_frame_;
}

// Joint names accessor.
std::vector<std::string> RobotManagerImpl::GetJointNames() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  if (!robot_description_) {
    RCLCPP_WARN(logger_, "Robot description not loaded, returning empty joint names");
    return {};
  }

  std::vector<std::string> names;
  for (int i = 0; i < robot_description_->numCSpaceCoords(); ++i) {
    names.push_back(robot_description_->cSpaceCoordName(i));
  }
  return names;
}

// Number of joints accessor.
int RobotManagerImpl::GetNumJoints() const
{
  std::lock_guard<std::mutex> lock(robot_description_mutex_);
  if (!robot_description_) {
    return 0;
  }
  return robot_description_->numCSpaceCoords();
}

// Get current joint state implementation.
bool RobotManagerImpl::GetCurrentJointState(Eigen::VectorXd & state) const
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!js_buffer_.IsValid()) {
    return false;
  }

  state = Eigen::Map<const Eigen::VectorXd>(
    js_buffer_.position.data(),
    js_buffer_.position.size());

  return true;
}

bool RobotManagerImpl::MapJointPositions(
  const std::vector<std::string> & names,
  const std::vector<double> & values,
  Eigen::VectorXd & position,
  std::string & error) const
{
  error.clear();
  position.resize(joint_name_to_index_.size());
  for (const auto & joint : joint_name_to_index_) {
    const auto & name = joint.first;
    const auto i = joint.second;
    const auto it = std::find(names.begin(), names.end(), name);
    const auto input_index = static_cast<std::size_t>(std::distance(names.begin(), it));
    if (it != names.end()) {
      if (input_index >= values.size()) {
        error = "Missing position for named controlled joint '" + name + "'";
        return false;
      }
      if (!std::isfinite(values[input_index])) {
        error = "Non-finite position for controlled joint '" + name + "'";
        return false;
      }
      position(i) = values[input_index];
    } else {
      error = "Missing position for controlled joint '" + name + "'";
      return false;
    }
  }
  return true;
}

bool RobotManagerImpl::ResolveMoveItStartCSpacePosition(
  const moveit_msgs::msg::RobotState & state,
  Eigen::VectorXd & position,
  std::string & error) const
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  return MapJointPositions(
    state.joint_state.name, state.joint_state.position, position, error);
}

bool RobotManagerImpl::ResolveMoveItGoalCSpacePosition(
  const std::vector<moveit_msgs::msg::JointConstraint> & constraints,
  Eigen::VectorXd & position,
  std::string & error) const
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  std::vector<std::string> names;
  std::vector<double> values;
  names.reserve(constraints.size());
  values.reserve(constraints.size());
  for (const auto & constraint : constraints) {
    if (joint_name_to_index_.find(constraint.joint_name) == joint_name_to_index_.end()) {
      error = "Unsupported goal joint '" + constraint.joint_name + "'";
      return false;
    }
    names.push_back(constraint.joint_name);
    values.push_back(constraint.position);
  }
  return MapJointPositions(names, values, position, error);
}

// Check if valid joint state is available.
bool RobotManagerImpl::HasValidJointState() const
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  return js_buffer_.IsValid();
}

// Get joint state buffer.
JointStateBuffer RobotManagerImpl::GetJointStateBuffer() const
{
  // Lock protects js_buffer_ while making a copy. Caller receives an independent copy,
  // eliminating data race concerns after the lock is released.
  std::lock_guard<std::mutex> lock(state_mutex_);
  return js_buffer_;
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
