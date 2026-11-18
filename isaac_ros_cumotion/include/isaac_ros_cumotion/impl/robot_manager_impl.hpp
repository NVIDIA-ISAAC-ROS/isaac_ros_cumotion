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

#ifndef ISAAC_ROS_CUMOTION__IMPL__ROBOT_MANAGER_IMPL_HPP_
#define ISAAC_ROS_CUMOTION__IMPL__ROBOT_MANAGER_IMPL_HPP_

#include <cumotion/kinematics.h>
#include <cumotion/robot_description.h>

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <moveit_msgs/msg/joint_constraint.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <rclcpp/logger.hpp>
#include "isaac_ros_cumotion/impl/utils.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

/**
 * This class is responsible for managing robot description loading from URDF/XRDF files, kinematics
 * solver, joint state subscription and buffering, and joint name/frame queries.
 */
class RobotManagerImpl
{
public:
  /**
   * Constructs a robot manager implementation.
   *
   * This constructor fully initializes the robot manager: robot description is loaded,
   * kinematics are created, tool frame handle is resolved, and internal joint ordering is built.
   *
   * Joint state updates are provided externally via UpdateJointState().
   */
  RobotManagerImpl(
    const std::string & urdf_file_path,
    const std::string & xrdf_file_path,
    const rclcpp::Logger & logger);

  // Gets the robot description object.
  std::shared_ptr<cumotion_lib::RobotDescription> GetRobotDescription() const;

  /**
   * Sets the robot description from in-memory URDF and XRDF strings.
   *
   * Robot description can be swapped at runtime via ROS services. On success, kinematics and tool
   * frame handle are recreated and internal joint-state ordering is refreshed.
   *
   * Returns true on success, false on failure.
   */
  bool SetRobotDescription(const std::string & urdf, const std::string & xrdf);

  // Gets the cached XRDF string.
  std::string GetXRDF() const;

  // Gets the cached URDF string.
  std::string GetURDF() const;

  // Gets the kinematics solver.
  std::shared_ptr<cumotion_lib::Kinematics> GetKinematics() const;

  // Gets the tool frame handle for task-space planning.
  cumotion_lib::Kinematics::FrameHandle GetToolFrameHandle() const;

  // Gets the tool frame name.
  std::string GetToolFrame() const;

  // Gets the robot base frame name.
  std::string GetBaseFrame() const;

  // Gets the list of joint names in the order expected by cuMotion.
  std::vector<std::string> GetJointNames() const;

  // Gets the number of configuration space coordinates (joints).
  int GetNumJoints() const;

  /**
   * Gets the current joint state from the buffered subscription. Returns true if valid joint
   * state is available, with the state vector populated with current joint positions.
   */
  bool GetCurrentJointState(Eigen::VectorXd & state) const;

  /**
   * Validate and map a complete controlled start state into c-space order.
   * Reject missing joint positions, including partial or empty diffs. Output is valid on success.
   */
  bool ResolveMoveItStartCSpacePosition(
    const moveit_msgs::msg::RobotState & state,
    Eigen::VectorXd & position,
    std::string & error) const;

  /**
   * Extract a complete joint goal in c-space order. Reject unsupported, missing, or
   * non-finite targets, returning false with a diagnostic in error. Output is valid on success.
   */
  bool ResolveMoveItGoalCSpacePosition(
    const std::vector<moveit_msgs::msg::JointConstraint> & constraints,
    Eigen::VectorXd & position,
    std::string & error) const;

  /**
   * Updates the internal joint state buffer using joint names and values. The provided joint
   * ordering is mapped into the cuMotion configuration-space ordering.
   *
   * Any joints that are not part of the robot's configuration space (e.g., gripper joints) are
   * ignored. This mirrors the previous ROS subscription callback behavior.
   */
  void UpdateJointState(
    const std::vector<std::string> & joint_names,
    const std::vector<double> & position,
    const std::vector<double> & velocity);

  // Checks if a valid joint state has been received.
  bool HasValidJointState() const;

  // Gets a copy of the joint state buffer.
  JointStateBuffer GetJointStateBuffer() const;

private:
  // Map named positions into c-space order. Caller must hold state_mutex_.
  bool MapJointPositions(
    const std::vector<std::string> & names,
    const std::vector<double> & values,
    Eigen::VectorXd & position,
    std::string & error) const;

  // Rebuild joint-name to index map for cuMotion c-space coordinate ordering.
  void BuildJointNameToIndexMap();

  // Path to the robot URDF file.
  std::string urdf_file_path_;

  // Path to the robot XRDF file.
  std::string xrdf_file_path_;

  // Name of the tool frame for task-space planning.
  std::string tool_frame_;

  // Logger for diagnostics.
  rclcpp::Logger logger_;

  // cuMotion robot description.
  std::shared_ptr<cumotion_lib::RobotDescription> robot_description_;

  // cuMotion kinematics solver.
  std::shared_ptr<cumotion_lib::Kinematics> kinematics_;

  // Handle to the tool frame for task-space planning.
  cumotion_lib::Kinematics::FrameHandle tool_frame_handle_;

  // Name of the robot base frame.
  std::string robot_base_frame_;

  // Buffer for storing the latest joint state.
  JointStateBuffer js_buffer_;

  // Mapping from joint name to its index in the configuration space / joint state buffer.
  std::unordered_map<std::string, std::size_t> joint_name_to_index_;

  /**
   * Mutex to protect js_buffer_ from concurrent access between callback and getter threads.
   * Mutable keyword allows locking in const getters (GetJointStateBuffer, GetCurrentJointState),
   * since mutex lock/unlock doesn't modify the object's logical state (only internal sync state).
   */
  mutable std::mutex state_mutex_;

  // Mutex to protect robot description and kinematics from concurrent swaps/reads.
  mutable std::mutex robot_description_mutex_;

  // Cached robot description strings.
  std::string robot_urdf_;
  std::string robot_xrdf_;
};

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION__IMPL__ROBOT_MANAGER_IMPL_HPP_
