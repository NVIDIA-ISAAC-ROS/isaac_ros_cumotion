// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "isaac_ros_cumotion_controllers/bimanual_ik_controller.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>

#include "cumotion/rotation3.h"
#include "pluginlib/class_list_macros.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

constexpr std::string_view kLeftTargetName = "left";
constexpr std::string_view kRightTargetName = "right";

std::optional<geometry_msgs::msg::Pose> FindValidNamedTarget(
  const teleop_ros2_interfaces::msg::NamedPoseArray & msg,
  std::string_view target_name)
{
  if (msg.name.size() != msg.pose.size() || msg.name.size() != msg.is_valid.size()) {
    return std::nullopt;
  }
  for (size_t i = 0; i < msg.name.size(); ++i) {
    if (std::string_view(msg.name[i]) != target_name) {
      continue;
    }
    if (!msg.is_valid[i] || !IsReferencePoseValid(msg.pose[i])) {
      return std::nullopt;
    }
    return msg.pose[i];
  }
  return std::nullopt;
}

controller_interface::CallbackReturn BimanualIkController::DeclareSubclassParameters()
{
  left_ee_frame_name_in_ = auto_declare<std::string>(
    "left_end_effector_frame", "left_hand_palm_link");
  right_ee_frame_name_in_ = auto_declare<std::string>(
    "right_end_effector_frame", "right_hand_palm_link");
  left_ee_command_frame_name_ = auto_declare<std::string>(
    "left_ee_command_frame", left_ee_frame_name_in_);
  right_ee_command_frame_name_ = auto_declare<std::string>(
    "right_ee_command_frame", right_ee_frame_name_in_);
  drift_reset_threshold_ = auto_declare<double>("drift_reset_threshold", 0.5);

  // Fixed gains + v_cmd=0 match the GR00T safety_controller chaining contract.
  kp_command_value_ = auto_declare<double>("kp", 20.0);
  kd_command_value_ = auto_declare<double>("kd", 1.0);
  zero_velocity_command_ = true;

  if (!std::isfinite(drift_reset_threshold_) || drift_reset_threshold_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "drift_reset_threshold must be positive and finite (got %f)", drift_reset_threshold_);
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

void BimanualIkController::ConfigureEndEffectors()
{
  left_ee_frame_handle_ = kinematics_->frame(left_ee_frame_name_in_);
  right_ee_frame_handle_ = kinematics_->frame(right_ee_frame_name_in_);
  left_ee_frame_name_ = kinematics_->frameName(left_ee_frame_handle_);
  right_ee_frame_name_ = kinematics_->frameName(right_ee_frame_handle_);
  rmpflow_->addTargetFrame(left_ee_frame_name_);
  rmpflow_->addTargetFrame(right_ee_frame_name_);
  // cuMotion can silently substitute a parent link for a non-kinematics frame.
  RCLCPP_INFO(get_node()->get_logger(),
    "BimanualIkController EE frames: left '%s' -> '%s', right '%s' -> '%s'",
    left_ee_frame_name_in_.c_str(), left_ee_frame_name_.c_str(),
    right_ee_frame_name_in_.c_str(), right_ee_frame_name_.c_str());
}

void BimanualIkController::SubscribeToReferencePose(
  rclcpp_lifecycle::LifecycleNode & node, const std::string & pose_topic)
{
  pose_sub_ = node.create_subscription<teleop_ros2_interfaces::msg::NamedPoseArray>(
    pose_topic, rclcpp::SensorDataQoS(),
    [this](teleop_ros2_interfaces::msg::NamedPoseArray::SharedPtr msg) {
      if (msg->name.size() != msg->pose.size() || msg->name.size() != msg->is_valid.size()) {
        RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Malformed NamedPoseArray reference: name=%zu pose=%zu is_valid=%zu — ignoring",
          msg->name.size(), msg->pose.size(), msg->is_valid.size());
        return;
      }

      const auto left_pose = FindValidNamedTarget(*msg, kLeftTargetName);
      const auto right_pose = FindValidNamedTarget(*msg, kRightTargetName);
      if (!left_pose || !right_pose) {
        RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Reference pose name rejected or unset: left_valid=%d right_valid=%d",
          static_cast<int>(left_pose.has_value()), static_cast<int>(right_pose.has_value()));
      }

      PoseTargets targets;
      targets.left = left_pose ? ExtractAndTransformPose(
        msg->header, *left_pose, left_ee_command_frame_name_, left_ee_frame_name_) :
      std::nullopt;
      targets.right = right_pose ? ExtractAndTransformPose(
        msg->header, *right_pose, right_ee_command_frame_name_, right_ee_frame_name_) :
      std::nullopt;
      pose_targets_buffer_.writeFromNonRT(targets);
    });
}

void BimanualIkController::OnSubclassActivate()
{
  const auto n_dof = static_cast<int>(joint_names_.size());
  rmpflow_->setPoseTarget(left_ee_frame_name_,
    kinematics_->pose(joint_position_integrated_, left_ee_frame_handle_));
  rmpflow_->setPoseTarget(right_ee_frame_name_,
    kinematics_->pose(joint_position_integrated_, right_ee_frame_handle_));
  // Attract the null space toward the zero configuration, not the live
  // integrated state — an intentional difference from the single-arm
  // IkController (which attracts toward joint_position_integrated_).
  rmpflow_->setCSpaceAttractor(Eigen::VectorXd::Zero(n_dof));
}

void BimanualIkController::OnSubclassDeactivate()
{
  if (rmpflow_) {
    rmpflow_->clearPoseTarget(left_ee_frame_name_);
    rmpflow_->clearPoseTarget(right_ee_frame_name_);
  }
}

void BimanualIkController::OnSubclassCleanup()
{
  // Drop the live NamedPoseArray subscriber BEFORE the base nulls tf_buffer_ —
  // otherwise a message arriving during cleanup would dereference the null
  // tf_buffer_ inside ExtractAndTransformPose.
  pose_sub_.reset();
  pose_targets_buffer_.writeFromNonRT(PoseTargets{});
}

void BimanualIkController::ApplyPoseTargetsToRmpflow()
{
  const auto targets = *pose_targets_buffer_.readFromRT();
  if (targets.right && IsPoseDataFinite(*targets.right)) {
    rmpflow_->setPoseTarget(right_ee_frame_name_,
      cumotion::Pose3(cumotion::Rotation3(targets.right->second), targets.right->first));
  }
  if (targets.left && IsPoseDataFinite(*targets.left)) {
    rmpflow_->setPoseTarget(left_ee_frame_name_,
      cumotion::Pose3(cumotion::Rotation3(targets.left->second), targets.left->first));
  }
}

void BimanualIkController::SyncIntegratorToHardware([[maybe_unused]] double dt)
{
  // Episodic hard-snap on L2 drift; the integrator runs open-loop otherwise.
  const double drift = (joint_position_integrated_ - joint_position_).norm();
  if (drift > drift_reset_threshold_) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Integrator drift %.3f rad exceeds threshold %.3f — resetting to hardware state",
      drift, drift_reset_threshold_);
    joint_position_integrated_ = joint_position_;
    joint_velocity_integrated_ = joint_velocity_;
  }
}

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

PLUGINLIB_EXPORT_CLASS(
  nvidia::isaac_ros::cumotion_controllers::BimanualIkController,
  controller_interface::ControllerInterface)
