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

#include "isaac_ros_cumotion_controllers/ik_controller.hpp"

#include "cumotion/rotation3.h"
#include "pluginlib/class_list_macros.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

controller_interface::CallbackReturn IkController::DeclareSubclassParameters()
{
  ee_frame_name_in_ = auto_declare<std::string>("end_effector_frame", "");
  ee_command_frame_name_ = auto_declare<std::string>("ee_command_frame", ee_frame_name_in_);
  integrator_sync_time_constant_ = auto_declare<double>(
    "integrator_sync_time_constant", integrator_sync_time_constant_);

  if (ee_frame_name_in_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "end_effector_frame must be set");
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

void IkController::ConfigureEndEffectors()
{
  ee_frame_handle_ = kinematics_->frame(ee_frame_name_in_);
  ee_frame_name_ = kinematics_->frameName(ee_frame_handle_);
  rmpflow_->addTargetFrame(ee_frame_name_);
  // cuMotion can silently substitute a parent link for a non-kinematics frame.
  RCLCPP_INFO(get_node()->get_logger(),
    "IkController EE frame: requested '%s' -> resolved '%s'",
    ee_frame_name_in_.c_str(), ee_frame_name_.c_str());
}

void IkController::SubscribeToReferencePose(
  rclcpp_lifecycle::LifecycleNode & node, const std::string & pose_topic)
{
  pose_sub_ = node.create_subscription<geometry_msgs::msg::PoseStamped>(
    pose_topic, rclcpp::SensorDataQoS(),
    [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      if (!IsReferencePoseValid(msg->pose)) {
        RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Reference pose rejected as unset (origin + identity within tolerance)");
        return;
      }
      auto pose = ExtractAndTransformPose(
        msg->header, msg->pose, ee_command_frame_name_, ee_frame_name_);
      pose_target_buffer_.writeFromNonRT(pose);
    });
}

void IkController::OnSubclassActivate()
{
  rmpflow_->setPoseTarget(ee_frame_name_,
    kinematics_->pose(joint_position_integrated_, ee_frame_handle_));
  rmpflow_->setCSpaceAttractor(joint_position_integrated_);
  pose_target_buffer_.writeFromNonRT(std::nullopt);
}

void IkController::OnSubclassDeactivate()
{
  pose_target_buffer_.writeFromNonRT(std::nullopt);
  if (rmpflow_) {
    rmpflow_->clearPoseTarget(ee_frame_name_);
  }
}

void IkController::OnSubclassCleanup()
{
  pose_sub_.reset();
  pose_target_buffer_.writeFromNonRT(std::nullopt);
}

void IkController::ApplyPoseTargetsToRmpflow()
{
  const auto target = *pose_target_buffer_.readFromRT();
  if (target && IsPoseDataFinite(*target)) {
    rmpflow_->setPoseTarget(ee_frame_name_,
      cumotion::Pose3(cumotion::Rotation3(target->second), target->first));
  }
}

void IkController::SyncIntegratorToHardware(double dt)
{
  // Continuous low-pass: safety_controller blends blend*q_ik + (1-blend)*q_freeze
  // while the integrator runs at full speed; the LPF keeps them aligned.
  if (integrator_sync_time_constant_ <= 0.0) {return;}
  const double alpha = dt / (integrator_sync_time_constant_ + dt);
  joint_position_integrated_ +=
    alpha * (joint_position_ - joint_position_integrated_);
  joint_velocity_integrated_ +=
    alpha * (joint_velocity_ - joint_velocity_integrated_);
}

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

PLUGINLIB_EXPORT_CLASS(
  nvidia::isaac_ros::cumotion_controllers::IkController,
  controller_interface::ControllerInterface)
