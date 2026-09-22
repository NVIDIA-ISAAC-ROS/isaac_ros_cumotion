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

#ifndef ISAAC_ROS_CUMOTION_CONTROLLERS__BIMANUAL_IK_CONTROLLER_HPP_
#define ISAAC_ROS_CUMOTION_CONTROLLERS__BIMANUAL_IK_CONTROLLER_HPP_

#include <optional>
#include <string>

#include "controller_interface/controller_interface.hpp"
#include "isaac_ros_cumotion_controllers/controller_utils.hpp"
#include "isaac_ros_cumotion_controllers/ik_controller_base.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "teleop_ros2_interfaces/msg/named_pose_array.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

struct PoseTargets
{
  std::optional<PoseData> right{};
  std::optional<PoseData> left{};
};

/// Bimanual cuMotion IK controller. Subscribes to a `NamedPoseArray` with
/// `left` and `right` end-effector targets, runs an open-loop integrator
/// hard-snapped to hardware on L2 drift, and writes fixed kp / kd command
/// values (v_cmd held at 0 for the GR00T deployment).
class BimanualIkController : public IkControllerBase
{
public:
  BimanualIkController() = default;

protected:
  controller_interface::CallbackReturn DeclareSubclassParameters() override;
  void ConfigureEndEffectors() override;
  void SubscribeToReferencePose(
    rclcpp_lifecycle::LifecycleNode & node, const std::string & pose_topic) override;
  void OnSubclassActivate() override;
  void OnSubclassDeactivate() override;
  void OnSubclassCleanup() override;
  void ApplyPoseTargetsToRmpflow() override;
  void SyncIntegratorToHardware(double dt) override;

private:
  // Parameters
  std::string left_ee_frame_name_in_{};
  std::string right_ee_frame_name_in_{};
  std::string left_ee_command_frame_name_{};
  std::string right_ee_command_frame_name_{};
  double drift_reset_threshold_{};

  // Resolved EE
  cumotion::Kinematics::FrameHandle left_ee_frame_handle_{};
  cumotion::Kinematics::FrameHandle right_ee_frame_handle_{};
  std::string left_ee_frame_name_{};
  std::string right_ee_frame_name_{};

  // ROS
  rclcpp::Subscription<teleop_ros2_interfaces::msg::NamedPoseArray>::SharedPtr pose_sub_{nullptr};
  realtime_tools::RealtimeBuffer<PoseTargets> pose_targets_buffer_{};
};

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_CONTROLLERS__BIMANUAL_IK_CONTROLLER_HPP_
