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

#ifndef ISAAC_ROS_CUMOTION_CONTROLLERS__IK_CONTROLLER_HPP_
#define ISAAC_ROS_CUMOTION_CONTROLLERS__IK_CONTROLLER_HPP_

#include <optional>
#include <string>

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "isaac_ros_cumotion_controllers/controller_utils.hpp"
#include "isaac_ros_cumotion_controllers/ik_controller_base.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

/// Single-arm cuMotion IK controller. Subscribes to a `PoseStamped` reference
/// target, runs an open-loop integrator continuously low-pass synced to
/// hardware state, and writes NaN kp/kd so a downstream SafetyController
/// defers to its per-joint fallback gains.
class IkController : public IkControllerBase
{
public:
  IkController() = default;

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
  std::string ee_frame_name_in_{};  // requested EE frame; resolved name lands in ee_frame_name_
  std::string ee_command_frame_name_{};
  // First-order low-pass time constant (seconds) for syncing the open-loop
  // integrator to hardware state every update(). 0 disables the sync.
  double integrator_sync_time_constant_{0.2};

  // Resolved EE
  cumotion::Kinematics::FrameHandle ee_frame_handle_{};
  std::string ee_frame_name_{};

  // ROS
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_{nullptr};
  realtime_tools::RealtimeBuffer<std::optional<PoseData>> pose_target_buffer_{};
};

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_CONTROLLERS__IK_CONTROLLER_HPP_
