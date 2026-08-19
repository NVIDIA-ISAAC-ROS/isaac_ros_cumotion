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

#ifndef ISAAC_ROS_CUMOTION_CONTROLLERS__IK_CONTROLLER_BASE_HPP_
#define ISAAC_ROS_CUMOTION_CONTROLLERS__IK_CONTROLLER_BASE_HPP_

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "isaac_ros_cumotion_controllers/controller_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "cumotion/kinematics.h"
#include "cumotion/rmpflow.h"
#include "isaac_ros_inverse_dynamics/inverse_dynamics_solver.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

/// Shared base for the single-arm and bimanual cuMotion IK controllers.
///
/// Owns the URDF/XRDF loading, the cuMotion + Pinocchio model + integrator
/// state, the hardware-interface plumbing, and the per-cycle RNEA
/// feed-forward torque computation. Subclasses pick the pose-message shape,
/// the number of end-effector targets, the integrator-sync strategy, and the
/// kp/kd command policy.
class IkControllerBase : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

protected:
  // ---- Customization points ---------------------------------------------

  /// Declare subclass-specific parameters (extra EE frames, drift_reset_threshold,
  /// kp/kd, etc.). Called from on_configure() after the base parameters are
  /// declared but before any of them are used. Return ERROR to abort configure.
  virtual controller_interface::CallbackReturn DeclareSubclassParameters() = 0;

  /// Look up subclass EE frame handles from `kinematics_` and add them as
  /// rmpflow target frames. Called from on_configure() once kinematics_ and
  /// rmpflow_ are ready.
  virtual void ConfigureEndEffectors() = 0;

  /// Create the pose-target subscriber on `node` listening on `pose_topic`.
  /// Called from on_configure() once tf_buffer_ / tf_listener_ exist.
  virtual void SubscribeToReferencePose(
    rclcpp_lifecycle::LifecycleNode & node, const std::string & pose_topic) = 0;

  /// Reset rmpflow pose targets and c-space attractor from the initial joint
  /// state captured into `joint_position_integrated_`. Called from on_activate().
  virtual void OnSubclassActivate() = 0;

  /// Clear rmpflow pose targets. Called from on_deactivate().
  virtual void OnSubclassDeactivate() = 0;

  /// Release subscriber + pose-target buffer. Called from on_cleanup() before
  /// base members are reset.
  virtual void OnSubclassCleanup() {}

  /// Per-cycle hook: pull the latest reference pose(s) from the subclass-owned
  /// realtime buffer and push to rmpflow_ via setPoseTarget. Called from
  /// update() after the integrator sync, before evalAccel.
  virtual void ApplyPoseTargetsToRmpflow() = 0;

  /// Per-cycle hook: bring `joint_position_integrated_` / `joint_velocity_integrated_`
  /// closer to the measured `joint_position_` / `joint_velocity_`. The two
  /// concrete strategies (continuous low-pass vs hard-snap on drift) are
  /// genuinely different and intentionally so.
  virtual void SyncIntegratorToHardware(double dt) = 0;

  // ---- Helpers ----------------------------------------------------------

  /// Apply prefix and suffix independently — either, both, or neither may be
  /// set. Matches the isaac_ros_deploy_ros2_control _raw chaining convention.
  std::string MakeCommandInterfaceName(
    const std::string & joint, const std::string & iface) const;

  /// Resolve a `geometry_msgs/Pose` (in `header.frame_id`) into a base-frame
  /// PoseData targeting `ee_frame` through `ee_command_frame`. Returns nullopt
  /// on TF lookup failure (already logged with WARN_THROTTLE).
  std::optional<PoseData> ExtractAndTransformPose(
    const std_msgs::msg::Header & header,
    const geometry_msgs::msg::Pose & pose,
    const std::string & ee_command_frame,
    const std::string & ee_frame) const;

  /// Cap RMPflow joint velocity at `max_velocity` (rad/s) and derive its
  /// braking-ramp width. Applied in on_configure() only when the
  /// `max_joint_velocity` parameter is positive.
  static void ApplyVelocityCap(cumotion::RmpFlowConfig & config, double max_velocity);

  // ---- Shared state (protected so subclasses can read/write) ------------

  // Parameters set in on_configure()
  std::vector<std::string> joint_names_{};
  std::string command_prefix_{};
  std::string command_suffix_{};

  // Per-joint command policy. Defaults: NaN kp/kd to defer to safety_controller's
  // fallback gains; velocity = v_target_(i). Subclasses tweak in
  // DeclareSubclassParameters() if they want fixed gains or v=0.
  double kp_command_value_{std::numeric_limits<double>::quiet_NaN()};
  double kd_command_value_{std::numeric_limits<double>::quiet_NaN()};
  bool zero_velocity_command_{false};

  // cuMotion
  std::unique_ptr<cumotion::RmpFlow> rmpflow_{nullptr};
  std::unique_ptr<cumotion::Kinematics> kinematics_{nullptr};
  std::string base_frame_{};

  // Feed-forward inverse dynamics. The shared solver assumes single-DOF,
  // fixed-base joints (idx_qs == idx_vs); it lives in the isaac_ros_deploy
  // module (maintainer: dtzoumanikas). A controller joint missing from the
  // model is rejected at configure (see on_configure) rather than silently
  // contributing zero torque.
  std::unique_ptr<isaac_ros_inverse_dynamics::InverseDynamicsSolver> id_solver_{nullptr};
  Eigen::VectorXd tau_ff_{};

  // Hardware interface index maps — built once in on_activate.
  struct HardwareIndexMaps
  {
    std::vector<size_t> pos_state{};
    std::vector<size_t> vel_state{};
    std::vector<size_t> pos_cmd{};
    std::vector<size_t> vel_cmd{};
    std::vector<size_t> effort_cmd{};
    std::vector<size_t> kp_cmd{};
    std::vector<size_t> kd_cmd{};
  };
  HardwareIndexMaps hw_{};

  // Per-cycle joint state — measured, integrated, and target.
  Eigen::VectorXd joint_position_{};
  Eigen::VectorXd joint_velocity_{};
  Eigen::VectorXd joint_accel_{};
  Eigen::VectorXd joint_position_integrated_{};
  Eigen::VectorXd joint_velocity_integrated_{};
  // Preallocated, reused per update() to keep the hot path allocation-free.
  Eigen::VectorXd q_target_{};
  Eigen::VectorXd v_target_{};

  // TF — populated in on_configure(), used from the pose subscriber callback.
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_{nullptr};
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
};

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_CONTROLLERS__IK_CONTROLLER_BASE_HPP_
