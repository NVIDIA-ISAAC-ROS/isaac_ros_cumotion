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

#include "isaac_ros_cumotion_controllers/ik_controller_base.hpp"

#include <filesystem>

#include "cumotion/cumotion.h"
#include "cumotion/robot_description.h"
#include "cumotion/world.h"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

void IkControllerBase::ApplyVelocityCap(cumotion::RmpFlowConfig & config, double max_velocity)
{
  constexpr double kDampingRegionFraction = 0.5;
  config.setParam("joint_velocity_cap_rmp/max_velocity", max_velocity);
  config.setParam(
    "joint_velocity_cap_rmp/velocity_damping_region",
    max_velocity * kDampingRegionFraction);
}

std::string IkControllerBase::MakeCommandInterfaceName(
  const std::string & joint, const std::string & iface) const
{
  std::string name = joint + "/" + iface;
  if (!command_prefix_.empty()) {name = command_prefix_ + "/" + name;}
  if (!command_suffix_.empty()) {name += command_suffix_;}
  return name;
}

std::optional<PoseData> IkControllerBase::ExtractAndTransformPose(
  const std_msgs::msg::Header & header, const geometry_msgs::msg::Pose & pose,
  const std::string & ee_command_frame, const std::string & ee_frame) const
{
  try {
    tf2::Transform cmd_parent_T_cmd;
    tf2::fromMsg(pose, cmd_parent_T_cmd);

    const auto base_T_cmd_parent_msg = tf_buffer_->lookupTransform(
      base_frame_, header.frame_id, tf2_ros::fromMsg(header.stamp),
      tf2::durationFromSec(0.1));
    tf2::Transform base_T_cmd_parent;
    tf2::fromMsg(base_T_cmd_parent_msg.transform, base_T_cmd_parent);

    const auto base_T_cmd = base_T_cmd_parent * cmd_parent_T_cmd;

    const auto cmd_T_ee_msg = tf_buffer_->lookupTransform(
      ee_command_frame, ee_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));
    tf2::Transform cmd_T_ee;
    tf2::fromMsg(cmd_T_ee_msg.transform, cmd_T_ee);

    const auto base_T_ee = base_T_cmd * cmd_T_ee;

    const auto & p = base_T_ee.getOrigin();
    const auto & q = base_T_ee.getRotation();
    return std::make_pair(
      Eigen::Vector3d{p.x(), p.y(), p.z()},
      Eigen::Quaterniond{q.w(), q.x(), q.y(), q.z()});
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Failed to transform pose from '%s' to '%s': %s",
      header.frame_id.c_str(), base_frame_.c_str(), ex.what());
    return std::nullopt;
  }
}

controller_interface::CallbackReturn IkControllerBase::on_init()
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn IkControllerBase::on_configure(
  const rclcpp_lifecycle::State &)
{
  joint_names_ = auto_declare<std::vector<std::string>>("joints", {});
  const std::filesystem::path urdf_path = auto_declare<std::string>("urdf_path", "");
  const std::filesystem::path xrdf_path = auto_declare<std::string>("xrdf_path", "");
  const std::filesystem::path rmpflow_config = auto_declare<std::string>("rmpflow_config_path", "");
  command_prefix_ = auto_declare<std::string>("command_prefix", "");
  command_suffix_ = auto_declare<std::string>("command_suffix", "");
  const auto pose_topic = auto_declare<std::string>("pose_topic", "~/reference_pose");
  const double max_joint_velocity = auto_declare<double>("max_joint_velocity", -1.0);

  if (const auto rc = DeclareSubclassParameters();
    rc != controller_interface::CallbackReturn::SUCCESS) {return rc;}

  auto & n = *get_node();
  if (joint_names_.empty()) {
    RCLCPP_ERROR(n.get_logger(), "No joints specified");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (urdf_path.empty() || xrdf_path.empty() || rmpflow_config.empty()) {
    RCLCPP_ERROR(n.get_logger(), "urdf_path, xrdf_path, and rmpflow_config_path must all be set");
    return controller_interface::CallbackReturn::ERROR;
  }

  auto robot_description = cumotion::LoadRobotFromFile(xrdf_path, urdf_path);
  if (!robot_description) {
    RCLCPP_ERROR(n.get_logger(), "Failed to load robot description from '%s' / '%s'",
      xrdf_path.c_str(), urdf_path.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (static_cast<int>(joint_names_.size()) != robot_description->numCSpaceCoords()) {
    RCLCPP_ERROR(n.get_logger(),
      "Joint count mismatch: controller has %zu joints but robot model has %d DOF",
      joint_names_.size(), robot_description->numCSpaceCoords());
    return controller_interface::CallbackReturn::ERROR;
  }

  kinematics_ = robot_description->kinematics();
  base_frame_ = kinematics_->frameName(kinematics_->baseFrame());

  auto world = cumotion::CreateWorld();
  auto rmpflow_cfg = cumotion::CreateRmpFlowConfigFromFile(
    rmpflow_config, *robot_description, world->addWorldView());
  if (!rmpflow_cfg) {
    RCLCPP_ERROR(n.get_logger(), "Failed to load RMPflow config from '%s'",
      rmpflow_config.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  if (max_joint_velocity > 0.0) {
    ApplyVelocityCap(*rmpflow_cfg, max_joint_velocity);
  }
  rmpflow_ = cumotion::CreateRmpFlow(*rmpflow_cfg);

  ConfigureEndEffectors();

  // Validate every controller joint against cuMotion's actuated-joint (cSpace)
  // names: the RMPflow/kinematics state we feed in is ordered by joint_names_, so a
  // joint absent from the planning model would corrupt the IK solve. The
  // inverse-dynamics map is validated separately against the solver's own model
  // below, since that model is built independently from the URDF.
  for (const auto & joint : joint_names_) {
    bool found = false;
    for (int i = 0; i < robot_description->numCSpaceCoords() && !found; ++i) {
      found = robot_description->cSpaceCoordName(i) == joint;
    }
    if (!found) {
      RCLCPP_ERROR(n.get_logger(),
        "Joint '%s' not found in the cuMotion planning model — refusing to configure",
        joint.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
  }

  // Feed-forward inverse dynamics (tau = M*a + C*v + G) via the shared solver.
  try {
    id_solver_ = std::make_unique<isaac_ros_inverse_dynamics::InverseDynamicsSolver>(
      urdf_path.string(), joint_names_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(n.get_logger(), "Failed to build inverse-dynamics solver from '%s': %s",
      urdf_path.c_str(), e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  // The solver builds its own Pinocchio model from the URDF and silently zeroes
  // torque for any controller joint it can't map; a fully empty map already throws
  // above. Reject a partial map loudly here so a stale config can't ship an
  // incomplete inverse-dynamics term against the model the solver actually uses.
  if (id_solver_->num_mapped_joints() != joint_names_.size()) {
    RCLCPP_ERROR(n.get_logger(),
      "Inverse-dynamics model maps only %zu of %zu controller joints — refusing to "
      "configure with an incomplete inverse-dynamics map",
      id_solver_->num_mapped_joints(), joint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(n.get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  SubscribeToReferencePose(n, pose_topic);

  RCLCPP_INFO(n.get_logger(), "%s configured: %zu joints",
    get_node()->get_name(), joint_names_.size());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn IkControllerBase::on_activate(
  const rclcpp_lifecycle::State &)
{
  const auto n_dof = static_cast<int>(joint_names_.size());
  joint_accel_ = Eigen::VectorXd::Zero(n_dof);
  joint_position_integrated_ = Eigen::VectorXd::Zero(n_dof);
  joint_velocity_integrated_ = Eigen::VectorXd::Zero(n_dof);
  joint_position_ = Eigen::VectorXd::Zero(n_dof);
  joint_velocity_ = Eigen::VectorXd::Zero(n_dof);
  tau_ff_ = Eigen::VectorXd::Zero(n_dof);
  q_target_ = Eigen::VectorXd::Zero(n_dof);
  v_target_ = Eigen::VectorXd::Zero(n_dof);

  auto make_names = [&](const std::string & iface, bool cmd) -> std::vector<std::string> {
      std::vector<std::string> out;
      out.reserve(joint_names_.size());
      for (const auto & j : joint_names_) {
        out.push_back(cmd ? MakeCommandInterfaceName(j, iface) : j + "/" + iface);
      }
      return out;
    };
  auto find_state = [&](const std::string & iface) -> std::optional<std::vector<size_t>> {
      return FindInterfaceIndices(make_names(iface, false), state_interfaces_);
    };
  auto find_cmd = [&](const std::string & iface) -> std::optional<std::vector<size_t>> {
      return FindInterfaceIndices(make_names(iface, true), command_interfaces_);
    };

  const auto pos_s = find_state(hardware_interface::HW_IF_POSITION);
  const auto vel_s = find_state(hardware_interface::HW_IF_VELOCITY);
  const auto pos_c = find_cmd(hardware_interface::HW_IF_POSITION);
  const auto vel_c = find_cmd(hardware_interface::HW_IF_VELOCITY);
  const auto eff_c = find_cmd(hardware_interface::HW_IF_EFFORT);
  const auto kp_c = find_cmd("kp");
  const auto kd_c = find_cmd("kd");
  const auto missing =
    !pos_s ? "<joint>/position state" :
    !vel_s ? "<joint>/velocity state" :
    !pos_c ? "<prefix>/<joint>/position<suffix> command" :
    !vel_c ? "<prefix>/<joint>/velocity<suffix> command" :
    !eff_c ? "<prefix>/<joint>/effort<suffix> command" :
    !kp_c ? "<prefix>/<joint>/kp<suffix> command" :
    !kd_c ? "<prefix>/<joint>/kd<suffix> command" : nullptr;
  if (missing) {
    RCLCPP_ERROR(get_node()->get_logger(),
      "Failed to find required interface '%s' (with command_prefix='%s' command_suffix='%s'). "
      "Check the URDF / controller_manager wiring.",
      missing, command_prefix_.c_str(), command_suffix_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }
  hw_.pos_state = *pos_s; hw_.vel_state = *vel_s;
  hw_.pos_cmd = *pos_c; hw_.vel_cmd = *vel_c;
  hw_.effort_cmd = *eff_c; hw_.kp_cmd = *kp_c; hw_.kd_cmd = *kd_c;

  for (int i = 0; i < n_dof; ++i) {
    const auto pos = state_interfaces_[hw_.pos_state[i]].get_optional<double>();
    const auto vel = state_interfaces_[hw_.vel_state[i]].get_optional<double>();
    if (!pos || !vel) {
      RCLCPP_ERROR(get_node()->get_logger(),
        "Failed to read state for joint '%s' on activation — hardware not ready",
        joint_names_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    joint_position_integrated_(i) = *pos;
    joint_velocity_integrated_(i) = *vel;
  }
  joint_position_ = joint_position_integrated_;
  joint_velocity_ = joint_velocity_integrated_;

  OnSubclassActivate();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn IkControllerBase::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  OnSubclassDeactivate();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn IkControllerBase::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  OnSubclassCleanup();
  tf_listener_.reset();
  tf_buffer_.reset();
  rmpflow_.reset();
  kinematics_.reset();
  id_solver_.reset();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
IkControllerBase::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & j : joint_names_) {
    config.names.push_back(MakeCommandInterfaceName(j, hardware_interface::HW_IF_POSITION));
    config.names.push_back(MakeCommandInterfaceName(j, hardware_interface::HW_IF_VELOCITY));
    config.names.push_back(MakeCommandInterfaceName(j, hardware_interface::HW_IF_EFFORT));
    config.names.push_back(MakeCommandInterfaceName(j, "kp"));
    config.names.push_back(MakeCommandInterfaceName(j, "kd"));
  }
  return config;
}

controller_interface::InterfaceConfiguration
IkControllerBase::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & j : joint_names_) {
    config.names.push_back(j + "/" + hardware_interface::HW_IF_POSITION);
    config.names.push_back(j + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::return_type IkControllerBase::update(
  const rclcpp::Time &, const rclcpp::Duration & period)
{
  const auto n_dof = static_cast<int>(joint_names_.size());
  const double dt = period.seconds();

  for (int i = 0; i < n_dof; ++i) {
    joint_position_(i) =
      state_interfaces_[hw_.pos_state[i]].get_optional<double>().value_or(joint_position_(i));
    joint_velocity_(i) =
      state_interfaces_[hw_.vel_state[i]].get_optional<double>().value_or(joint_velocity_(i));
  }

  SyncIntegratorToHardware(dt);
  ApplyPoseTargetsToRmpflow();

  rmpflow_->evalAccel(joint_position_integrated_, joint_velocity_integrated_, joint_accel_);

  v_target_.noalias() = joint_velocity_integrated_ + dt * joint_accel_;
  q_target_.noalias() = joint_position_integrated_ + dt * v_target_;

  // Feed-forward inverse dynamics: tau_ff = M(q)*a + C(q,v)*v + G(q). This is a
  // feed-forward effort term; a downstream chained stage may override it per
  // deployment (e.g. the isaac_ros_deploy SafetyController applies gravity-only
  // compensation in "overwrite" mode for the G1 arms). Writing it is still
  // correct for chains that pass effort through (e.g. franka_fr3).
  id_solver_->computeInverseDynamics(
    joint_position_integrated_, joint_velocity_integrated_, joint_accel_, tau_ff_);

  for (int i = 0; i < n_dof; ++i) {
    (void)command_interfaces_[hw_.pos_cmd[i]].set_value(q_target_(i));
    (void)command_interfaces_[hw_.vel_cmd[i]].set_value(
      zero_velocity_command_ ? 0.0 : v_target_(i));
    (void)command_interfaces_[hw_.effort_cmd[i]].set_value(tau_ff_(i));
    (void)command_interfaces_[hw_.kp_cmd[i]].set_value(kp_command_value_);
    (void)command_interfaces_[hw_.kd_cmd[i]].set_value(kd_command_value_);
  }

  joint_velocity_integrated_ += dt * joint_accel_;
  joint_position_integrated_ += dt * joint_velocity_integrated_;

  return controller_interface::return_type::OK;
}

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia
