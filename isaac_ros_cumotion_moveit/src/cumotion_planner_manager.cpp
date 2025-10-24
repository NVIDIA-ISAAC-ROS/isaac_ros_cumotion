// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion_moveit/cumotion_planner_manager.hpp"

#include "moveit/planning_interface/planning_interface.hpp"
#include "moveit/planning_scene/planning_scene.hpp"
#include "pluginlib/class_list_macros.hpp"

#include "isaac_ros_cumotion_moveit/cumotion_planning_context.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

bool CumotionPlannerManager::initialize(
  const moveit::core::RobotModelConstPtr & model,
  const rclcpp::Node::SharedPtr & node,
  const std::string & parameter_namespace)
{
  node_ = node;
  for (const std::string & group_name : model->getJointModelGroupNames()) {
    planning_contexts_[group_name] =
      std::make_shared<CumotionPlanningContext>("cumotion_planning_context", group_name, node);
  }
  static_cast<void>(model);  // Suppress "unused" warning.
  static_cast<void>(parameter_namespace);  // Suppress "unused" warning.
  return true;
}

std::string CumotionPlannerManager::getDescription() const
{
  return "Generate minimum-jerk trajectories using NVIDIA Isaac ROS cuMotion";
}

void CumotionPlannerManager::getPlanningAlgorithms(std::vector<std::string> & algs) const
{
  algs.clear();
  algs.push_back(kCumotionPlannerId);
}

planning_interface::PlanningContextPtr CumotionPlannerManager::getPlanningContext(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & req,
  moveit_msgs::msg::MoveItErrorCodes & error_code) const
{
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  if (!planning_scene) {
    RCLCPP_ERROR(node_->get_logger(), "No planning scene supplied as input");
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return planning_interface::PlanningContextPtr();
  }

  if (req.group_name.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No group specified to plan for");
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return planning_interface::PlanningContextPtr();
  }

  // Retrieve and configure existing context.
  const std::shared_ptr<CumotionPlanningContext> & context = planning_contexts_.at(req.group_name);

  context->setPlanningScene(planning_scene);
  context->setMotionPlanRequest(req);

  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  return context;
}

void CumotionPlannerManager::setPlannerConfigurations(
  const planning_interface::PlannerConfigurationMap & pcs)
{
  planner_configs_ = pcs;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register the `CumotionPlannerManager` class as a plugin.
PLUGINLIB_EXPORT_CLASS(
  nvidia::isaac::manipulation::CumotionPlannerManager,
  planning_interface::PlannerManager)
