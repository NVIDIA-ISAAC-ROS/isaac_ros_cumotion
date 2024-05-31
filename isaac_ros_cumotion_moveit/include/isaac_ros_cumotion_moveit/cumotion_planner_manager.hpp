// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_CUMOTION_PLANNER_MANAGER_H
#define ISAAC_ROS_CUMOTION_PLANNER_MANAGER_H

#include <map>
#include <string>
#include <vector>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"

#include "isaac_ros_cumotion_moveit/cumotion_planning_context.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

class CumotionPlannerManager : public planning_interface::PlannerManager
{
  inline static constexpr char kCumotionPlannerId[] = "cuMotion";

public:
  CumotionPlannerManager()
  {
  }

  bool initialize(
    const moveit::core::RobotModelConstPtr & model,
    const rclcpp::Node::SharedPtr & node,
    const std::string & parameter_namespace) override;

  bool canServiceRequest(const planning_interface::MotionPlanRequest & req) const override
  {
    return req.planner_id == kCumotionPlannerId;
  }

  std::string getDescription() const override;

  void getPlanningAlgorithms(std::vector<std::string> & algs) const override;

  planning_interface::PlanningContextPtr getPlanningContext(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & req,
    moveit_msgs::msg::MoveItErrorCodes & error_code) const override;

  void setPlannerConfigurations(const planning_interface::PlannerConfigurationMap & pcs) override;

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::map<std::string, std::shared_ptr<CumotionPlanningContext>> planning_contexts_;
  planning_interface::PlannerConfigurationMap planner_configs_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_PLANNER_MANAGER_H
