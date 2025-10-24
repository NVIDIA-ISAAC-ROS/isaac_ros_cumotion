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

#ifndef ISAAC_ROS_CUMOTION_PLANNING_CONTEXT_H
#define ISAAC_ROS_CUMOTION_PLANNING_CONTEXT_H

#include <memory>
#include <string>

#include "moveit/planning_interface/planning_interface.hpp"

#include "isaac_ros_cumotion_moveit/cumotion_interface.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

class CumotionPlanningContext : public planning_interface::PlanningContext
{
public:
  CumotionPlanningContext(
    const std::string & context_name,
    const std::string & group_name,
    const rclcpp::Node::SharedPtr & node)
  : planning_interface::PlanningContext(context_name, group_name),
    cumotion_interface_(std::make_shared<CumotionInterface>(node))
  {
  }

  ~CumotionPlanningContext() override
  {
  }

  void solve(planning_interface::MotionPlanResponse & res) override;

  void solve(planning_interface::MotionPlanDetailedResponse & res) override;

  bool terminate() override
  {
    return true;
  }

  void clear() override
  {
  }

private:
  std::shared_ptr<CumotionInterface> cumotion_interface_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_PLANNING_CONTEXT_H
