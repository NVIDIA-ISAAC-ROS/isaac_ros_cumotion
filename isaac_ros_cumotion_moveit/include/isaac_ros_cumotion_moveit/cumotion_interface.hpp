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

#ifndef ISAAC_ROS_CUMOTION_INTERFACE_H
#define ISAAC_ROS_CUMOTION_INTERFACE_H

#include <memory>

#include "moveit/planning_interface/planning_interface.hpp"
#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_cumotion_moveit/cumotion_move_group_client.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

class CumotionInterface
{
public:
  CumotionInterface(const rclcpp::Node::SharedPtr & node)
  : node_(node),
    action_client_(std::make_shared<CumotionMoveGroupClient>(node))
  {
  }

  void solve(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & request,
    planning_interface::MotionPlanDetailedResponse & response);

  bool planner_busy = false;

private:
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<CumotionMoveGroupClient> action_client_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_INTERFACE_H
