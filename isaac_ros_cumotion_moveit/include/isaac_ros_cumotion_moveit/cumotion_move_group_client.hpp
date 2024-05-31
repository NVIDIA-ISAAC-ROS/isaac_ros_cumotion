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

#ifndef ISAAC_ROS_CUMOTION_MOVE_GROUP_CLIENT_H
#define ISAAC_ROS_CUMOTION_MOVE_GROUP_CLIENT_H

#include <future>
#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit_msgs/action/move_group.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

class CumotionMoveGroupClient
{
  using GoalHandle = rclcpp_action::ClientGoalHandle<moveit_msgs::action::MoveGroup>;

public:
  CumotionMoveGroupClient(const rclcpp::Node::SharedPtr & node);

  bool sendGoal();

  void updateGoal(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & req);

  void getGoal();

  bool result_ready;
  bool success;
  moveit_msgs::msg::MotionPlanDetailedResponse plan_response;

private:
  void goalResponseCallback(const GoalHandle::SharedPtr & future);

  void feedbackCallback(
    GoalHandle::SharedPtr,
    const std::shared_ptr<const moveit_msgs::action::MoveGroup::Feedback> feedback);

  void resultCallback(const GoalHandle::WrappedResult & result);

  bool get_goal_handle_;
  bool get_result_handle_;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::CallbackGroup::SharedPtr client_cb_group_;
  rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SharedPtr client_;
  rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SendGoalOptions send_goal_options_;
  std::shared_future<GoalHandle::SharedPtr> goal_h_;
  std::shared_future<GoalHandle::WrappedResult> result_future_;
  moveit_msgs::msg::PlanningScene planning_scene_;
  planning_interface::MotionPlanRequest planning_request_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_MOVE_GROUP_CLIENT_H
