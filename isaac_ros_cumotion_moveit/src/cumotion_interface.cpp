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

#include "isaac_ros_cumotion_moveit/cumotion_interface.hpp"

#include <chrono>
#include <memory>

#include "moveit/planning_interface/planning_interface.h"
#include "moveit/planning_scene/planning_scene.h"
#include "moveit/robot_state/conversions.h"
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

namespace
{

constexpr unsigned kSleepIntervalInMs = 5;
constexpr unsigned kTimeoutIntervalInSeconds = 5;

}  // namespace

bool CumotionInterface::solve(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & request,
  planning_interface::MotionPlanDetailedResponse & response)
{
  RCLCPP_INFO(node_->get_logger(), "Planning trajectory");

  if (!planner_busy) {
    action_client_->updateGoal(planning_scene, request);
    action_client_->sendGoal();
    planner_busy = true;
  }

  rclcpp::Time start_time = node_->now();
  while (
    !action_client_->result_ready &&
    node_->now().seconds() - start_time.seconds() < kTimeoutIntervalInSeconds)
  {
    action_client_->getGoal();
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepIntervalInMs));
  }

  if (!action_client_->result_ready) {
    RCLCPP_ERROR(node_->get_logger(), "Timed out!");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
    planner_busy = false;
    return false;
  }
  RCLCPP_INFO(node_->get_logger(), "Received trajectory result");

  if (!action_client_->success) {
    RCLCPP_ERROR(node_->get_logger(), "No trajectory");
    response.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    planner_busy = false;
    return false;
  }
  RCLCPP_INFO(node_->get_logger(), "Trajectory success!");

  response.error_code_ = action_client_->plan_response.error_code;
  response.description_ = action_client_->plan_response.description;
  auto result_traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    planning_scene->getRobotModel(), request.group_name);
  moveit::core::RobotState robot_state(planning_scene->getRobotModel());
  moveit::core::robotStateMsgToRobotState(
    action_client_->plan_response.trajectory_start,
    robot_state);
  result_traj->setRobotTrajectoryMsg(
    robot_state,
    action_client_->plan_response.trajectory[0]);
  response.trajectory_.clear();
  response.trajectory_.push_back(result_traj);
  response.processing_time_ = action_client_->plan_response.processing_time;

  planner_busy = false;
  return true;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia
