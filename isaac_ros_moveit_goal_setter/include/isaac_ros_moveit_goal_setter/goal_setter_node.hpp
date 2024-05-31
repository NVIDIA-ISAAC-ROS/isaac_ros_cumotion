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

#ifndef ISAAC_ROS_MOVEIT_GOAL_SETTER__GOAL_SETTER_NODE_HPP_
#define ISAAC_ROS_MOVEIT_GOAL_SETTER__GOAL_SETTER_NODE_HPP_

#include <memory>

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_goal_setter_interfaces/srv/set_target_pose.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace manipulation
{

class GoalSetterNode
{
public:
  GoalSetterNode(std::string name, const rclcpp::NodeOptions & options);
  ~GoalSetterNode() = default;

  std::shared_ptr<rclcpp::Node> GetNode() const {return node_;}

  void ConfigureMoveit();

private:
  void SetTargetPoseCallback(
    const std::shared_ptr<isaac_ros_goal_setter_interfaces::srv::SetTargetPose_Request> req,
    std::shared_ptr<isaac_ros_goal_setter_interfaces::srv::SetTargetPose_Response> res);

  const std::shared_ptr<rclcpp::Node> node_;
  std::string planner_group_name_;
  std::string planner_id_;
  std::string end_effector_link_;
  moveit::planning_interface::MoveGroupInterface move_group_interface_;

  rclcpp::Service<isaac_ros_goal_setter_interfaces::srv::SetTargetPose>::SharedPtr
    set_target_pose_service_;

};

}  // namespace manipulation
}  // namespace isaac_ros
}  // namespace nvidia


#endif //  ISAAC_ROS_MOVEIT_GOAL_SETTER__GOAL_SETTER_NODE_HPP_
