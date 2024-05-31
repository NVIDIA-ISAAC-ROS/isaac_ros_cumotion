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

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "isaac_ros_moveit_goal_setter/goal_setter_node.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace manipulation
{

GoalSetterNode::GoalSetterNode(std::string name, const rclcpp::NodeOptions & options)
: node_{std::make_shared<rclcpp::Node>(name, options)},
  planner_group_name_(node_->declare_parameter<std::string>(
      "planner_group_name", "ur_manipulator")),
  planner_id_(node_->declare_parameter<std::string>("planner_id", "cuMotion")),
  end_effector_link_(node_->declare_parameter<std::string>("end_effector_link", "wrist_3_link")),
  move_group_interface_{moveit::planning_interface::MoveGroupInterface(node_, planner_group_name_)}
{
  set_target_pose_service_ =
    node_->create_service<isaac_ros_goal_setter_interfaces::srv::SetTargetPose>(
    "set_target_pose", std::bind(
      &GoalSetterNode::SetTargetPoseCallback, this, std::placeholders::_1, std::placeholders::_2));
  ConfigureMoveit();
}

void GoalSetterNode::ConfigureMoveit()
{
  // Initialize the move group interface
  move_group_interface_.setPlannerId(planner_id_);
  RCLCPP_INFO(node_->get_logger(), "Planner ID : %s", move_group_interface_.getPlannerId().c_str());

  move_group_interface_.setEndEffectorLink(end_effector_link_);
  RCLCPP_INFO(node_->get_logger(), "End Effector Link : %s", end_effector_link_.c_str());
}

void GoalSetterNode::SetTargetPoseCallback(
  const std::shared_ptr<isaac_ros_goal_setter_interfaces::srv::SetTargetPose_Request> req,
  std::shared_ptr<isaac_ros_goal_setter_interfaces::srv::SetTargetPose_Response> res)
{
  res->success = false;
  RCLCPP_DEBUG(node_->get_logger(), "Triggered SetTargetPoseCallback");
  RCLCPP_DEBUG(
    node_->get_logger(), "Pose : x=%f, y=%f, z=%f, qx=%f, qy=%f, qz=%f, qw=%f",
    req->pose.pose.position.x, req->pose.pose.position.y, req->pose.pose.position.z,
    req->pose.pose.orientation.x, req->pose.pose.orientation.y, req->pose.pose.orientation.z,
    req->pose.pose.orientation.w);

  auto success = move_group_interface_.setPoseTarget(req->pose, end_effector_link_);
  if (!success) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to set target pose!");
    return;
  }

  auto const [status, plan] = [this] {
      moveit::planning_interface::MoveGroupInterface::Plan msg;
      auto const ok = static_cast<bool>(move_group_interface_.plan(msg));
      return std::make_pair(ok, msg);
    }();

  // Execute the plan
  if (status) {
    RCLCPP_ERROR(node_->get_logger(), "Executing!");
    move_group_interface_.execute(plan);
    res->success = true;
  } else {
    RCLCPP_ERROR(node_->get_logger(), "Planning failed!");
  }

}

} // namespace manipulation
} // namespace isaac_ros
} // namespace nvidia

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto goal_setter_node = std::make_shared<nvidia::isaac_ros::manipulation::GoalSetterNode>(
    "moveit_goal_setter",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(goal_setter_node->GetNode());
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
