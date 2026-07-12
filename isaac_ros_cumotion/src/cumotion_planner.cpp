// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion/cumotion_planner.hpp"

#include <cumotion/cumotion.h>
#include <rcl_action/action_server.h>
#include <tf2/exceptions.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <geometry_msgs/msg/pose.hpp>
#include <isaac_ros_common/qos.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "isaac_ros_cumotion/impl/cumotion_logger_bridge.hpp"
#include "isaac_ros_cumotion/impl/utils.hpp"
#include "isaac_ros_cumotion/impl/world_manager_impl.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

CumotionPlanner::CumotionPlanner(const rclcpp::NodeOptions & options)
: Node("cumotion_action_server", options),
  // Robot configuration parameters.
  urdf_file_path_(declare_parameter<std::string>("urdf_file_path", "")),
  xrdf_file_path_(declare_parameter<std::string>("xrdf_file_path", "")),
  joint_states_topic_(declare_parameter<std::string>("joint_states_topic", "/joint_states")),
  // Trajectory interpolation parameter.
  interpolation_dt_(declare_parameter<double>("interpolation_dt", 0.025)),
  // World/ESDF parameters.
  read_esdf_world_(declare_parameter<bool>("read_esdf_world", false)),
  publish_cumotion_world_as_voxels_(
    declare_parameter<bool>("publish_cumotion_world_as_voxels", false)),
  add_ground_plane_(declare_parameter<bool>("add_ground_plane", false)),
  publish_voxel_size_(declare_parameter<double>("publish_voxel_size", 0.05)),
  max_publish_voxels_(declare_parameter<int>("max_publish_voxels", 500000)),
  ground_plane_size_x_(declare_parameter<double>("ground_plane_size_x", 2.0)),
  ground_plane_size_y_(declare_parameter<double>("ground_plane_size_y", 2.0)),
  ground_plane_thickness_(declare_parameter<double>("ground_plane_thickness", 0.1)),
  ground_plane_z_offset_(declare_parameter<double>("ground_plane_z_offset", -0.05)),
  update_esdf_on_request_(declare_parameter<bool>("update_esdf_on_request", true)),
  esdf_service_name_(
    declare_parameter<std::string>("esdf_service_name", "/nvblox_node/get_esdf_and_gradient")),
  // Service timeout parameters.
  static_scene_service_max_wait_attempts_(
    declare_parameter<int>("static_scene_service_max_wait_attempts", 30)),
  static_planning_scene_service_name_(
    declare_parameter<std::string>(
      "static_planning_scene_service_name",
      "publish_static_planning_scene")),
  time_dilation_factor_(declare_parameter<double>("time_dilation_factor", 0.5)),
  override_moveit_scaling_(declare_parameter<bool>("override_moveit_scaling_factors", false)),
  debug_mode_(declare_parameter<bool>("enable_cumotion_debug_mode", false)),
  publish_world_collision_spheres_(
    declare_parameter<bool>("publish_world_collision_spheres", true)),
  publish_self_collision_spheres_(
    declare_parameter<bool>("publish_self_collision_spheres", false))
{
  // Load world configuration from parameters.
  world_config_.read_esdf_world = read_esdf_world_;
  world_config_.publish_world_as_voxels = publish_cumotion_world_as_voxels_;
  world_config_.add_ground_plane = add_ground_plane_;
  world_config_.publish_voxel_size = publish_voxel_size_;
  world_config_.max_publish_voxels = max_publish_voxels_;
  world_config_.ground_plane_size_x = ground_plane_size_x_;
  world_config_.ground_plane_size_y = ground_plane_size_y_;
  world_config_.ground_plane_thickness = ground_plane_thickness_;
  world_config_.ground_plane_z_offset = ground_plane_z_offset_;
  world_config_.update_esdf_on_request = update_esdf_on_request_;
  world_config_.esdf_service_name = esdf_service_name_;

  // Initialize cuMotion logger.
  logger_ = std::make_shared<CumotionLoggerBridge>(this->get_name());
  nvidia::isaac_ros::cumotion::SetLogger(logger_);

  // Set log level based on debug mode.
  if (debug_mode_) {
    nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::INFO);
  } else {
    nvidia::isaac_ros::cumotion::SetLogLevel(cumotion_lib::LogLevel::WARNING);
  }

  // TF2 buffer and listener for transform lookups.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Publisher for robot collision sphere visualization (similar to Python UpdateLinkSpheresServer).
  collision_spheres_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "cumotion/collision_spheres", rclcpp::SystemDefaultsQoS());

  // Publisher for visualizing planned trajectories in RViz (similar to MoveIt display path).
  display_trajectory_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>(
    "cumotion/display_trajectory", rclcpp::SystemDefaultsQoS());

  InitializeRobotManager();
  InitializeWorldManager();
  InitializeRobotWorldInspector();
  InitializeTrajectoryOptimizer();
  InitializeIkSolver();

  InitializeMoveGroupActionServer();
  InitializeIkActionServer();
  InitializeMotionPlanActionServer();
  InitializeStaticPlanningSceneClient();
  InitializeRobotDescriptionService();

  // Initialize world objects and call static planning scene service.
  world_objects_.clear();
  CallPublishStaticPlanningSceneService();
}

void CumotionPlanner::InitializeWorldManager()
{
  world_manager_ = std::make_shared<WorldManagerImpl>(world_config_, this->get_logger());

  // Publisher for world voxel visualization.
  if (publish_cumotion_world_as_voxels_) {
    voxel_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("cumotion/voxels", 10);
  }

  // ESDF service client setup with mutually exclusive callback group.
  if (read_esdf_world_) {
    esdf_client_cb_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    esdf_client_ = this->create_client<nvblox_msgs::srv::EsdfAndGradients>(
      esdf_service_name_, rclcpp::ServicesQoS(), esdf_client_cb_group_);

    // Wait for service to become available.
    while (!esdf_client_->wait_for_service(std::chrono::seconds(15))) {
      RCLCPP_INFO(
        this->get_logger(),
        "ESDF service %s not available, waiting...",
        esdf_service_name_.c_str());
    }
    RCLCPP_INFO(
      this->get_logger(),
      "ESDF service %s is available",
      esdf_service_name_.c_str());
  }

  RCLCPP_INFO(this->get_logger(), "World manager initialized.");
}

void CumotionPlanner::InitializeRobotManager()
{
  robot_manager_ = std::make_unique<RobotManagerImpl>(
    urdf_file_path_,
    xrdf_file_path_,
    this->get_logger());
  RCLCPP_INFO(this->get_logger(), "Robot manager initialized.");

  // Update world config with robot base frame from robot manager.
  world_config_.robot_base_frame = robot_manager_->GetBaseFrame();

  // Joint-state subscription.
  joint_state_cb_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::QoS joint_state_qos = ::isaac_ros::common::AddQosParameter(
    *this, "SENSOR_DATA", "joint_state_qos");

  rclcpp::SubscriptionOptions joint_state_sub_options;
  joint_state_sub_options.callback_group = joint_state_cb_group_;

  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_,
    joint_state_qos,
    std::bind(&CumotionPlanner::JointStateCallback, this, std::placeholders::_1),
    joint_state_sub_options);
  RCLCPP_INFO(
    this->get_logger(),
    "Subscribed to joint states on topic: %s",
    joint_states_topic_.c_str());
}

void CumotionPlanner::JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (robot_description_update_in_progress_) {
      return;
    }
  }
  if (!robot_manager_) {
    return;
  }
  robot_manager_->UpdateJointState(msg->name, msg->position, msg->velocity);
  PublishCollisionSpheres();
}

void CumotionPlanner::InitializeRobotWorldInspector()
{
  robot_world_inspector_ = cumotion_lib::CreateRobotWorldInspector(
    *robot_manager_->GetRobotDescription(),
    world_manager_->GetWorldView());
  RCLCPP_INFO(this->get_logger(), "Robot-world inspector initialized.");
}

std::shared_ptr<nvblox_msgs::srv::EsdfAndGradients::Request> CumotionPlanner::CreateEsdfRequest(
  const EsdfClearingObjects * clearing_objects,
  bool update_esdf) const
{
  auto request = std::make_shared<nvblox_msgs::srv::EsdfAndGradients::Request>();

  request->visualize_esdf = true;
  request->update_esdf = update_esdf && update_esdf_on_request_;
  request->frame_id = world_config_.robot_base_frame;

  // Add clearing objects if provided.
  if (clearing_objects && clearing_objects->HasObjects()) {
    request->aabbs_to_clear_min_m = clearing_objects->aabbs_min;
    request->aabbs_to_clear_size_m = clearing_objects->aabbs_size;
    request->spheres_to_clear_center_m = clearing_objects->spheres_center;
    // Convert vector<double> to vector<float> for radius
    request->spheres_to_clear_radius_m = std::vector<float>(
      clearing_objects->spheres_radius.begin(),
      clearing_objects->spheres_radius.end());
  }

  RCLCPP_DEBUG(this->get_logger(), "Requesting ESDF from nvblox");

  return request;
}

bool CumotionPlanner::UpdateEsdfFromNvblox(
  const EsdfClearingObjects * clearing_objects, bool update_esdf)
{
  if (!read_esdf_world_ || !esdf_client_ || !world_manager_) {
    return false;
  }

  auto request = CreateEsdfRequest(clearing_objects, update_esdf);
  auto future = esdf_client_->async_send_request(request);

  // Wait for response using the mutually exclusive callback group.
  if (future.wait_for(std::chrono::seconds(15)) != std::future_status::ready) {
    RCLCPP_ERROR(this->get_logger(), "ESDF service call timed out");
    return false;
  }

  auto response = future.get();
  if (!response || !response->success) {
    return false;
  }

  return world_manager_->ProcessEsdfResponse(*response);
}

void CumotionPlanner::PublishWorldVoxels()
{
  if (!publish_cumotion_world_as_voxels_ || !voxel_pub_ ||
    voxel_pub_->get_subscription_count() == 0U || !world_manager_)
  {
    return;
  }

  auto voxels = world_manager_->CalculateOccupancyForVisualization();
  auto marker = CreateVoxelMarker(
    voxels, world_config_.robot_base_frame, world_config_.publish_voxel_size);
  marker.header.stamp = this->get_clock()->now();
  voxel_pub_->publish(marker);
}

void CumotionPlanner::InitializeTrajectoryOptimizer()
{
  auto trajopt_config = cumotion_lib::CreateDefaultTrajectoryOptimizerConfig(
    *robot_manager_->GetRobotDescription(),
    robot_manager_->GetToolFrame(),
    world_manager_->GetWorldView());
  if (!trajopt_config) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create TrajectoryOptimizerConfig");
    throw std::runtime_error("Failed to create TrajectoryOptimizerConfig");
  }

  trajectory_optimizer_ = std::make_unique<TrajectoryOptimizerImpl>(
    std::move(trajopt_config),
    interpolation_dt_,
    this->get_logger());
  RCLCPP_INFO(this->get_logger(), "Trajectory optimizer initialized.");
}

void CumotionPlanner::InitializeIkSolver()
{
  auto ik_cf_config = cumotion_lib::CreateDefaultCollisionFreeIkSolverConfig(
    *robot_manager_->GetRobotDescription(),
    robot_manager_->GetToolFrame(),
    world_manager_->GetWorldView());
  if (!ik_cf_config) {
    RCLCPP_FATAL(this->get_logger(), "Failed to create CollisionFreeIkSolverConfig");
    throw std::runtime_error("Failed to create CollisionFreeIkSolverConfig");
  }

  ik_solver_ = std::make_unique<IkSolverImpl>(
    std::move(ik_cf_config),
    this->get_logger());
  RCLCPP_INFO(this->get_logger(), "IK solver initialized.");
}

void CumotionPlanner::InitializeMoveGroupActionServer()
{
  move_group_action_server_ = rclcpp_action::create_server<moveit_msgs::action::MoveGroup>(
    this, "cumotion/move_group",
    std::bind(
      &CumotionPlanner::HandleMoveGroupGoal, this, std::placeholders::_1,
      std::placeholders::_2),
    std::bind(&CumotionPlanner::HandleMoveGroupCancel, this, std::placeholders::_1),
    std::bind(&CumotionPlanner::HandleMoveGroupAccepted, this, std::placeholders::_1));
}

void CumotionPlanner::InitializeIkActionServer()
{
  ik_action_server_ =
    rclcpp_action::create_server<IKAction>(
    this, "cumotion/ik",
    std::bind(
      &CumotionPlanner::HandleIkGoal, this, std::placeholders::_1,
      std::placeholders::_2),
    std::bind(&CumotionPlanner::HandleIkCancel, this, std::placeholders::_1),
    std::bind(&CumotionPlanner::HandleIkAccepted, this, std::placeholders::_1));
}

void CumotionPlanner::InitializeMotionPlanActionServer()
{
  // Create callback group for the action server (parity with goalset planner server).
  action_server_cb_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  motion_plan_server_ = rclcpp_action::create_server<MotionPlan>(
    this,
    "cumotion/motion_plan",
    std::bind(
      &CumotionPlanner::HandleMotionPlanGoal, this,
      std::placeholders::_1, std::placeholders::_2),
    std::bind(&CumotionPlanner::HandleMotionPlanCancel, this, std::placeholders::_1),
    std::bind(
      &CumotionPlanner::HandleMotionPlanAccepted, this,
      std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_server_cb_group_);

  RCLCPP_INFO(this->get_logger(), "MotionPlan action server initialized.");
}

void CumotionPlanner::InitializeStaticPlanningSceneClient()
{
  auto static_scene_cb_group =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  static_planning_scene_client_ =
    this->create_client<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene>(
    static_planning_scene_service_name_, rclcpp::ServicesQoS(),
    static_scene_cb_group);

  // Wait for static planning scene service.
  int wait_attempts = 0;
  while (!static_planning_scene_client_->wait_for_service(std::chrono::seconds(1))) {
    wait_attempts++;
    if (wait_attempts >= static_scene_service_max_wait_attempts_) {
      RCLCPP_FATAL(
        this->get_logger(), "Service(%s) not available after %d seconds",
        static_planning_scene_service_name_.c_str(), static_scene_service_max_wait_attempts_);
      throw std::runtime_error("Static planning scene service not available");
    }
    RCLCPP_INFO(
      this->get_logger(), "Service(%s) not available, waiting again...",
      static_planning_scene_service_name_.c_str());
  }

  RCLCPP_INFO(this->get_logger(), "Static planning scene service initialized.");
}

void CumotionPlanner::InitializeRobotDescriptionService()
{
  robot_description_cb_group_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  get_robot_description_service_ =
    this->create_service<isaac_ros_cumotion_interfaces::srv::GetRobotDescription>(
    "cumotion/get_robot_description",
    std::bind(
      &CumotionPlanner::HandleGetRobotDescription, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    robot_description_cb_group_);

  set_robot_description_service_ =
    this->create_service<isaac_ros_cumotion_interfaces::srv::SetRobotDescription>(
    "cumotion/set_robot_description",
    std::bind(
      &CumotionPlanner::HandleSetRobotDescription, this,
      std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
    robot_description_cb_group_);

  /**
   * Publish an event whenever robot description is updated.
   * Use transient-local durability so late-joining subscribers can observe the most recent update.
   */
  robot_description_updated_pub_ = this->create_publisher<std_msgs::msg::UInt64>(
    "cumotion/robot_description_updated",
    rclcpp::QoS(rclcpp::KeepLast(1)).transient_local());

  RCLCPP_INFO(this->get_logger(), "Robot description services initialized.");
}

void CumotionPlanner::HandleGetRobotDescription(
  const std::shared_ptr<isaac_ros_cumotion_interfaces::srv::GetRobotDescription::Request> request,
  std::shared_ptr<isaac_ros_cumotion_interfaces::srv::GetRobotDescription::Response> response)
{
  static_cast<void>(request);
  std::lock_guard<std::mutex> lock(robot_description_mutex_);

  if (!robot_manager_) {
    RCLCPP_WARN(this->get_logger(), "Robot manager not initialized, returning empty description");
    return;
  }

  response->urdf = robot_manager_->GetURDF();
  response->xrdf = robot_manager_->GetXRDF();

  RCLCPP_INFO(
    this->get_logger(),
    "Returning robot description (URDF: %zu bytes, XRDF: %zu bytes)",
    response->urdf.size(), response->xrdf.size());
}

void CumotionPlanner::HandleSetRobotDescription(
  const std::shared_ptr<isaac_ros_cumotion_interfaces::srv::SetRobotDescription::Request> request,
  std::shared_ptr<isaac_ros_cumotion_interfaces::srv::SetRobotDescription::Response> response)
{
  // Limit the scope of state_mutex_ to the quick state checks/flag update; release it before taking
  // robot_description_mutex_ and performing the potentially long robot-model update.
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (planner_busy_) {
      RCLCPP_ERROR(this->get_logger(), "Rejecting SetRobotDescription: planner is currently busy");
      response->success = false;
      response->message = "Planner is busy";
      return;
    }
    if (robot_description_update_in_progress_) {
      RCLCPP_ERROR(
        this->get_logger(), "Rejecting SetRobotDescription: update already in progress");
      response->success = false;
      response->message = "Robot description update already in progress";
      return;
    }
    robot_description_update_in_progress_ = true;
  }

  std::lock_guard<std::mutex> lock(robot_description_mutex_);

  if (!robot_manager_) {
    response->success = false;
    response->message = "Robot manager not initialized";
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    robot_description_update_in_progress_ = false;
    return;
  }

  if (!robot_manager_->SetRobotDescription(request->urdf, request->xrdf)) {
    response->success = false;
    response->message = "Failed to set robot description";
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    robot_description_update_in_progress_ = false;
    return;
  }

  // Update base frame for ESDF requests and visualization.
  world_config_.robot_base_frame = robot_manager_->GetBaseFrame();
  if (world_manager_) {
    world_manager_->SetRobotBaseFrame(world_config_.robot_base_frame);
  }

  // Recreate dependent components to reference the new robot description.
  InitializeRobotWorldInspector();
  InitializeTrajectoryOptimizer();
  InitializeIkSolver();

  response->success = true;
  response->message = "Robot description updated";

  if (robot_description_updated_pub_) {
    std_msgs::msg::UInt64 msg;
    msg.data = ++robot_description_update_seq_;
    robot_description_updated_pub_->publish(msg);
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    robot_description_update_in_progress_ = false;
  }
}

void CumotionPlanner::PublishCollisionSpheres()
{
  if (!collision_spheres_pub_ || collision_spheres_pub_->get_subscription_count() == 0) {
    return;
  }

  if (!robot_world_inspector_ || !robot_manager_) {
    return;
  }

  // Get current joint state from robot manager.
  Eigen::VectorXd current_state;
  if (!robot_manager_->GetCurrentJointState(current_state)) {
    return;
  }

  const auto stamp = this->get_clock()->now();
  visualization_msgs::msg::MarkerArray marker_array;
  int id = 0;
  const std::string & base_frame = robot_manager_->GetBaseFrame();

  auto append_spheres =
    [&](const std::vector<Eigen::Vector3d> & centers,
    const std::vector<double> & radii,
    const std::string & ns,
    float r, float g, float b)
    {
      const size_t count = std::min(centers.size(), radii.size());
      for (size_t i = 0; i < count; ++i) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = base_frame;
        marker.header.stamp = stamp;
        marker.ns = ns;
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = centers[i].x();
        marker.pose.position.y = centers[i].y();
        marker.pose.position.z = centers[i].z();
        marker.pose.orientation.w = 1.0;

        const double diameter = 2.0 * radii[i];
        marker.scale.x = diameter;
        marker.scale.y = diameter;
        marker.scale.z = diameter;

        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 0.5f;

        marker.lifetime = rclcpp::Duration::from_seconds(0.0);
        marker.frame_locked = true;

        marker_array.markers.push_back(std::move(marker));
      }
    };

  if (publish_world_collision_spheres_) {
    std::vector<double> world_radii;
    std::vector<Eigen::Vector3d> world_centers;
    robot_world_inspector_->worldCollisionSphereRadii(world_radii);
    robot_world_inspector_->worldCollisionSpherePositions(current_state, world_centers);
    append_spheres(world_centers, world_radii, "world_collision_spheres", 0.0f, 1.0f, 1.0f);
  }

  if (publish_self_collision_spheres_) {
    std::vector<double> self_radii;
    std::vector<Eigen::Vector3d> self_centers;
    robot_world_inspector_->selfCollisionSphereRadii(self_radii);
    robot_world_inspector_->selfCollisionSpherePositions(current_state, self_centers);
    append_spheres(self_centers, self_radii, "self_collision_spheres", 1.0f, 0.0f, 1.0f);
  }

  if (!marker_array.markers.empty()) {
    collision_spheres_pub_->publish(marker_array);
  }
}

rclcpp_action::GoalResponse CumotionPlanner::HandleMoveGroupGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const moveit_msgs::action::MoveGroup::Goal> goal)
{
  static_cast<void>(uuid);
  static_cast<void>(goal);

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (robot_description_update_in_progress_) {
    RCLCPP_ERROR(
      this->get_logger(), "Robot description update in progress, rejecting MoveGroup goal");
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (planner_busy_) {
    RCLCPP_ERROR(this->get_logger(), "Planner is busy, rejecting MoveGroup goal");
    return rclcpp_action::GoalResponse::REJECT;
  }

  RCLCPP_INFO(this->get_logger(), "Received MoveGroup planning request");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse CumotionPlanner::HandleMoveGroupCancel(
  const std::shared_ptr<GoalHandleMoveGroup> goal_handle)
{
  static_cast<void>(goal_handle);
  RCLCPP_INFO(this->get_logger(), "Received request to cancel MoveGroup goal");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void CumotionPlanner::HandleMoveGroupAccepted(
  const std::shared_ptr<GoalHandleMoveGroup> goal_handle)
{
  std::thread{std::bind(&CumotionPlanner::ExecuteMoveGroupGoal, this, goal_handle)}.detach();
}

void CumotionPlanner::ExecuteMoveGroupGoal(
  const std::shared_ptr<GoalHandleMoveGroup> goal_handle)
{
  /**
   * Update planner_busy_ under mutex, but release the lock before doing any
   * long-running planning work to avoid holding state_mutex_ longer than needed.
   */
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    planner_busy_ = true;
  }

  RCLCPP_INFO(this->get_logger(), "Executing MoveGroup goal...");

  const auto & request = goal_handle->get_goal()->request;
  auto result = std::make_shared<moveit_msgs::action::MoveGroup::Result>();

  // Get time dilation factor.
  double time_dilation = time_dilation_factor_;
  if (!override_moveit_scaling_) {
    double min_scaling = std::min(
      request.max_velocity_scaling_factor, request.max_acceleration_scaling_factor);
    if (min_scaling > 0.0) {
      time_dilation = std::min(1.0, min_scaling);
    }
  }

  RCLCPP_INFO(this->get_logger(), "Planning with time_dilation_factor: %f", time_dilation);

  // Update world with planning scene - combine static and dynamic objects.
  const auto & scene = goal_handle->get_goal()->planning_options.planning_scene_diff;
  const auto & dynamic_objects = scene.world.collision_objects;

  // Combine static objects (from scene file) with dynamic objects (from request).
  std::vector<moveit_msgs::msg::CollisionObject> all_objects = world_objects_;
  all_objects.insert(all_objects.end(), dynamic_objects.begin(), dynamic_objects.end());

  bool world_updated = world_manager_->UpdateWorldObjects(all_objects);
  // ESDF update + voxel visualization are handled here.
  if (world_updated && read_esdf_world_) {
    world_updated = UpdateEsdfFromNvblox(nullptr, true);
  }

  if (world_updated) {
    PublishWorldVoxels();
  }
  if (!world_updated) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::COLLISION_CHECKING_UNAVAILABLE;
    RCLCPP_ERROR(this->get_logger(), "World update failed");
    goal_handle->succeed(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  // Get start state.
  Eigen::VectorXd start_cspace_position;
  if (!GetStartCSpacePosition(request, start_cspace_position)) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID;
    goal_handle->succeed(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  // Get goal.
  cumotion_lib::Pose3 goal_task_space_pose;
  Eigen::VectorXd goal_cspace_position;
  bool has_pose_goal = false;
  bool has_joint_goal = false;

  // Parse and validate the goal constraints from the MoveIt request. This extracts either a
  // task-space pose target or a joint-space target, which is then sent to cuMotion.
  if (!GetGoal(
      request, goal_task_space_pose, goal_cspace_position, has_pose_goal,
      has_joint_goal))
  {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    RCLCPP_ERROR(this->get_logger(), "Failed to extract goal from MoveGroup request");
    goal_handle->succeed(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  // Plan trajectory.
  bool success = false;
  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;
  auto start_time = std::chrono::high_resolution_clock::now();

  if (has_pose_goal) {
    // Build cuMotion task-space target from the desired pose.
    cumotion_lib::TrajectoryOptimizer::TranslationConstraint translation_constraint =
      cumotion_lib::TrajectoryOptimizer::TranslationConstraint::Target(
      goal_task_space_pose.translation);
    cumotion_lib::TrajectoryOptimizer::OrientationConstraint orientation_constraint =
      cumotion_lib::TrajectoryOptimizer::OrientationConstraint::TerminalTarget(
      goal_task_space_pose.rotation);
    cumotion_lib::TrajectoryOptimizer::TaskSpaceTarget task_target(
      translation_constraint, orientation_constraint);

    success = trajectory_optimizer_->PlanToTaskSpaceTarget(
      start_cspace_position, task_target, time_dilation, trajectory, velocities,
      accelerations, dt);
  } else if (has_joint_goal) {
    cumotion_lib::TrajectoryOptimizer::CSpaceTarget cspace_target(goal_cspace_position);
    success = trajectory_optimizer_->PlanToCSpaceTarget(
      start_cspace_position, cspace_target, time_dilation, trajectory, velocities,
      accelerations, dt);
  }

  if (success) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    result->trajectory_start = request.start_state;
    auto end_time = std::chrono::high_resolution_clock::now();
    double planning_time =
      std::chrono::duration<double>(end_time - start_time).count();

    result->planned_trajectory = ToROSTrajectory(
      trajectory, velocities, accelerations, robot_manager_->GetJointNames(), dt,
      robot_manager_->GetBaseFrame());
    result->planning_time = planning_time;

    RCLCPP_INFO(this->get_logger(), "MoveGroup planning succeeded in %.3f seconds", planning_time);

    if (display_trajectory_pub_) {
      moveit_msgs::msg::DisplayTrajectory display_msg;
      display_msg.model_id = "";
      display_msg.trajectory_start = request.start_state;
      display_msg.trajectory_start.is_diff = false;
      display_msg.trajectory.clear();
      display_msg.trajectory.push_back(result->planned_trajectory);
      display_trajectory_pub_->publish(display_msg);
    }
  } else {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    RCLCPP_ERROR(this->get_logger(), "MoveGroup motion planning failed");
  }

  goal_handle->succeed(result);
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    planner_busy_ = false;
  }

  RCLCPP_INFO(
    this->get_logger(), "MoveGroup planning completed with success: %s",
    success ? "true" : "false");
}

bool CumotionPlanner::GetStartCSpacePosition(
  const moveit_msgs::msg::MotionPlanRequest & request,
  Eigen::VectorXd & start_cspace_position)
{
  const auto & js = request.start_state.joint_state;
  if (!js.position.empty()) {
    // When joint names are present, extract only the cuMotion C-space joints in XRDF order,
    // silently dropping any joints not in C-space (e.g. mimic joints added by MoveIt).
    if (!js.name.empty() && js.name.size() == js.position.size()) {
      const auto & cumotion_joints = robot_manager_->GetJointNames();
      std::unordered_map<std::string, double> name_to_pos;
      name_to_pos.reserve(js.name.size());
      for (size_t i = 0; i < js.name.size(); ++i) {
        name_to_pos[js.name[i]] = js.position[i];
      }
      start_cspace_position.resize(static_cast<Eigen::Index>(cumotion_joints.size()));
      for (size_t i = 0; i < cumotion_joints.size(); ++i) {
        auto it = name_to_pos.find(cumotion_joints[i]);
        if (it == name_to_pos.end()) {
          RCLCPP_ERROR(
            this->get_logger(),
            "Start state is missing cuMotion C-space joint '%s'",
            cumotion_joints[i].c_str());
          return false;
        }
        start_cspace_position(static_cast<Eigen::Index>(i)) = it->second;
      }
    } else {
      start_cspace_position = ToEigenVector(js.position);
    }

    if (request.start_state.is_diff) {
      Eigen::VectorXd current;
      if (robot_manager_->GetCurrentJointState(current)) {
        start_cspace_position += current;
      }
    }
    return true;
  }

  if (!robot_manager_->GetCurrentJointState(start_cspace_position)) {
    RCLCPP_ERROR(
      this->get_logger(), "No valid joint state available from robot manager");
    return false;
  }

  return true;
}

bool CumotionPlanner::GetGoal(
  const moveit_msgs::msg::MotionPlanRequest & request,
  cumotion_lib::Pose3 & goal_task_space_pose,
  Eigen::VectorXd & goal_cspace_position,
  bool & has_pose_goal,
  bool & has_joint_goal)
{
  has_pose_goal = false;
  has_joint_goal = false;

  if (request.goal_constraints.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No goal constraints specified");
    return false;
  }

  const auto & constraint = request.goal_constraints[0];

  // Check for joint constraints.
  if (!constraint.joint_constraints.empty()) {
    RCLCPP_INFO(this->get_logger(), "Using joint space goal");

    // Map constraint positions by name and select only the cuMotion C-space joints in XRDF order,
    // dropping any non-C-space joints (e.g. mimic joints forwarded by MoveIt).
    const auto & cumotion_joints = robot_manager_->GetJointNames();
    std::unordered_map<std::string, double> name_to_pos;
    name_to_pos.reserve(constraint.joint_constraints.size());
    for (const auto & jc : constraint.joint_constraints) {
      name_to_pos[jc.joint_name] = jc.position;
    }
    goal_cspace_position.resize(static_cast<Eigen::Index>(cumotion_joints.size()));
    for (size_t i = 0; i < cumotion_joints.size(); ++i) {
      auto it = name_to_pos.find(cumotion_joints[i]);
      if (it == name_to_pos.end()) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Goal constraints are missing cuMotion C-space joint '%s'",
          cumotion_joints[i].c_str());
        return false;
      }
      goal_cspace_position(static_cast<Eigen::Index>(i)) = it->second;
    }

    has_joint_goal = true;

    return true;
  }

  // Check for pose constraints.
  if (!constraint.position_constraints.empty() && !constraint.orientation_constraints.empty()) {
    RCLCPP_INFO(this->get_logger(), "Using task space goal");

    const auto & pos_constraint = constraint.position_constraints[0];
    const auto & orient_constraint = constraint.orientation_constraints[0];

    if (pos_constraint.link_name != orient_constraint.link_name) {
      RCLCPP_ERROR(
        this->get_logger(), "Position and orientation constraints have different link names");
      return false;
    }

    if (pos_constraint.link_name != robot_manager_->GetToolFrame()) {
      RCLCPP_ERROR(
        this->get_logger(), "Target link '%s' does not match end effector '%s'",
        pos_constraint.link_name.c_str(), robot_manager_->GetToolFrame().c_str());
      return false;
    }

    if (pos_constraint.constraint_region.primitive_poses.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No primitive poses in position constraint");
      return false;
    }

    geometry_msgs::msg::Pose ros_pose;
    ros_pose.position = pos_constraint.constraint_region.primitive_poses[0].position;
    ros_pose.orientation = orient_constraint.orientation;

    goal_task_space_pose = ToCuMotionPose(ros_pose);
    has_pose_goal = true;

    return true;
  }

  RCLCPP_ERROR(this->get_logger(), "Goal constraints not supported");
  return false;
}

rclcpp_action::GoalResponse CumotionPlanner::HandleIkGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const IKAction::Goal> goal)
{
  static_cast<void>(uuid);
  static_cast<void>(goal);

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (robot_description_update_in_progress_) {
    RCLCPP_ERROR(
      this->get_logger(), "Robot description update in progress, rejecting IK goal");
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (planner_busy_) {
    RCLCPP_ERROR(this->get_logger(), "Planner is busy, rejecting IK goal");
    return rclcpp_action::GoalResponse::REJECT;
  }

  RCLCPP_INFO(this->get_logger(), "Received IK request");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse CumotionPlanner::HandleIkCancel(
  const std::shared_ptr<GoalHandleIk> goal_handle)
{
  static_cast<void>(goal_handle);
  RCLCPP_INFO(this->get_logger(), "Received request to cancel IK goal");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void CumotionPlanner::HandleIkAccepted(
  const std::shared_ptr<GoalHandleIk> goal_handle)
{
  std::thread{std::bind(&CumotionPlanner::ExecuteIkGoal, this, goal_handle)}.detach();
}

void CumotionPlanner::ExecuteIkGoal(
  const std::shared_ptr<GoalHandleIk> goal_handle)
{
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    planner_busy_ = true;
  }

  RCLCPP_INFO(this->get_logger(), "Executing IK goal...");

  const auto & goal = goal_handle->get_goal();
  auto result = std::make_shared<IKAction::Result>();

  // Get object pose from TF if needed.
  geometry_msgs::msg::Pose world_pose_object;
  bool has_object_pose = false;
  if (!goal->object_frame.empty() && !goal->world_frame.empty()) {
    has_object_pose = GetObjectPose(goal->world_frame, goal->object_frame, world_pose_object);
    if (!has_object_pose) {
      result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::FRAME_TRANSFORM_FAILURE;
      RCLCPP_ERROR(
        this->get_logger(), "Failed to get TF (%s -> %s), aborting IK",
        goal->object_frame.c_str(), goal->world_frame.c_str());
      goal_handle->abort(result);
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        planner_busy_ = false;
      }
      return;
    }
  }

  // Calculate AABBs to clear if requested.
  std::unique_ptr<EsdfClearingObjects> clearing_objects = nullptr;
  if (goal->enable_aabb_clearing && has_object_pose) {
    std::vector<double> padding_double(
      goal->object_esdf_clearing_padding.begin(), goal->object_esdf_clearing_padding.end());

    clearing_objects = world_manager_->CalculateAabbsToClear(
      world_pose_object, goal->mesh_resource, padding_double, goal->object_shape,
      goal->object_scale);

    if (clearing_objects && clearing_objects->HasObjects()) {
      RCLCPP_INFO(
        this->get_logger(), "ESDF clearing enabled with %zu AABBs",
        clearing_objects->aabbs_min.size());
    }
  }

  bool world_updated = world_manager_->UpdateWorldObjects(world_objects_);
  if (world_updated && read_esdf_world_) {
    world_updated = UpdateEsdfFromNvblox(clearing_objects.get(), true);
  }

  if (world_updated) {
    PublishWorldVoxels();
  }
  if (!world_updated) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::COLLISION_CHECKING_UNAVAILABLE;
    RCLCPP_ERROR(this->get_logger(), "World update failed for IK");
    goal_handle->abort(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  // Extract seed configuration.
  Eigen::VectorXd seed_cspace_position;
  if (!goal->seed_state.position.empty()) {
    seed_cspace_position = Eigen::Map<const Eigen::VectorXd>(
      goal->seed_state.position.data(),
      goal->seed_state.position.size());
  } else {
    if (!robot_manager_->GetCurrentJointState(seed_cspace_position)) {
      RCLCPP_ERROR(
        this->get_logger(), "No seed state provided and no current joint state available");
      result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID;
      goal_handle->abort(result);
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        planner_busy_ = false;
      }
      return;
    }
  }

  // Convert goal pose to cuMotion format.
  cumotion_lib::Pose3 target_task_space_pose = ToCuMotionPose(goal->goal_pose);

  if (!ik_solver_) {
    RCLCPP_ERROR(this->get_logger(), "IK solver is not initialized");
    result->planning_time = 0.0;
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
    goal_handle->abort(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  std::vector<Eigen::VectorXd> solutions;
  const auto start_time = std::chrono::high_resolution_clock::now();
  auto translation_constraint =
    cumotion_lib::CollisionFreeIkSolver::TranslationConstraint::Target(
    target_task_space_pose.translation);
  auto orientation_constraint =
    cumotion_lib::CollisionFreeIkSolver::OrientationConstraint::Target(
    target_task_space_pose.rotation);
  cumotion_lib::CollisionFreeIkSolver::TaskSpaceTarget task_target(
    translation_constraint, orientation_constraint);

  std::vector<Eigen::VectorXd> seeds;
  if (seed_cspace_position.size() > 0) {
    seeds.emplace_back(seed_cspace_position);
  }

  bool ik_success = ik_solver_->Solve(task_target, seeds, solutions);
  const auto end_time = std::chrono::high_resolution_clock::now();
  const double planning_time =
    std::chrono::duration<double>(end_time - start_time).count();

  if (!ik_success) {
    result->planning_time = planning_time;
  } else {
    auto joint_names = robot_manager_->GetJointNames();
    result->joint_states.reserve(solutions.size());
    result->success.reserve(solutions.size());
    for (const auto & q : solutions) {
      sensor_msgs::msg::JointState joint_state;
      joint_state.header.stamp = this->get_clock()->now();
      joint_state.name = joint_names;
      joint_state.position.resize(q.size());
      for (int j = 0; j < q.size(); ++j) {
        joint_state.position[j] = q(j);
      }
      joint_state.velocity.resize(q.size(), 0.0);
      joint_state.effort.resize(q.size(), 0.0);
      result->joint_states.push_back(std::move(joint_state));
      result->success.push_back(true);
    }
    result->planning_time = planning_time;
  }

  if (ik_success) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    RCLCPP_INFO(
      this->get_logger(), "IK succeeded with %zu solutions", result->joint_states.size());
  } else {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
    RCLCPP_ERROR(this->get_logger(), "IK failed");
  }

  if (ik_success) {
    goal_handle->succeed(result);
  } else {
    goal_handle->abort(result);
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    planner_busy_ = false;
  }
}

void CumotionPlanner::CallPublishStaticPlanningSceneService()
{
  if (!static_planning_scene_client_) {
    RCLCPP_WARN(
      this->get_logger(),
      "Static planning scene client is not initialized, skipping service call");
    return;
  }

  try {
    auto request =
      std::make_shared<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene::Request>();

    auto future = static_planning_scene_client_->async_send_request(
      request, std::bind(
        &CumotionPlanner::StaticSceneServiceCallback, this,
        std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Calling static planning scene service");
    (void)future;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      this->get_logger(), "Failed to call static planning scene service: %s", e.what());
  }
}

void CumotionPlanner::StaticSceneServiceCallback(
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::PublishStaticPlanningScene>::SharedFuture
  future)
{
  try {
    auto response = future.get();
    if (!response->success) {
      RCLCPP_WARN(
        this->get_logger(), "Static planning scene service failed: %s",
        response->message.c_str());
    } else {
      world_objects_ = response->planning_scene.world.collision_objects;
      RCLCPP_INFO(
        this->get_logger(), "Updated world objects with %zu collision objects",
        world_objects_.size());

      if (!world_objects_.empty()) {
        bool world_updated = world_manager_->UpdateWorldObjects(world_objects_);
        if (world_updated && read_esdf_world_) {
          world_updated = UpdateEsdfFromNvblox(nullptr, true);
        }

        if (world_updated) {
          PublishWorldVoxels();
        }

        if (world_updated) {
          RCLCPP_INFO(
            this->get_logger(),
            "Static planning scene obstacles added to cuMotion world");
        } else {
          RCLCPP_WARN(
            this->get_logger(),
            "Failed to add static planning scene obstacles to cuMotion world");
        }
      }

      RCLCPP_INFO(this->get_logger(), "Static planning scene loaded successfully.");
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      this->get_logger(), "Failed to get static planning scene service result: %s",
      e.what());
  }
}

rclcpp_action::GoalResponse CumotionPlanner::HandleMotionPlanGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const MotionPlan::Goal> goal)
{
  static_cast<void>(uuid);
  static_cast<void>(goal);

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (robot_description_update_in_progress_) {
    RCLCPP_ERROR(
      this->get_logger(), "Robot description update in progress, rejecting MotionPlan goal");
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (planner_busy_) {
    RCLCPP_ERROR(this->get_logger(), "Planner is busy, rejecting MotionPlan goal");
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse CumotionPlanner::HandleMotionPlanCancel(
  const std::shared_ptr<GoalHandleMotionPlan> goal_handle)
{
  static_cast<void>(goal_handle);
  RCLCPP_INFO(this->get_logger(), "Received request to cancel MotionPlan goal");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void CumotionPlanner::HandleMotionPlanAccepted(
  const std::shared_ptr<GoalHandleMotionPlan> goal_handle)
{
  std::thread{
    std::bind(
      &CumotionPlanner::ExecuteMotionPlan,
      this,
      goal_handle)}.detach();
}

void CumotionPlanner::ExecuteMotionPlan(
  const std::shared_ptr<GoalHandleMotionPlan> goal_handle)
{
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    planner_busy_ = true;
  }

  const auto & request = goal_handle->get_goal();
  auto result = std::make_shared<MotionPlan::Result>();
  result->success = false;

  double time_dilation = request->time_dilation_factor;
  if (time_dilation <= 0.0) {
    time_dilation = 0.1;
    RCLCPP_WARN(this->get_logger(), "Cannot set time_dilation_factor = 0.0, using 0.1");
  }
  if (time_dilation > 1.0) {
    RCLCPP_WARN(
      this->get_logger(),
      "time_dilation_factor > 1.0 is not supported (got %.6f), clamping to 1.0",
      time_dilation);
    time_dilation = 1.0;
  }
  RCLCPP_INFO(this->get_logger(), "Planning with time_dilation_factor: %f", time_dilation);

  bool world_updated = true;

  if (request->use_planning_scene) {
    RCLCPP_INFO(this->get_logger(), "Updating planning scene");
    // Merge static world objects (loaded from the static planning scene server) with any
    // collision objects provided in the current MotionPlan request.
    std::vector<moveit_msgs::msg::CollisionObject> all_objects = world_objects_;
    const auto & request_world_objects = request->world.collision_objects;
    all_objects.insert(
      all_objects.end(), request_world_objects.begin(),
      request_world_objects.end());

    world_updated = world_manager_->UpdateWorldObjects(all_objects);
  }

  if (world_updated && read_esdf_world_) {
    std::unique_ptr<EsdfClearingObjects> clearing_objects = nullptr;
    // NOTE: For plan_grasp we intentionally do NOT apply ESDF AABB clearing here. Grasp planning
    // performs a 2-phase ESDF update: (1) full scene for start->pregrasp goalset planning, then
    // (2) optional cleared ESDF for constrained pregrasp->grasp and grasp->postgrasp segments.
    if (request->enable_aabb_clearing && request->clear_esdf && !request->plan_grasp) {
      if (request->plan_pose && !request->goal_pose.poses.empty()) {
        std::vector<double> padding_double(
          request->object_esdf_clearing_padding.begin(),
          request->object_esdf_clearing_padding.end());

        clearing_objects = world_manager_->CalculateAabbsToClear(
          request->goal_pose.poses[0],
          request->mesh_resource,
          padding_double,
          request->object_shape,
          request->object_scale);
      }
    }

    // Decide whether to request an ESDF update for this planning request.
    // We always allow the first request to initialize the ESDF grid even if update_esdf is false.
    bool esdf_enabled = world_manager_->IsEsdfEnabled();
    Eigen::Vector3i grid_shape = world_manager_->GetGridShape();
    bool has_esdf_grid = (grid_shape != Eigen::Vector3i::Zero());
    bool update_esdf_for_request = esdf_enabled && (request->update_esdf || !has_esdf_grid);

    if (update_esdf_for_request) {
      world_updated = UpdateEsdfFromNvblox(clearing_objects.get(), /*update_esdf=*/ true);
      if (world_updated) {
        PublishWorldVoxels();
      }
    }
  }

  if (!world_updated) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::COLLISION_CHECKING_UNAVAILABLE;
    RCLCPP_ERROR(this->get_logger(), "World update failed");
    goal_handle->abort(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  Eigen::VectorXd start_state;
  if (!GetMotionPlanStartState(request, start_state)) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_INVALID;
    goal_handle->abort(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  if (request->plan_grasp) {
    RCLCPP_INFO(this->get_logger(), "Planning to Grasp Object");
    ExecuteGraspPlanning(request, start_state, time_dilation, result);
  } else if (request->plan_cspace) {
    RCLCPP_INFO(this->get_logger(), "Planning CSpace target");
    ExecuteCSpacePlanning(request, start_state, time_dilation, result);
  } else if (request->plan_pose) {
    RCLCPP_INFO(this->get_logger(), "Planning Pose target");
    ExecutePosePlanning(request, start_state, time_dilation, result);
  } else {
    RCLCPP_ERROR(this->get_logger(), "No valid planning mode specified");
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    goal_handle->abort(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      planner_busy_ = false;
    }
    return;
  }

  if (result->success && display_trajectory_pub_ && !result->planned_trajectory.empty()) {
    moveit_msgs::msg::DisplayTrajectory display_msg;
    display_msg.model_id = "";
    display_msg.trajectory_start.is_diff = false;
    display_msg.trajectory_start.joint_state.header.stamp = this->get_clock()->now();
    display_msg.trajectory_start.joint_state.name = robot_manager_->GetJointNames();
    display_msg.trajectory_start.joint_state.position.resize(
      static_cast<size_t>(start_state.size()));
    display_msg.trajectory_start.joint_state.velocity.resize(
      static_cast<size_t>(start_state.size()), 0.0);
    display_msg.trajectory_start.joint_state.effort.resize(
      static_cast<size_t>(start_state.size()), 0.0);
    for (size_t i = 0; i < static_cast<size_t>(start_state.size()); ++i) {
      display_msg.trajectory_start.joint_state.position[i] = start_state(static_cast<int>(i));
    }
    display_msg.trajectory = result->planned_trajectory;
    display_trajectory_pub_->publish(display_msg);
  }

  if (result->success) {
    goal_handle->succeed(result);
  } else {
    goal_handle->abort(result);
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    planner_busy_ = false;
  }

  RCLCPP_INFO(
    this->get_logger(), "MotionPlan completed with success: %s",
    result->success ? "true" : "false");
}

bool CumotionPlanner::GetMotionPlanStartState(
  const std::shared_ptr<const MotionPlan::Goal> & request,
  Eigen::VectorXd & start_state)
{
  if (request->use_current_state) {
    if (!robot_manager_->GetCurrentJointState(start_state)) {
      RCLCPP_ERROR(this->get_logger(), "No valid joint state available from robot manager");
      return false;
    }
  } else if (!request->start_state.position.empty()) {
    start_state.resize(request->start_state.position.size());
    for (size_t i = 0; i < request->start_state.position.size(); ++i) {
      start_state(i) = request->start_state.position[i];
    }
  } else {
    RCLCPP_ERROR(this->get_logger(), "No valid start state provided");
    return false;
  }
  return true;
}

bool CumotionPlanner::ExtractGoalPoses(
  const std::shared_ptr<const MotionPlan::Goal> & request,
  std::vector<cumotion_lib::Pose3> & goal_poses)
{
  for (const auto & pose : request->goal_pose.poses) {
    goal_poses.push_back(ToCuMotionPose(pose));
  }

  if (goal_poses.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No goal poses found");
    return false;
  }

  return true;
}

bool CumotionPlanner::GetObjectPose(
  const std::string & world_frame,
  const std::string & object_frame,
  geometry_msgs::msg::Pose & object_pose)
{
  if (!tf_buffer_) {
    RCLCPP_ERROR(this->get_logger(), "TF buffer not initialized");
    return false;
  }

  try {
    auto transform = tf_buffer_->lookupTransform(
      world_frame, object_frame, tf2::TimePointZero);

    object_pose.position.x = transform.transform.translation.x;
    object_pose.position.y = transform.transform.translation.y;
    object_pose.position.z = transform.transform.translation.z;
    object_pose.orientation = transform.transform.rotation;

    RCLCPP_INFO(
      this->get_logger(), "Got transform from %s to %s",
      world_frame.c_str(), object_frame.c_str());
    return true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      this->get_logger(), "Could not transform %s to %s: %s",
      world_frame.c_str(), object_frame.c_str(), ex.what());
    return false;
  }
}

cumotion_lib::Pose3 CumotionPlanner::ComposeWithOffset(
  const cumotion_lib::Pose3 & base_pose,
  const cumotion_lib::Pose3 & offset,
  bool offset_in_base_frame)
{
  return ComposePoseWithOffset(base_pose, offset, offset_in_base_frame);
}

void CumotionPlanner::ExecuteCSpacePlanning(
  const std::shared_ptr<const MotionPlan::Goal> & request,
  const Eigen::VectorXd & start_state,
  double time_dilation,
  std::shared_ptr<MotionPlan::Result> & result)
{
  if (request->goal_state.position.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Goal state is empty");
    result->success = false;
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return;
  }

  Eigen::VectorXd goal_state(request->goal_state.position.size());
  for (size_t i = 0; i < request->goal_state.position.size(); ++i) {
    goal_state(i) = request->goal_state.position[i];
  }

  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;
  auto start_time = std::chrono::high_resolution_clock::now();

  cumotion_lib::TrajectoryOptimizer::CSpaceTarget cspace_target(goal_state);
  bool success = trajectory_optimizer_->PlanToCSpaceTarget(
    start_state, cspace_target, time_dilation,
    trajectory, velocities, accelerations, dt);

  if (success) {
    result->success = true;
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    auto end_time = std::chrono::high_resolution_clock::now();
    double planning_time =
      std::chrono::duration<double>(end_time - start_time).count();

    auto ros_traj = ToROSTrajectory(
      trajectory, velocities, accelerations,
      robot_manager_->GetJointNames(), dt, robot_manager_->GetBaseFrame());
    result->planned_trajectory.push_back(ros_traj);
    result->planning_time = planning_time;
  } else {
    result->success = false;
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
  }
}

void CumotionPlanner::ExecutePosePlanning(
  const std::shared_ptr<const MotionPlan::Goal> & request,
  const Eigen::VectorXd & start_state,
  double time_dilation,
  std::shared_ptr<MotionPlan::Result> & result)
{
  std::vector<cumotion_lib::Pose3> goal_poses;
  if (!ExtractGoalPoses(request, goal_poses)) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return;
  }

  if (request->hold_partial_pose) {
    RCLCPP_WARN(
      this->get_logger(),
      "Partial pose hold is not available.");
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    result->success = false;
    return;
  }

  auto plan_start = std::chrono::high_resolution_clock::now();

  std::vector<Eigen::Vector3d> translations;
  std::vector<cumotion_lib::Rotation3> orientations;
  for (const auto & pose : goal_poses) {
    translations.push_back(pose.translation);
    orientations.push_back(pose.rotation);
  }

  auto translation_goalset =
    cumotion_lib::TrajectoryOptimizer::TranslationConstraintGoalset::Target(
    translations);
  auto orientation_goalset =
    cumotion_lib::TrajectoryOptimizer::OrientationConstraintGoalset::TerminalTarget(
    orientations);

  cumotion_lib::TrajectoryOptimizer::TaskSpaceTargetGoalset task_goalset(
    translation_goalset, orientation_goalset);

  std::vector<Eigen::VectorXd> trajectory;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt = 0.0;
  int goal_index = -1;

  bool success = trajectory_optimizer_->PlanToTaskSpaceGoalset(
    start_state, task_goalset, time_dilation,
    trajectory, velocities, accelerations, dt, goal_index);

  if (success) {
    result->success = true;
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    auto ros_traj = ToROSTrajectory(
      trajectory, velocities, accelerations,
      robot_manager_->GetJointNames(), dt, robot_manager_->GetBaseFrame());
    result->planned_trajectory.push_back(ros_traj);
    result->goal_index = goal_index;

    auto plan_end = std::chrono::high_resolution_clock::now();
    result->planning_time = std::chrono::duration<double>(plan_end - plan_start).count();

    RCLCPP_INFO(
      this->get_logger(), "Goalset planning successful, selected goal %d of %zu",
      result->goal_index, goal_poses.size());
  } else {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    RCLCPP_ERROR(this->get_logger(), "Goalset trajectory optimization failed");
  }
}

void CumotionPlanner::CreatePathConstraintsFromParams(
  float translation_path_deviation_limit,
  float translation_terminal_deviation_limit,
  bool enable_orientation_path_axis_constraint,
  float orientation_path_axis_deviation_limit,
  float orientation_terminal_deviation_limit,
  const cumotion_lib::Pose3 & target_pose,
  const std::string & constraint_name,
  cumotion_lib::TrajectoryOptimizer::TranslationConstraint & translation_constraint,
  cumotion_lib::TrajectoryOptimizer::OrientationConstraint & orientation_constraint)
{
  const bool enable_translation_path_constraint = translation_path_deviation_limit >= 0.0F;

  if (enable_translation_path_constraint) {
    double path_dev = static_cast<double>(translation_path_deviation_limit);
    double terminal_dev = static_cast<double>(translation_terminal_deviation_limit);

    translation_constraint =
      cumotion_lib::TrajectoryOptimizer::TranslationConstraint::LinearPathConstraint(
      target_pose.translation, &path_dev,
      (translation_terminal_deviation_limit >= 0.0F) ? &terminal_dev : nullptr);

    const std::string terminal_dev_str =
      (translation_terminal_deviation_limit >= 0.0F) ?
      std::to_string(terminal_dev) : "default";
    RCLCPP_DEBUG(
      this->get_logger(),
      "%s Translation: linear path (path_dev=%.4fm, terminal_dev=%s)",
      constraint_name.c_str(),
      path_dev,
      terminal_dev_str.c_str());
  } else {
    translation_constraint = cumotion_lib::TrajectoryOptimizer::TranslationConstraint::Target(
      target_pose.translation);
    RCLCPP_DEBUG(
      this->get_logger(), "%s Translation: terminal target only",
      constraint_name.c_str());
  }

  if (enable_orientation_path_axis_constraint) {
    double terminal_dev = static_cast<double>(orientation_terminal_deviation_limit);
    double path_axis_dev = static_cast<double>(orientation_path_axis_deviation_limit);

    orientation_constraint =
      cumotion_lib::TrajectoryOptimizer::OrientationConstraint::Constant(
      (orientation_path_axis_deviation_limit >= 0.0F) ? &path_axis_dev : nullptr,
      (orientation_terminal_deviation_limit >= 0.0F) ? &terminal_dev : nullptr);

    const std::string path_dev_str =
      (orientation_path_axis_deviation_limit >= 0.0F) ?
      std::to_string(path_axis_dev) : "default";
    const std::string terminal_dev_str =
      (orientation_terminal_deviation_limit >= 0.0F) ?
      std::to_string(terminal_dev) : "default";
    RCLCPP_DEBUG(
      this->get_logger(),
      "%s Orientation: constant (path_dev=%s, terminal_dev=%s)",
      constraint_name.c_str(),
      path_dev_str.c_str(),
      terminal_dev_str.c_str());
  } else {
    double terminal_dev = static_cast<double>(orientation_terminal_deviation_limit);

    orientation_constraint =
      cumotion_lib::TrajectoryOptimizer::OrientationConstraint::TerminalTarget(
      target_pose.rotation,
      (orientation_terminal_deviation_limit >= 0.0F) ? &terminal_dev : nullptr);

    const std::string terminal_dev_str =
      (orientation_terminal_deviation_limit >= 0.0F) ?
      std::to_string(terminal_dev) : "default";
    RCLCPP_DEBUG(
      this->get_logger(),
      "%s Orientation: terminal target only (terminal_dev=%s)",
      constraint_name.c_str(),
      terminal_dev_str.c_str());
  }
}

void CumotionPlanner::ExecuteGraspPlanning(
  const std::shared_ptr<const MotionPlan::Goal> & request,
  const Eigen::VectorXd & start_state,
  double time_dilation,
  std::shared_ptr<MotionPlan::Result> & result)
{
  std::vector<cumotion_lib::Pose3> goal_poses;
  if (!ExtractGoalPoses(request, goal_poses)) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return;
  }

  cumotion_lib::Pose3 grasp_offset = ToCuMotionPose(request->grasp_offset_pose);
  cumotion_lib::Pose3 retract_offset = ToCuMotionPose(request->retract_offset_pose);
  bool grasp_in_goal_frame = request->grasp_approach_constraint_in_goal_frame;
  bool retract_in_goal_frame = request->retract_constraint_in_goal_frame;

  auto plan_start = std::chrono::high_resolution_clock::now();

  // -------------------------------------------------------------
  // 1) Request ESDF update from nvblox without any AABB clearing.
  // -------------------------------------------------------------
  std::vector<moveit_msgs::msg::CollisionObject> all_objects;
  bool world_updated = true;

  if (request->use_planning_scene) {
    all_objects = world_objects_;
    const auto & request_world_objects = request->world.collision_objects;
    all_objects.insert(
      all_objects.end(), request_world_objects.begin(),
      request_world_objects.end());

    world_updated = world_manager_->UpdateWorldObjects(all_objects);
  }

  if (world_updated && read_esdf_world_) {
    bool update_esdf_for_request = world_manager_->IsEsdfEnabled();
    if (update_esdf_for_request) {
      world_updated = UpdateEsdfFromNvblox(nullptr, true);
    }
    if (world_updated) {
      PublishWorldVoxels();
    }
  }

  if (!world_updated) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::COLLISION_CHECKING_UNAVAILABLE;
    result->message = "World update failed (full-scene)";
    RCLCPP_ERROR(this->get_logger(), "World update failed (full-scene)");
    return;
  }

  // --------------------------------------------
  // 2) Compute pre-grasp poses from grasp poses.
  // --------------------------------------------
  std::vector<cumotion_lib::Pose3> pregrasp_poses;
  pregrasp_poses.reserve(goal_poses.size());
  for (const auto & grasp_pose : goal_poses) {
    pregrasp_poses.push_back(ComposeWithOffset(grasp_pose, grasp_offset, grasp_in_goal_frame));
  }

  // ----------------------------------------------------------------
  // 3) Plan start->goalset(pregrasp poses) with the full-scene ESDF.
  // ----------------------------------------------------------------
  std::vector<Eigen::Vector3d> pregrasp_translations;
  std::vector<cumotion_lib::Rotation3> pregrasp_orientations;
  pregrasp_translations.reserve(pregrasp_poses.size());
  pregrasp_orientations.reserve(pregrasp_poses.size());
  for (const auto & pose : pregrasp_poses) {
    pregrasp_translations.push_back(pose.translation);
    pregrasp_orientations.push_back(pose.rotation);
  }

  auto translation_goalset =
    cumotion_lib::TrajectoryOptimizer::TranslationConstraintGoalset::Target(pregrasp_translations);
  auto orientation_goalset =
    cumotion_lib::TrajectoryOptimizer::OrientationConstraintGoalset::TerminalTarget(
    pregrasp_orientations);
  cumotion_lib::TrajectoryOptimizer::TaskSpaceTargetGoalset pregrasp_goalset(
    translation_goalset, orientation_goalset);

  std::vector<Eigen::VectorXd> approach_trajectory;
  std::vector<Eigen::VectorXd> approach_velocities;
  std::vector<Eigen::VectorXd> approach_accelerations;
  double approach_dt = 0.0;
  int best_grasp_index = -1;

  bool goalset_success = trajectory_optimizer_->PlanToTaskSpaceGoalset(
    start_state, pregrasp_goalset, time_dilation,
    approach_trajectory, approach_velocities, approach_accelerations, approach_dt,
    best_grasp_index);

  if (!goalset_success || approach_trajectory.empty()) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    result->message = "No pre-grasp in goal set was reachable";
    RCLCPP_ERROR(this->get_logger(), "Pre-grasp goalset planning failed");
    return;
  }

  if (best_grasp_index < 0 || best_grasp_index >= static_cast<int>(goal_poses.size())) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    result->message = "Invalid selected pre-grasp index";
    RCLCPP_ERROR(
      this->get_logger(), "Invalid selected pre-grasp index: %d (num grasp poses: %zu)",
      best_grasp_index, goal_poses.size());
    return;
  }

  const auto & selected_grasp = goal_poses[best_grasp_index];
  std::vector<moveit_msgs::msg::RobotTrajectory> planned_trajectories;

  auto approach_ros_traj = ToROSTrajectory(
    approach_trajectory, approach_velocities, approach_accelerations,
    robot_manager_->GetJointNames(), approach_dt, robot_manager_->GetBaseFrame());
  planned_trajectories.push_back(approach_ros_traj);

  if (!request->plan_approach_to_grasp) {
    result->success = true;
    result->planned_trajectory = planned_trajectories;
    result->goal_index = best_grasp_index;
    auto plan_end = std::chrono::high_resolution_clock::now();
    result->planning_time =
      std::chrono::duration<double>(plan_end - plan_start).count();
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return;
  }

  Eigen::VectorXd approach_final_state = approach_trajectory.back();

  // ---------------------------------------------------------------
  // 4) Request ESDF update from nvblox with optional AABB clearing.
  // ---------------------------------------------------------------
  world_updated = true;

  if (request->use_planning_scene) {
    world_updated = world_manager_->UpdateWorldObjects(all_objects);
  }

  if (world_updated && read_esdf_world_) {
    std::unique_ptr<EsdfClearingObjects> clearing_objects = nullptr;
    if (request->enable_aabb_clearing && request->clear_esdf) {
      geometry_msgs::msg::Pose world_pose_object;
      if (!GetObjectPose(request->world_frame, request->object_frame, world_pose_object)) {
        result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::FRAME_TRANSFORM_FAILURE;
        result->message = "Failed to get object pose from TF";
        RCLCPP_ERROR(
          this->get_logger(), "Failed to get TF (%s -> %s) for grasp planning",
          request->object_frame.c_str(), request->world_frame.c_str());
        return;
      }

      std::vector<double> padding_double(
        request->object_esdf_clearing_padding.begin(),
        request->object_esdf_clearing_padding.end());

      clearing_objects = world_manager_->CalculateAabbsToClear(
        world_pose_object,
        request->mesh_resource,
        padding_double,
        request->object_shape,
        request->object_scale);
    }

    bool update_esdf_for_request = world_manager_->IsEsdfEnabled();
    if (update_esdf_for_request) {
      world_updated = UpdateEsdfFromNvblox(clearing_objects.get(), true);
    }
    if (world_updated) {
      PublishWorldVoxels();
    }
  }

  if (!world_updated) {
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::COLLISION_CHECKING_UNAVAILABLE;
    result->message = "World update failed (cleared ESDF)";
    RCLCPP_ERROR(this->get_logger(), "World update failed (cleared ESDF)");
    return;
  }

  cumotion_lib::TrajectoryOptimizer::TranslationConstraint grasp_translation;
  cumotion_lib::TrajectoryOptimizer::OrientationConstraint grasp_orientation;

  CreatePathConstraintsFromParams(
    request->grasp_translation_path_deviation_limit,
    request->grasp_translation_terminal_deviation_limit,
    request->grasp_enable_orientation_path_axis_constraint,
    request->grasp_orientation_path_axis_deviation_limit,
    request->grasp_orientation_terminal_deviation_limit,
    selected_grasp,
    "Grasp",
    grasp_translation,
    grasp_orientation);

  cumotion_lib::TrajectoryOptimizer::TaskSpaceTarget grasp_task_target(
    grasp_translation, grasp_orientation);

  std::vector<Eigen::VectorXd> grasp_trajectory;
  std::vector<Eigen::VectorXd> grasp_velocities;
  std::vector<Eigen::VectorXd> grasp_accelerations;
  double grasp_dt = 0.0;

  bool grasp_success = trajectory_optimizer_->PlanToTaskSpaceTarget(
    approach_final_state, grasp_task_target, time_dilation,
    grasp_trajectory, grasp_velocities, grasp_accelerations, grasp_dt);
  if (!grasp_success) {
    result->success = false;
    // Return an empty trajectory to indicate that planning failed.
    result->planned_trajectory = std::vector<moveit_msgs::msg::RobotTrajectory>();
    result->goal_index = -1;
    result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    result->message = "Failed to plan from pre-grasp to grasp";
    auto plan_end = std::chrono::high_resolution_clock::now();
    result->planning_time = std::chrono::duration<double>(plan_end - plan_start).count();
    return;
  }

  if (!grasp_trajectory.empty()) {
    auto grasp_ros_traj = ToROSTrajectory(
      grasp_trajectory, grasp_velocities, grasp_accelerations,
      robot_manager_->GetJointNames(), grasp_dt, robot_manager_->GetBaseFrame());
    planned_trajectories.push_back(grasp_ros_traj);
  }

  if (request->plan_grasp_to_retract && !approach_trajectory.empty()) {
    cumotion_lib::Pose3 retract_pose = ComposeWithOffset(
      selected_grasp, retract_offset, retract_in_goal_frame);

    Eigen::VectorXd grasp_final_state = !grasp_trajectory.empty() ?
      grasp_trajectory.back() : approach_final_state;

    cumotion_lib::TrajectoryOptimizer::TranslationConstraint retract_translation;
    cumotion_lib::TrajectoryOptimizer::OrientationConstraint retract_orientation;

    CreatePathConstraintsFromParams(
      request->retract_translation_path_deviation_limit,
      request->retract_translation_terminal_deviation_limit,
      request->retract_enable_orientation_path_axis_constraint,
      request->retract_orientation_path_axis_deviation_limit,
      request->retract_orientation_terminal_deviation_limit,
      retract_pose,
      "Retract",
      retract_translation,
      retract_orientation);

    cumotion_lib::TrajectoryOptimizer::TaskSpaceTarget retract_task_target(
      retract_translation, retract_orientation);

    std::vector<Eigen::VectorXd> retract_trajectory;
    std::vector<Eigen::VectorXd> retract_velocities;
    std::vector<Eigen::VectorXd> retract_accelerations;
    double retract_dt = 0.0;

    bool retract_success = trajectory_optimizer_->PlanToTaskSpaceTarget(
      grasp_final_state, retract_task_target, time_dilation,
      retract_trajectory, retract_velocities, retract_accelerations, retract_dt);

    if (!retract_success) {
      result->success = false;
      // Return an empty trajectory to indicate that planning failed.
      result->planned_trajectory = std::vector<moveit_msgs::msg::RobotTrajectory>();
      result->goal_index = -1;
      result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
      result->message = "Failed to plan from grasp to retract";
      auto plan_end = std::chrono::high_resolution_clock::now();
      result->planning_time = std::chrono::duration<double>(plan_end - plan_start).count();
      return;
    }

    auto retract_ros_traj = ToROSTrajectory(
      retract_trajectory, retract_velocities, retract_accelerations,
      robot_manager_->GetJointNames(), retract_dt, robot_manager_->GetBaseFrame());
    planned_trajectories.push_back(retract_ros_traj);
  }

  result->success = true;
  result->planned_trajectory = planned_trajectories;
  result->goal_index = best_grasp_index;
  result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  auto plan_end = std::chrono::high_resolution_clock::now();
  result->planning_time = std::chrono::duration<double>(plan_end - plan_start).count();

  RCLCPP_INFO(
    this->get_logger(),
    "Grasp planning successful: selected goal %d, generated %zu trajectories",
    best_grasp_index, planned_trajectories.size());
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with rclcpp.
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::cumotion::CumotionPlanner)
