// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion_object_attachment/object_attachment.hpp"

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#include <assimp/Importer.hpp>
#include <tf2/exceptions.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "cumotion/collision_sphere_generator.h"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

constexpr const char * kAttachedObjectSpheresNamespace = "attached_object_spheres";
constexpr const char * kAttachedObjectLinkName = "attached_object";
constexpr const char * kFileUriPrefix = "file://";
constexpr size_t kFileUriPrefixLength = 7;
constexpr double kDefaultMarkerColorR = 0.0;
constexpr double kDefaultMarkerColorG = 1.0;
constexpr double kDefaultMarkerColorB = 0.0;
constexpr double kDefaultMarkerColorA = 0.5;
constexpr double kDefaultOrientationW = 1.0;
constexpr std::chrono::seconds kServiceDiscoveryTimeout{5};
constexpr std::chrono::seconds kServiceResponseTimeout{10};
constexpr std::chrono::seconds kTfLookupTimeout{1};
constexpr unsigned int kAssimpPostProcessFlags =
  aiProcess_Triangulate |
  aiProcess_JoinIdenticalVertices |
  aiProcess_GenNormals;
constexpr unsigned int kVertexCount = 3;
constexpr double kQuaternionNormTolerance = 1e-3;

ObjectAttachmentNode::ObjectAttachmentNode(const rclcpp::NodeOptions & options)
: Node("object_attachment_node", options),
  cached_mesh_importer_(std::make_unique<Assimp::Importer>()),
  robot_description_cached_(false),
  object_attached_(false),
  max_overshoot_(declare_parameter<double>("max_overshoot", 0.05)),
  clear_esdf_on_attach_(declare_parameter<bool>("clear_esdf_on_attach", true)),
  object_esdf_clearing_padding_(declare_parameter<std::vector<double>>(
      "object_esdf_clearing_padding", std::vector<double>{0.05, 0.05, 0.05})),
  esdf_reference_frame_(declare_parameter<std::string>("esdf_reference_frame", "base_link")),
  esdf_visualize_when_clearing_(declare_parameter<bool>("esdf_visualize_when_clearing", true)),
  get_robot_description_service_(declare_parameter<std::string>(
      "get_robot_description_service", "/cumotion/get_robot_description")),
  set_robot_description_service_(declare_parameter<std::string>(
      "set_robot_description_service", "/cumotion/set_robot_description"))
{
  RCLCPP_INFO(this->get_logger(), "Initializing Object Attachment Node");

  ValidateParameters();

  get_robot_description_client_ =
    this->create_client<GetRobotDescription>(get_robot_description_service_);
  set_robot_description_client_ =
    this->create_client<SetRobotDescription>(set_robot_description_service_);

  action_server_ = rclcpp_action::create_server<AttachObjectAction>(
    this,
    "attach_object",
    std::bind(
      &ObjectAttachmentNode::HandleGoal, this, std::placeholders::_1,
      std::placeholders::_2),
    std::bind(&ObjectAttachmentNode::HandleCancel, this, std::placeholders::_1),
    std::bind(&ObjectAttachmentNode::HandleAccepted, this, std::placeholders::_1));

  sphere_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "object_sphere_markers", 10);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  if (clear_esdf_on_attach_) {
    esdf_client_ = this->create_client<nvblox_msgs::srv::EsdfAndGradients>(
      "nvblox_node/get_esdf_and_gradient");
  }

  RCLCPP_INFO(this->get_logger(), "Object Attachment Node initialized successfully");
}

void ObjectAttachmentNode::ValidateParameters()
{
  if (max_overshoot_ < 0.0) {
    throw std::invalid_argument(
            "max_overshoot (" + std::to_string(max_overshoot_) +
            ") must be non-negative.");
  }

  if (object_esdf_clearing_padding_.size() != 3) {
    throw std::invalid_argument(
            "object_esdf_clearing_padding must have exactly 3 elements [x, y, z]. "
            "Received " + std::to_string(object_esdf_clearing_padding_.size()) + " elements.");
  }

  const Eigen::Vector3d padding(
    object_esdf_clearing_padding_[0],
    object_esdf_clearing_padding_[1],
    object_esdf_clearing_padding_[2]);
  if (padding.minCoeff() < 0.0) {
    throw std::invalid_argument(
            "object_esdf_clearing_padding [" +
            std::to_string(padding.x()) + ", " +
            std::to_string(padding.y()) + ", " +
            std::to_string(padding.z()) +
            "] contains negative values. All elements must be non-negative.");
  }
}

rclcpp_action::GoalResponse ObjectAttachmentNode::HandleGoal(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const AttachObjectAction::Goal> goal)
{
  (void)uuid;

  RCLCPP_INFO(this->get_logger(), "Received %s request", goal->attach_object ? "attach" : "detach");

  // Validate goal
  if (goal->attach_object && object_attached_) {
    RCLCPP_WARN(
      this->get_logger(),
      "Object already attached. Detach first before attaching a new object.");
    return rclcpp_action::GoalResponse::REJECT;
  }

  if (!goal->attach_object && !object_attached_) {
    RCLCPP_WARN(this->get_logger(), "No object to detach.");
    return rclcpp_action::GoalResponse::REJECT;
  }

  if (goal->attach_object) {
    const auto & config = goal->object_config;

    // Validate scale: all dimensions must be positive
    const Eigen::Vector3d scale(config.scale.x, config.scale.y, config.scale.z);
    if (scale.minCoeff() <= 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Invalid object scale: [%.4f, %.4f, %.4f]. "
        "All scale dimensions must be positive (> 0).",
        scale.x(), scale.y(), scale.z());
      return rclcpp_action::GoalResponse::REJECT;
    }

    // Validate orientation: quaternion must be approximately unit-length
    const Eigen::Quaterniond quat(
      config.pose.orientation.w,
      config.pose.orientation.x,
      config.pose.orientation.y,
      config.pose.orientation.z);
    if (std::abs(quat.squaredNorm() - 1.0) > kQuaternionNormTolerance) {
      RCLCPP_WARN(
        this->get_logger(),
        "Invalid orientation quaternion: [w=%.4f, x=%.4f, y=%.4f, z=%.4f] "
        "(squared norm=%.6f). The quaternion must be normalized to represent a valid rotation.",
        quat.w(), quat.x(), quat.y(), quat.z(), quat.squaredNorm());
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (config.type == visualization_msgs::msg::Marker::MESH_RESOURCE &&
      config.mesh_resource.empty())
    {
      RCLCPP_WARN(
        this->get_logger(),
        "mesh_resource must not be empty for MESH_RESOURCE marker type.");
      return rclcpp_action::GoalResponse::REJECT;
    }
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ObjectAttachmentNode::HandleCancel(
  const std::shared_ptr<GoalHandleAttachObject> goal_handle)
{
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ObjectAttachmentNode::HandleAccepted(
  const std::shared_ptr<GoalHandleAttachObject> goal_handle)
{
  // Execute in a separate thread to not block
  std::thread{std::bind(&ObjectAttachmentNode::Execute, this, goal_handle)}.detach();
}

void ObjectAttachmentNode::Execute(const std::shared_ptr<GoalHandleAttachObject> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Executing attachment action");

  const auto goal = goal_handle->get_goal();
  auto feedback = std::make_shared<AttachObjectAction::Feedback>();
  auto result = std::make_shared<AttachObjectAction::Result>();

  bool success = false;

  try {
    if (goal->attach_object) {
      feedback->status = "Attaching object to robot collision geometry";
      goal_handle->publish_feedback(feedback);

      success = AttachObject(goal->object_config);

      if (success) {
        object_attached_ = true;
        result->outcome = "Object successfully attached";
        RCLCPP_INFO(this->get_logger(), "%s", result->outcome.c_str());
      } else {
        result->outcome = "Failed to attach object";
        RCLCPP_ERROR(this->get_logger(), "%s", result->outcome.c_str());
      }
    } else {
      feedback->status = "Detaching object from robot collision geometry";
      goal_handle->publish_feedback(feedback);

      success = DetachObject();

      if (success) {
        object_attached_ = false;
        result->outcome = "Object successfully detached";
        RCLCPP_INFO(this->get_logger(), "%s", result->outcome.c_str());
      } else {
        result->outcome = "Failed to detach object";
        RCLCPP_ERROR(this->get_logger(), "%s", result->outcome.c_str());
      }
    }
  } catch (const std::exception & e) {
    result->outcome = std::string("Exception during execution: ") + e.what();
    RCLCPP_ERROR(this->get_logger(), "%s", result->outcome.c_str());
    goal_handle->abort(result);
    return;
  }

  // Mark goal as succeeded or aborted
  if (success) {
    goal_handle->succeed(result);
  } else {
    goal_handle->abort(result);
  }
}

bool ObjectAttachmentNode::GetRobotDescriptionFromService()
{
  if (robot_description_cached_) {
    RCLCPP_DEBUG(this->get_logger(), "Robot description already cached");
    return true;
  }

  RCLCPP_INFO(this->get_logger(), "Requesting robot description from service");

  if (!get_robot_description_client_->wait_for_service(kServiceDiscoveryTimeout)) {
    RCLCPP_ERROR(this->get_logger(), "GetRobotDescription service not available");
    return false;
  }

  auto request = std::make_shared<GetRobotDescription::Request>();
  auto future = get_robot_description_client_->async_send_request(request);
  if (future.wait_for(kServiceResponseTimeout) != std::future_status::ready) {
    RCLCPP_ERROR(this->get_logger(), "GetRobotDescription service call timed out");
    return false;
  }

  auto response = future.get();
  if (response) {
    urdf_ = response->urdf;
    xrdf_ = response->xrdf;
    robot_description_cached_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "Successfully retrieved robot description (URDF: %zu bytes, XRDF: %zu bytes)",
      urdf_.size(), xrdf_.size());
    return true;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to get robot description from service");
    return false;
  }
}

bool ObjectAttachmentNode::UpdateRobotDescriptionWithSpheres(
  const std::vector<CollisionSphere> & spheres)
{
  const std::string link_name = kAttachedObjectLinkName;

  RCLCPP_INFO(
    this->get_logger(),
    "Updating robot description with %zu collision spheres for link '%s'",
    spheres.size(), link_name.c_str());

  try {
    // Parse XRDF string to YAML
    YAML::Node xrdf_yaml = YAML::Load(xrdf_);

    // Find the geometry name from collision section
    std::string geometry_name;
    if (xrdf_yaml["collision"] && xrdf_yaml["collision"]["geometry"]) {
      geometry_name = xrdf_yaml["collision"]["geometry"].as<std::string>();
      RCLCPP_INFO(
        this->get_logger(), "Found geometry name from collision section: '%s'",
        geometry_name.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "No collision geometry section found in XRDF");
      return false;
    }

    // Navigate to geometry section
    // Expected structure: geometry -> <geometry_name> -> spheres -> <link_name>
    if (!xrdf_yaml["geometry"]) {
      RCLCPP_ERROR(this->get_logger(), "No geometry section found in XRDF");
      return false;
    }

    if (!xrdf_yaml["geometry"][geometry_name]) {
      RCLCPP_ERROR(this->get_logger(), "Geometry '%s' not found in XRDF", geometry_name.c_str());
      return false;
    }

    if (!xrdf_yaml["geometry"][geometry_name]["spheres"]) {
      xrdf_yaml["geometry"][geometry_name]["spheres"] = YAML::Node(YAML::NodeType::Map);
    }

    // Create collision spheres entry for the link
    YAML::Node link_spheres;
    for (const auto & sphere : spheres) {
      YAML::Node sphere_node;
      sphere_node["center"] = std::vector<double>{
        sphere.center.x(), sphere.center.y(), sphere.center.z()};
      sphere_node["radius"] = sphere.radius;
      link_spheres.push_back(sphere_node);
    }

    // Add spheres to the geometry
    xrdf_yaml["geometry"][geometry_name]["spheres"][link_name] = link_spheres;

    // Convert back to string
    YAML::Emitter emitter;
    emitter << xrdf_yaml;
    std::string updated_xrdf = emitter.c_str();

    if (!set_robot_description_client_->wait_for_service(kServiceDiscoveryTimeout)) {
      RCLCPP_ERROR(this->get_logger(), "SetRobotDescription service not available");
      return false;
    }

    // Create request with updated XRDF
    auto request = std::make_shared<SetRobotDescription::Request>();
    request->xrdf = updated_xrdf;
    request->urdf = urdf_;  // URDF remains unchanged

    auto future = set_robot_description_client_->async_send_request(request);
    if (future.wait_for(kServiceResponseTimeout) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "SetRobotDescription service call timed out");
      return false;
    }

    auto response = future.get();
    if (response && response->success) {
      RCLCPP_INFO(this->get_logger(), "Successfully updated robot description");
      return true;
    } else {
      RCLCPP_ERROR(
        this->get_logger(), "Failed to set robot description: %s",
        response ? response->message.c_str() : "No response");
      return false;
    }
  } catch (const YAML::Exception & e) {
    RCLCPP_ERROR(
      this->get_logger(), "YAML parsing error while updating robot description: %s",
      e.what());
    return false;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Error updating robot description: %s", e.what());
    return false;
  }
}


std::vector<Eigen::Vector3d> ObjectAttachmentNode::GetCuboidVertices(
  double size_x, double size_y, double size_z)
{
  return GenerateCuboidVertices(size_x, size_y, size_z);
}

std::vector<Eigen::Vector3i> ObjectAttachmentNode::GetCuboidTriangles()
{
  return GenerateCuboidTriangles();
}

std::vector<Eigen::Vector3d> ObjectAttachmentNode::GenerateCuboidVertices(
  double size_x, double size_y, double size_z)
{
  // Calculate half-extents from full dimensions
  const double hx = size_x / 2.0;
  const double hy = size_y / 2.0;
  const double hz = size_z / 2.0;

  // Define the 8 vertices of the cuboid in local object frame
  // Coordinate system: x = left/right, y = bottom/top, z = back/front
  return {
    Eigen::Vector3d(-hx, -hy, -hz),  // 0: back-bottom-left
    Eigen::Vector3d(hx, -hy, -hz),   // 1: back-bottom-right
    Eigen::Vector3d(hx, hy, -hz),    // 2: back-top-right
    Eigen::Vector3d(-hx, hy, -hz),   // 3: back-top-left
    Eigen::Vector3d(-hx, -hy, hz),   // 4: front-bottom-left
    Eigen::Vector3d(hx, -hy, hz),    // 5: front-bottom-right
    Eigen::Vector3d(hx, hy, hz),     // 6: front-top-right
    Eigen::Vector3d(-hx, hy, hz)     // 7: front-top-left
  };
}

std::vector<Eigen::Vector3i> ObjectAttachmentNode::GenerateCuboidTriangles()
{
  // Define the 12 triangles (2 per face, 6 faces) that form the cuboid mesh.
  // Each Vector3i(a, b, c) contains vertex indices from GenerateCuboidVertices().
  // Vertices wind counter-clockwise from outside → normals point outward (right-hand rule).

  return {
    // Back face (z = -h): vertices 0, 1, 2, 3 form a quad
    Eigen::Vector3i(0, 2, 1),  // Triangle 1: bottom-left, top-right, bottom-right
    Eigen::Vector3i(0, 3, 2),  // Triangle 2: bottom-left, top-left, top-right

    // Front face (z = +h): vertices 4, 5, 6, 7 form a quad
    Eigen::Vector3i(4, 5, 6),  // Triangle 1: bottom-left, bottom-right, top-right
    Eigen::Vector3i(4, 6, 7),  // Triangle 2: bottom-left, top-right, top-left

    // Left face (x = -h): vertices 0, 3, 4, 7 form a quad
    Eigen::Vector3i(0, 4, 7),  // Triangle 1: back-bottom, front-bottom, front-top
    Eigen::Vector3i(0, 7, 3),  // Triangle 2: back-bottom, front-top, back-top

    // Right face (x = +h): vertices 1, 2, 5, 6 form a quad
    Eigen::Vector3i(1, 2, 6),  // Triangle 1: back-bottom, back-top, front-top
    Eigen::Vector3i(1, 6, 5),  // Triangle 2: back-bottom, front-top, front-bottom

    // Bottom face (y = -h): vertices 0, 1, 4, 5 form a quad
    Eigen::Vector3i(0, 1, 5),  // Triangle 1: back-left, back-right, front-right
    Eigen::Vector3i(0, 5, 4),  // Triangle 2: back-left, front-right, front-left

    // Top face (y = +h): vertices 2, 3, 6, 7 form a quad
    Eigen::Vector3i(3, 7, 6),  // Triangle 1: back-left, front-left, front-right
    Eigen::Vector3i(3, 6, 2)   // Triangle 2: back-left, front-right, back-right
  };
}

bool ObjectAttachmentNode::LoadMesh(const std::string & mesh_resource)
{
  // If already loaded, return early
  if (cached_mesh_importer_->GetScene()) {
    RCLCPP_DEBUG(this->get_logger(), "Using cached mesh");
    return true;
  }

  if (mesh_resource.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Mesh resource path is empty");
    return false;
  }

  // Extract file path from resource URI (handle "file://", "package://", etc.)
  std::string mesh_path = mesh_resource;
  if (mesh_path.find(kFileUriPrefix) == 0) {
    mesh_path = mesh_path.substr(kFileUriPrefixLength);  // Remove "file://"
  } else if (mesh_path.find("package://") == 0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "package:// URIs not yet supported. Please use absolute file paths.");
    return false;
  }

  std::filesystem::path file_path(mesh_path);
  if (!std::filesystem::exists(file_path)) {
    RCLCPP_ERROR(this->get_logger(), "Mesh file does not exist: %s", mesh_path.c_str());
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Loading mesh from: %s", mesh_path.c_str());

  // Load mesh into the cached importer
  cached_mesh_importer_->ReadFile(mesh_path, kAssimpPostProcessFlags);

  // Non-owning pointer; scene lifetime is managed by cached_mesh_importer_ (freed in DetachObject)
  const aiScene * const scene = cached_mesh_importer_->GetScene();
  if (!scene || !scene->HasMeshes()) {
    RCLCPP_ERROR(
      this->get_logger(), "Failed to load mesh from %s: %s",
      mesh_path.c_str(), cached_mesh_importer_->GetErrorString());
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Loaded mesh with %d submeshes", scene->mNumMeshes);

  return true;
}

bool ObjectAttachmentNode::ExtractMeshGeometry()
{
  // Non-owning pointer; scene lifetime is managed by cached_mesh_importer_ (freed in DetachObject)
  const aiScene * const scene = cached_mesh_importer_->GetScene();
  if (scene == nullptr) {
    RCLCPP_ERROR(this->get_logger(), "Scene is null");
    return false;
  }

  // If we have cached geometry, return early
  if (!cached_mesh_vertices_.empty()) {
    RCLCPP_DEBUG(this->get_logger(), "Using cached mesh geometry");
    return true;
  }

  // Extract vertices (local/mesh frame) and triangles from all submeshes
  for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
    const aiMesh * mesh = scene->mMeshes[m];
    int vertex_offset = cached_mesh_vertices_.size();

    // Extract vertices
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      const aiVector3D & v = mesh->mVertices[i];
      cached_mesh_vertices_.push_back(Eigen::Vector3d(v.x, v.y, v.z));
    }

    // Extract triangles (indices are frame-independent)
    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
      const aiFace & face = mesh->mFaces[i];
      if (face.mNumIndices == kVertexCount) {
        cached_mesh_triangles_.push_back(
          Eigen::Vector3i(
            vertex_offset + face.mIndices[0],
            vertex_offset + face.mIndices[1],
            vertex_offset + face.mIndices[2]));
      }
    }
  }

  RCLCPP_INFO(
    this->get_logger(), "Extracted %zu vertices and %zu triangles from mesh",
    cached_mesh_vertices_.size(), cached_mesh_triangles_.size());

  if (cached_mesh_vertices_.empty() || cached_mesh_triangles_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Mesh has no valid vertices or triangles");
    return false;
  }

  return true;
}

bool ObjectAttachmentNode::GenerateSpheresForSphere(
  const visualization_msgs::msg::Marker & object_config,
  std::vector<CollisionSphere> & spheres)
{
  // For sphere, create a single collision sphere
  CollisionSphere sphere;
  sphere.center = Eigen::Vector3d(
    object_config.pose.position.x,
    object_config.pose.position.y,
    object_config.pose.position.z);
  // Use the largest scale dimension as radius (assuming uniform scaling for sphere)
  sphere.radius =
    std::max({object_config.scale.x, object_config.scale.y, object_config.scale.z}) / 2.0;
  spheres.push_back(sphere);
  RCLCPP_INFO(
    this->get_logger(), "Created single sphere with radius %.3f at [%.3f, %.3f, %.3f]",
    sphere.radius, sphere.center.x(), sphere.center.y(), sphere.center.z());
  return true;
}

bool ObjectAttachmentNode::GenerateSpheresForCuboid(
  const visualization_msgs::msg::Marker & object_config,
  std::vector<CollisionSphere> & spheres)
{
  // Create cuboid mesh vertices in local frame with specified dimensions
  std::vector<Eigen::Vector3d> vertices = GenerateCuboidVertices(
    object_config.scale.x, object_config.scale.y, object_config.scale.z);

  // Define 12 triangles (2 per face, 6 faces)
  std::vector<Eigen::Vector3i> triangles = GenerateCuboidTriangles();

  RCLCPP_INFO(
    this->get_logger(), "Generated cuboid mesh with %zu vertices and %zu triangles",
    vertices.size(), triangles.size());

  // Generate collision spheres using cuMotion API (centers/radii in local/object frame)
  try {
    auto collision_spheres = ::cumotion::GenerateCollisionSpheres(
      vertices, triangles, max_overshoot_);

    RCLCPP_INFO(
      this->get_logger(), "Generated %zu collision spheres for cuboid object",
      collision_spheres.size());

    // Transform from local object frame to grasp frame
    Eigen::Isometry3d grasp_pose_object = Eigen::Isometry3d::Identity();
    grasp_pose_object.translate(
      Eigen::Vector3d(
        object_config.pose.position.x,
        object_config.pose.position.y,
        object_config.pose.position.z));
    grasp_pose_object.rotate(
      Eigen::Quaterniond(
        object_config.pose.orientation.w,
        object_config.pose.orientation.x,
        object_config.pose.orientation.y,
        object_config.pose.orientation.z));

    for (const auto & sphere : collision_spheres) {
      CollisionSphere sphere_in_grasp_frame;
      sphere_in_grasp_frame.center = grasp_pose_object * sphere.center;
      sphere_in_grasp_frame.radius = sphere.radius;
      spheres.push_back(sphere_in_grasp_frame);
    }

    RCLCPP_INFO(
      this->get_logger(), "Transformed %zu collision spheres to grasp frame",
      spheres.size());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to generate collision spheres: %s", e.what());
    return false;
  }

  return true;
}

bool ObjectAttachmentNode::GenerateSpheresForMesh(
  const visualization_msgs::msg::Marker & object_config,
  std::vector<CollisionSphere> & spheres)
{
  // Load mesh with caching
  if (!LoadMesh(object_config.mesh_resource)) {
    return false;
  }

  // Extract vertices and triangles into cached members
  if (!ExtractMeshGeometry()) {
    return false;
  }

  // Apply scale to vertices so the sphere generation algorithm gets the actual scaled geometry
  std::vector<Eigen::Vector3d> scaled_vertices;
  scaled_vertices.reserve(cached_mesh_vertices_.size());
  for (const auto & vertex : cached_mesh_vertices_) {
    scaled_vertices.push_back(
      Eigen::Vector3d(
        vertex.x() * object_config.scale.x,
        vertex.y() * object_config.scale.y,
        vertex.z() * object_config.scale.z));
  }

  // Generate collision spheres using cuMotion API (centers/radii in local/mesh frame)
  try {
    auto collision_spheres = ::cumotion::GenerateCollisionSpheres(
      scaled_vertices, cached_mesh_triangles_, max_overshoot_);

    RCLCPP_INFO(
      this->get_logger(), "Generated %zu collision spheres for mesh object",
      collision_spheres.size());

    // Transform from local mesh frame to grasp frame
    Eigen::Isometry3d grasp_pose_object = Eigen::Isometry3d::Identity();
    grasp_pose_object.translate(
      Eigen::Vector3d(
        object_config.pose.position.x,
        object_config.pose.position.y,
        object_config.pose.position.z));
    grasp_pose_object.rotate(
      Eigen::Quaterniond(
        object_config.pose.orientation.w,
        object_config.pose.orientation.x,
        object_config.pose.orientation.y,
        object_config.pose.orientation.z));

    for (const auto & sphere : collision_spheres) {
      CollisionSphere sphere_in_grasp_frame;
      sphere_in_grasp_frame.center = grasp_pose_object * sphere.center;
      sphere_in_grasp_frame.radius = sphere.radius;
      spheres.push_back(sphere_in_grasp_frame);
    }

    RCLCPP_INFO(
      this->get_logger(), "Transformed %zu collision spheres to grasp frame",
      spheres.size());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to generate collision spheres: %s", e.what());
    return false;
  }
  return true;
}

bool ObjectAttachmentNode::AttachObject(
  const visualization_msgs::msg::Marker & object_config)
{
  RCLCPP_INFO(this->get_logger(), "Attaching object with type %d", object_config.type);

  if (!GetRobotDescriptionFromService()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to get robot description");
    return false;
  }

  try {
    std::vector<CollisionSphere> spheres;

    // Generate collision spheres based on object type
    bool success = false;
    switch (object_config.type) {
      case visualization_msgs::msg::Marker::SPHERE:
        success = GenerateSpheresForSphere(object_config, spheres);
        break;

      case visualization_msgs::msg::Marker::CUBE:
        success = GenerateSpheresForCuboid(object_config, spheres);
        break;

      case visualization_msgs::msg::Marker::MESH_RESOURCE:
        success = GenerateSpheresForMesh(object_config, spheres);
        break;

      default:
        RCLCPP_ERROR(
          this->get_logger(),
          "Unsupported marker type: %d. Supported types: SPHERE(%d), CUBE(%d), MESH_RESOURCE(%d)",
          object_config.type,
          visualization_msgs::msg::Marker::SPHERE,
          visualization_msgs::msg::Marker::CUBE,
          visualization_msgs::msg::Marker::MESH_RESOURCE);
        return false;
    }

    if (!success) {
      RCLCPP_ERROR(this->get_logger(), "Failed to generate collision spheres");
      return false;
    }

    if (spheres.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No collision spheres generated");
      return false;
    }

    // Publish visualization markers for the generated spheres
    visualization_msgs::msg::MarkerArray marker_array;
    for (size_t i = 0; i < spheres.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = object_config.header.frame_id;
      marker.header.stamp = this->now();
      marker.ns = kAttachedObjectSpheresNamespace;
      marker.id = i;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x = spheres[i].center.x();
      marker.pose.position.y = spheres[i].center.y();
      marker.pose.position.z = spheres[i].center.z();
      marker.pose.orientation.w = kDefaultOrientationW;

      marker.scale.x = spheres[i].radius * 2.0;
      marker.scale.y = spheres[i].radius * 2.0;
      marker.scale.z = spheres[i].radius * 2.0;
      marker.color.r = kDefaultMarkerColorR;
      marker.color.g = kDefaultMarkerColorG;
      marker.color.b = kDefaultMarkerColorB;
      marker.color.a = kDefaultMarkerColorA;

      marker_array.markers.push_back(marker);
    }
    sphere_markers_pub_->publish(marker_array);

    if (!UpdateRobotDescriptionWithSpheres(spheres)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to update robot description with collision spheres");
      return false;
    }

    // Update world representation: Clear voxels occupied by the attached object.
    // Since the object is now part of the robot (not an independent obstacle),
    // we need to remove it from the 3D voxel grid to avoid collision detection issues.
    if (clear_esdf_on_attach_) {
      if (!ClearObjectVoxelsFromWorld(object_config)) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to clear object voxels from ESDF. "
          "Object attachment aborted because clear_esdf_on_attach is enabled.");
        return false;
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Object attached successfully with %zu collision spheres to link 'attached_object'",
      spheres.size());

    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to attach object: %s", e.what());

    return false;
  }
}

bool ObjectAttachmentNode::DetachObject()
{
  RCLCPP_INFO(this->get_logger(), "Detaching object from robot");
  try {
    if (!set_robot_description_client_->wait_for_service(kServiceDiscoveryTimeout)) {
      RCLCPP_ERROR(this->get_logger(), "SetRobotDescription service not available");
      return false;
    }

    // Create request with updated XRDF
    auto request = std::make_shared<SetRobotDescription::Request>();
    request->xrdf = xrdf_;
    request->urdf = urdf_;

    auto future = set_robot_description_client_->async_send_request(request);
    if (future.wait_for(kServiceResponseTimeout) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "SetRobotDescription service call timed out");
      return false;
    }

    auto response = future.get();
    if (response && response->success) {
      // Clear visualization markers
      visualization_msgs::msg::MarkerArray marker_array;
      visualization_msgs::msg::Marker delete_marker;
      delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
      delete_marker.ns = kAttachedObjectSpheresNamespace;
      marker_array.markers.push_back(delete_marker);
      sphere_markers_pub_->publish(marker_array);

      RCLCPP_INFO(this->get_logger(), "Object detached successfully");

      // Clear mesh cache after successful detach operation
      RCLCPP_DEBUG(this->get_logger(), "Clearing mesh cache");
      cached_mesh_importer_->FreeScene();
      cached_mesh_vertices_.clear();
      cached_mesh_triangles_.clear();

      return true;
    } else {
      RCLCPP_ERROR(
        this->get_logger(), "Failed to set robot description: %s",
        response ? response->message.c_str() : "No response");

      return false;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to detach object: %s", e.what());

    return false;
  }
}

ObjectAttachmentNode::EsdfClearingObjects ObjectAttachmentNode::CalculateClearingRegions(
  const geometry_msgs::msg::Pose & world_pose_object,
  const visualization_msgs::msg::Marker & object_config)
{
  RCLCPP_DEBUG(this->get_logger(), "Calculating clearing regions for object...");

  EsdfClearingObjects clearing_objects;
  bool success = false;

  switch (object_config.type) {
    case visualization_msgs::msg::Marker::SPHERE:
      success = CalculateClearingRegionsForSphere(
        world_pose_object, object_config, clearing_objects);
      break;

    case visualization_msgs::msg::Marker::CUBE:
      success = CalculateClearingRegionsForCuboid(
        world_pose_object, object_config, clearing_objects);
      break;

    case visualization_msgs::msg::Marker::MESH_RESOURCE:
      success = CalculateClearingRegionsForMesh(
        world_pose_object, object_config, clearing_objects);
      break;

    default:
      RCLCPP_ERROR(
        this->get_logger(),
        "Unknown object shape type: %d. Supported types: SPHERE (2), CUBE (1), MESH_RESOURCE (10)",
        object_config.type);
      return clearing_objects;
  }

  if (!success) {
    RCLCPP_WARN(
      this->get_logger(),
      "Failed to calculate clearing regions for object type %d", object_config.type);
    return clearing_objects;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Successfully calculated %zu AABB(s) and %zu sphere(s) for clearing",
    clearing_objects.aabbs_min.size(), clearing_objects.spheres_center.size());

  return clearing_objects;
}

bool ObjectAttachmentNode::CalculateClearingRegionsForSphere(
  const geometry_msgs::msg::Pose & world_pose_object,
  const visualization_msgs::msg::Marker & object_config,
  EsdfClearingObjects & clearing_objects)
{
  RCLCPP_DEBUG(this->get_logger(), "Calculating clearing regions for SPHERE object");

  Eigen::Vector3d padding(
    object_esdf_clearing_padding_[0],
    object_esdf_clearing_padding_[1],
    object_esdf_clearing_padding_[2]
  );

  // For sphere, create a single sphere to clear
  float radius = std::max(
    {object_config.scale.x, object_config.scale.y, object_config.scale.z}) / 2.0;
  // Add object_esdf_clearing_padding so the sphere surface is padded outward
  // by padding.maxCoeff() in every direction, matching AABB per-side clearance.
  radius += padding.maxCoeff();

  geometry_msgs::msg::Point center;
  center.x = world_pose_object.position.x;
  center.y = world_pose_object.position.y;
  center.z = world_pose_object.position.z;

  clearing_objects.spheres_center.push_back(center);
  clearing_objects.spheres_radius.push_back(radius);

  RCLCPP_DEBUG(
    this->get_logger(), "Clearing sphere: center=(%.3f, %.3f, %.3f), radius=%.3f",
    center.x, center.y, center.z, radius);

  return true;
}

bool ObjectAttachmentNode::ComputeAABBFromVertices(
  const std::vector<Eigen::Vector3d> & vertices,
  const Eigen::Affine3d & world_transform,
  geometry_msgs::msg::Point & aabb_min,
  geometry_msgs::msg::Vector3 & aabb_size)
{
  if (vertices.empty()) {
    return false;
  }

  // Transform each vertex to world frame and track component-wise min/max
  Eigen::Vector3d min_corner(
    std::numeric_limits<double>::max(),
    std::numeric_limits<double>::max(),
    std::numeric_limits<double>::max());
  Eigen::Vector3d max_corner(
    std::numeric_limits<double>::lowest(),
    std::numeric_limits<double>::lowest(),
    std::numeric_limits<double>::lowest());

  for (const auto & vertex : vertices) {
    Eigen::Vector3d vertex_in_world_frame = world_transform * vertex;
    for (int i = 0; i < 3; ++i) {
      min_corner(i) = std::min(min_corner(i), vertex_in_world_frame(i));
      max_corner(i) = std::max(max_corner(i), vertex_in_world_frame(i));
    }
  }

  // Apply object_esdf_clearing_padding_ per-side: each AABB face is pushed
  // outward by the full padding value so the clearance between the object
  // surface and the clearing boundary equals object_esdf_clearing_padding_.
  const Eigen::Vector3d padding(
    object_esdf_clearing_padding_[0],
    object_esdf_clearing_padding_[1],
    object_esdf_clearing_padding_[2]);
  min_corner -= padding;
  max_corner += padding;

  aabb_min.x = min_corner(0);
  aabb_min.y = min_corner(1);
  aabb_min.z = min_corner(2);

  aabb_size.x = max_corner(0) - min_corner(0);
  aabb_size.y = max_corner(1) - min_corner(1);
  aabb_size.z = max_corner(2) - min_corner(2);

  return true;
}

bool ObjectAttachmentNode::CalculateClearingRegionsForCuboid(
  const geometry_msgs::msg::Pose & world_pose_object,
  const visualization_msgs::msg::Marker & object_config,
  EsdfClearingObjects & clearing_objects)
{
  RCLCPP_DEBUG(this->get_logger(), "Calculating clearing regions for CUBOID object");

  // Generate cuboid vertices in local frame with actual dimensions
  std::vector<Eigen::Vector3d> vertices = GenerateCuboidVertices(
    object_config.scale.x, object_config.scale.y, object_config.scale.z);

  // Transform vertices from local frame to world frame using pose
  Eigen::Isometry3d world_pose_object_iso = Eigen::Isometry3d::Identity();
  world_pose_object_iso.translate(
    Eigen::Vector3d(
      world_pose_object.position.x,
      world_pose_object.position.y,
      world_pose_object.position.z));
  world_pose_object_iso.rotate(
    Eigen::Quaterniond(
      world_pose_object.orientation.w,
      world_pose_object.orientation.x,
      world_pose_object.orientation.y,
      world_pose_object.orientation.z));

  geometry_msgs::msg::Point aabb_min;
  geometry_msgs::msg::Vector3 aabb_size;
  if (!ComputeAABBFromVertices(
      vertices,
      Eigen::Affine3d(world_pose_object_iso.matrix()),
      aabb_min,
      aabb_size))
  {
    return false;
  }

  clearing_objects.aabbs_min.push_back(aabb_min);
  clearing_objects.aabbs_size.push_back(aabb_size);

  RCLCPP_DEBUG(
    this->get_logger(),
    "Clearing AABB for cuboid: min=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)",
    aabb_min.x, aabb_min.y, aabb_min.z,
    aabb_size.x, aabb_size.y, aabb_size.z);

  return true;
}

bool ObjectAttachmentNode::CalculateClearingRegionsForMesh(
  const geometry_msgs::msg::Pose & world_pose_object,
  const visualization_msgs::msg::Marker & object_config,
  EsdfClearingObjects & clearing_objects)
{
  RCLCPP_DEBUG(this->get_logger(), "Calculating clearing regions for MESH_RESOURCE object");

  // Load mesh with caching
  if (!LoadMesh(object_config.mesh_resource)) {
    return false;
  }

  // Extract vertices and triangles into cached members
  if (!ExtractMeshGeometry()) {
    return false;
  }

  // Create transform matrix from pose and scale
  // Build transformation matrix: T * R * S (scale, then rotate, then translate)
  // Note: Methods are called in reverse order because they post-multiply
  Eigen::Affine3d world_transform_object_affine = Eigen::Affine3d::Identity();
  world_transform_object_affine.translate(
    Eigen::Vector3d(
      world_pose_object.position.x,
      world_pose_object.position.y,
      world_pose_object.position.z));
  world_transform_object_affine.rotate(
    Eigen::Quaterniond(
      world_pose_object.orientation.w,
      world_pose_object.orientation.x,
      world_pose_object.orientation.y,
      world_pose_object.orientation.z));
  world_transform_object_affine.scale(
    Eigen::Vector3d(
      object_config.scale.x,
      object_config.scale.y,
      object_config.scale.z));

  geometry_msgs::msg::Point aabb_min;
  geometry_msgs::msg::Vector3 aabb_size;
  if (!ComputeAABBFromVertices(
      cached_mesh_vertices_,
      world_transform_object_affine,
      aabb_min,
      aabb_size))
  {
    return false;
  }

  clearing_objects.aabbs_min.push_back(aabb_min);
  clearing_objects.aabbs_size.push_back(aabb_size);

  RCLCPP_DEBUG(
    this->get_logger(),
    "Clearing AABB for custom mesh: min=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)",
    aabb_min.x, aabb_min.y, aabb_min.z,
    aabb_size.x, aabb_size.y, aabb_size.z);

  return true;
}

bool ObjectAttachmentNode::SendEsdfClearingRequest(
  const EsdfClearingObjects & clearing_objects)
{
  RCLCPP_DEBUG(this->get_logger(), "Sending ESDF clearing request to nvblox...");

  // Verify ESDF service availability before calling.
  RCLCPP_INFO(this->get_logger(), "Checking ESDF service availability...");
  if (!esdf_client_->wait_for_service(kServiceDiscoveryTimeout)) {
    RCLCPP_ERROR(
      this->get_logger(),
      "ESDF service (nvblox_node/get_esdf_and_gradient) not available after %ld seconds. "
      "Cannot clear object voxels.",
      kServiceDiscoveryTimeout.count());
    return false;
  }
  RCLCPP_INFO(this->get_logger(), "ESDF service is available");

  auto request = std::make_shared<nvblox_msgs::srv::EsdfAndGradients::Request>();
  request->visualize_esdf = esdf_visualize_when_clearing_;
  request->update_esdf = true;
  // Must match nvblox's global_frame parameter
  request->frame_id = esdf_reference_frame_;

  if (clearing_objects.aabbs_min.size() != clearing_objects.aabbs_size.size()) {
    RCLCPP_ERROR(
      this->get_logger(), "AABB vector size mismatch: min=%zu, size=%zu",
      clearing_objects.aabbs_min.size(), clearing_objects.aabbs_size.size());
    return false;
  }

  if (clearing_objects.spheres_center.size() != clearing_objects.spheres_radius.size()) {
    RCLCPP_ERROR(
      this->get_logger(), "Sphere vector size mismatch: centers=%zu, radii=%zu",
      clearing_objects.spheres_center.size(), clearing_objects.spheres_radius.size());
    return false;
  }

  if (!clearing_objects.HasObjects()) {
    RCLCPP_WARN(
      this->get_logger(),
      "No clearing objects provided (neither AABBs nor spheres)");
    return false;
  }

  request->aabbs_to_clear_min_m = clearing_objects.aabbs_min;
  request->aabbs_to_clear_size_m = clearing_objects.aabbs_size;
  request->spheres_to_clear_center_m = clearing_objects.spheres_center;
  request->spheres_to_clear_radius_m = clearing_objects.spheres_radius;

  RCLCPP_INFO(
    this->get_logger(),
    "ESDF request: clearing %zu AABB(s) and %zu sphere(s) from attached object",
    clearing_objects.aabbs_min.size(), clearing_objects.spheres_center.size());

  auto future = esdf_client_->async_send_request(request);
  auto status = future.wait_for(kServiceResponseTimeout);

  if (status != std::future_status::ready) {
    RCLCPP_ERROR(
      this->get_logger(), "ESDF service call timed out after %d seconds",
      static_cast<int>(kServiceResponseTimeout.count()));
    return false;
  }

  auto response = future.get();

  if (!response) {
    RCLCPP_ERROR(this->get_logger(), "ESDF service returned null response");
    return false;
  }

  if (!response->success) {
    RCLCPP_ERROR(this->get_logger(), "ESDF service returned failure status");
    return false;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "ESDF clearing request succeeded. Object voxels cleared from world representation.");

  return true;
}

bool ObjectAttachmentNode::ClearObjectVoxelsFromWorld(
  const visualization_msgs::msg::Marker & object_config)
{
  RCLCPP_DEBUG(this->get_logger(), "Clearing attached object voxels from world representation...");

  // Lookup transform: pose of grasp frame expressed in ESDF reference frame (world frame)
  geometry_msgs::msg::TransformStamped world_pose_grasp_tf_stamped;
  try {
    world_pose_grasp_tf_stamped = tf_buffer_->lookupTransform(
      esdf_reference_frame_,     // Frame to express result in (world)
      object_config.header.frame_id,  // Frame to transform from (grasp)
      tf2::TimePointZero,   // Latest available
      kTfLookupTimeout);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(
      this->get_logger(),
      "ESDF clearing failed: transform lookup from '%s' to '%s' failed: %s",
      object_config.header.frame_id.c_str(), esdf_reference_frame_.c_str(), ex.what());
    return false;
  }

  // Convert geometry_msgs::msg::Transform -> tf2::Transform for composition
  tf2::Transform world_pose_grasp_tf2;
  tf2::fromMsg(world_pose_grasp_tf_stamped.transform, world_pose_grasp_tf2);

  // Convert object pose in grasp frame: geometry_msgs::msg::Pose -> tf2::Transform
  tf2::Transform grasp_pose_object_tf2;
  tf2::fromMsg(object_config.pose, grasp_pose_object_tf2);

  // world_pose_object = world_pose_grasp * grasp_pose_object
  tf2::Transform world_pose_object_tf2 = world_pose_grasp_tf2 * grasp_pose_object_tf2;

  // Convert tf2::Transform -> geometry_msgs::msg::Pose
  geometry_msgs::msg::Pose world_pose_object;
  tf2::toMsg(world_pose_object_tf2, world_pose_object);

  // Calculate clearing regions (AABBs/spheres) for the object
  EsdfClearingObjects clearing_objects = CalculateClearingRegions(world_pose_object, object_config);

  if (!clearing_objects.HasObjects()) {
    RCLCPP_ERROR(
      this->get_logger(),
      "ESDF clearing failed: no clearing regions could be calculated for object type %d",
      object_config.type);
    return false;
  }

  if (!SendEsdfClearingRequest(clearing_objects)) {
    RCLCPP_ERROR(
      this->get_logger(),
      "ESDF clearing failed: clearing request to nvblox was unsuccessful");
    return false;
  }

  RCLCPP_DEBUG(this->get_logger(), "ESDF clearing request completed successfully");
  return true;
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::cumotion::ObjectAttachmentNode)
