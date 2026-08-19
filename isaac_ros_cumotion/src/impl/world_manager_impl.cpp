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

#include "isaac_ros_cumotion/impl/world_manager_impl.hpp"

#include <Eigen/Geometry>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cumotion/world_inspector.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

#include <assimp/Importer.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

constexpr const char * kFileUriPrefix = "file://";
constexpr size_t kFileUriPrefixLength = 7;
constexpr unsigned int kAssimpPostProcessFlags =
  aiProcess_Triangulate |
  aiProcess_JoinIdenticalVertices |
  aiProcess_GenNormals;

WorldManagerImpl::WorldManagerImpl(const Config & config, const rclcpp::Logger & logger)
: logger_(logger),
  config_(config),
  mesh_importer_(std::make_unique<Assimp::Importer>())
{
  Initialize();
}

bool WorldManagerImpl::Initialize()
{
  std::lock_guard<std::mutex> lock(world_mutex_);

  // Create the world using factory function.
  world_ = cumotion_lib::CreateWorld();

  // Add ground plane if configured.
  if (config_.add_ground_plane) {
    AddGroundPlane();
  }

  // Create world view for planning
  world_view_ = world_->addWorldView();

  return true;
}

bool WorldManagerImpl::UpdateWorldObjects(
  const std::vector<moveit_msgs::msg::CollisionObject> & collision_objects)
{
  // Step 1: Update primitive/mesh obstacles under the world mutex.
  bool success = true;
  {
    std::lock_guard<std::mutex> lock(world_mutex_);

    // Clear existing obstacles (except ground plane and ESDF) using iterator erase.
    for (auto it = obstacle_handles_.begin(); it != obstacle_handles_.end(); ) {
      if (it->first == "ground_plane" || it->first == "world_voxel") {
        ++it;
      } else {
        world_->removeObstacle(it->second);
        it = obstacle_handles_.erase(it);
      }
    }

    // Add new collision objects.
    success = AddCollisionObjects(collision_objects);

    // Update world view.
    world_view_.update();
  }

  return success;
}

void WorldManagerImpl::ClearWorld()
{
  std::lock_guard<std::mutex> lock(world_mutex_);

  // Remove all active obstacles except the ground plane using iterator erase.
  for (auto it = obstacle_handles_.begin(); it != obstacle_handles_.end(); ) {
    if (it->first == "ground_plane") {
      ++it;
    } else {
      world_->removeObstacle(it->second);
      it = obstacle_handles_.erase(it);
    }
  }

  // Re-add ground plane if it was configured.
  if (config_.add_ground_plane && !ground_plane_added_) {
    AddGroundPlane();
  }

  // Update world view.
  world_view_.update();
}

Eigen::Vector3i WorldManagerImpl::GetGridShape() const
{
  std::lock_guard<std::mutex> lock(world_mutex_);
  return config_.grid_shape;
}

Eigen::Vector3d WorldManagerImpl::GetGridOrigin() const
{
  std::lock_guard<std::mutex> lock(world_mutex_);
  return config_.grid_origin_m;
}

double WorldManagerImpl::GetVoxelSize() const
{
  std::lock_guard<std::mutex> lock(world_mutex_);
  return config_.voxel_size_m;
}

void WorldManagerImpl::SetRobotBaseFrame(const std::string & robot_base_frame)
{
  std::lock_guard<std::mutex> lock(world_mutex_);
  config_.robot_base_frame = robot_base_frame;
}

std::vector<Eigen::Vector4f> WorldManagerImpl::CalculateOccupancyForVisualization()
{
  std::lock_guard<std::mutex> lock(world_mutex_);

  std::vector<Eigen::Vector4f> voxels;

  if (!has_esdf_obstacle_) {
    return voxels;
  }

  // Check if grid has been initialized from ESDF response
  if (config_.grid_shape == Eigen::Vector3i::Zero()) {
    RCLCPP_ERROR(
      logger_,
      "Cannot visualize: grid not initialized from ESDF response. "
      "Ensure read_esdf_world is enabled and nvblox is publishing ESDF data.");
    return voxels;
  }

  // Compute grid size from shape and voxel size.
  const Eigen::Vector3d grid_size_m(
    config_.grid_shape.x() * config_.voxel_size_m,
    config_.grid_shape.y() * config_.voxel_size_m,
    config_.grid_shape.z() * config_.voxel_size_m
  );

  // Validate computed grid size
  if ((grid_size_m.array() <= 0.0).any()) {
    RCLCPP_ERROR(
      logger_,
      "Cannot visualize: invalid computed grid size [%.2f, %.2f, %.2f]m from shape [%d, %d, %d]. "
      "Check voxel_size configuration.",
      grid_size_m.x(), grid_size_m.y(), grid_size_m.z(),
      config_.grid_shape.x(), config_.grid_shape.y(), config_.grid_shape.z());
    return voxels;
  }

  // Validate grid origin (check for NaN or infinity)
  if (!config_.grid_origin_m.array().isFinite().all()) {
    RCLCPP_ERROR(
      logger_,
      "Cannot visualize: invalid grid origin [%.2f, %.2f, %.2f]m. "
      "Grid origin contains NaN or infinity values.",
      config_.grid_origin_m.x(), config_.grid_origin_m.y(), config_.grid_origin_m.z());
    return voxels;
  }

  // Create an inspector over the current world view.
  auto inspector = cumotion_lib::CreateWorldInspector(world_view_);
  if (!inspector) {
    RCLCPP_ERROR(logger_, "Failed to create WorldInspector for visualization");
    return voxels;
  }

  // Use grid geometry of the workspace
  const Eigen::Vector3d & min_corner = config_.grid_origin_m;
  const Eigen::Vector3d max_corner = min_corner + grid_size_m;

  // Use publish_voxel_size as sampling resolution; fall back to voxel_size if needed.
  const double step =
    (config_.publish_voxel_size > 0.0) ? config_.publish_voxel_size : config_.voxel_size_m;
  if (step <= 0.0) {
    RCLCPP_ERROR(logger_, "Invalid voxel step size for visualization: %.4f", step);
    return voxels;
  }

  const int nx = static_cast<int>(std::ceil(grid_size_m.x() / step));
  const int ny = static_cast<int>(std::ceil(grid_size_m.y() / step));
  const int nz = static_cast<int>(std::ceil(grid_size_m.z() / step));

  const std::size_t max_voxels =
    (config_.max_publish_voxels > 0) ?
    static_cast<std::size_t>(config_.max_publish_voxels) :
    std::numeric_limits<std::size_t>::max();

  voxels.reserve(
    std::min<std::size_t>(
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz),
      max_voxels));

  // Sample at voxel centers within the workspace AABB and mark occupied locations.
  for (int ix = 0; ix < nx && voxels.size() < max_voxels; ++ix) {
    const double x = min_corner.x() + (ix + 0.5) * step;
    if (x < min_corner.x() || x > max_corner.x()) {
      continue;
    }
    for (int iy = 0; iy < ny && voxels.size() < max_voxels; ++iy) {
      const double y = min_corner.y() + (iy + 0.5) * step;
      if (y < min_corner.y() || y > max_corner.y()) {
        continue;
      }
      for (int iz = 0; iz < nz && voxels.size() < max_voxels; ++iz) {
        const double z = min_corner.z() + (iz + 0.5) * step;
        if (z < min_corner.z() || z > max_corner.z()) {
          continue;
        }

        const Eigen::Vector3d p(x, y, z);
        const double distance = inspector->distanceTo(esdf_obstacle_handle_, p);
        if (distance < 0.0) {
          voxels.emplace_back(
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(z),
            1.0F);  // Occupancy value (1.0 = fully occupied)
        }
      }
    }
  }

  return voxels;
}

bool WorldManagerImpl::ProcessEsdfResponse(
  const nvblox_msgs::srv::EsdfAndGradients::Response & response)
{
  // Extract grid shape from nvblox response
  const auto & esdf_array = response.esdf_and_gradients;
  if (esdf_array.layout.dim.size() < 3) {
    RCLCPP_ERROR(
      logger_,
      "Invalid ESDF array layout: expected >= 3 dimensions, got %zu",
      esdf_array.layout.dim.size());
    return false;
  }
  Eigen::Vector3i esdf_shape(
    esdf_array.layout.dim[0].size,
    esdf_array.layout.dim[1].size,
    esdf_array.layout.dim[2].size
  );

  // Get grid origin from response
  Eigen::Vector3d grid_origin(
    response.origin_m.x,
    response.origin_m.y,
    response.origin_m.z
  );

  // Create SDF obstacle structure.
  auto sdf_result = CreateSdfObstacleStructure(esdf_shape, grid_origin, response.voxel_size_m);

  if (!sdf_result.first) {
    RCLCPP_ERROR(logger_, "Failed to create SDF obstacle from ESDF");
    return false;
  }

  // Serialize cached-grid updates and world updates.
  std::lock_guard<std::mutex> lock(world_mutex_);

  // On first update, populate Config grid geometry from nvblox.
  if (config_.grid_shape == Eigen::Vector3i::Zero()) {
    config_.grid_shape = esdf_shape;
    config_.grid_origin_m = grid_origin;
    config_.voxel_size_m = response.voxel_size_m;
    RCLCPP_INFO(
      logger_,
      "Initialized grid from nvblox: shape=[%d, %d, %d] voxels, "
      "origin=[%.2f, %.2f, %.2f]m, voxel_size=%.3fm",
      esdf_shape.x(), esdf_shape.y(), esdf_shape.z(),
      grid_origin.x(), grid_origin.y(), grid_origin.z(),
      config_.voxel_size_m);
  } else {
    // Subsequent updates must match initial geometry.
    if (esdf_shape != config_.grid_shape) {
      RCLCPP_ERROR(
        logger_,
        "ESDF grid shape mismatch: expected [%d, %d, %d], got [%d, %d, %d]",
        config_.grid_shape.x(), config_.grid_shape.y(), config_.grid_shape.z(),
        esdf_shape.x(), esdf_shape.y(), esdf_shape.z());
      return false;
    }
    // Check for voxel size changes
    if (std::abs(response.voxel_size_m - config_.voxel_size_m) > 1e-4) {
      RCLCPP_WARN(
        logger_,
        "ESDF voxel size changed: %.4f -> %.4f (updating config)",
        config_.voxel_size_m, response.voxel_size_m);
      config_.voxel_size_m = response.voxel_size_m;
    }
  }

  // Remove old ESDF obstacle if it exists
  if (has_esdf_obstacle_) {
    world_->removeObstacle(esdf_obstacle_handle_);
  }

  // Add new SDF obstacle
  esdf_obstacle_handle_ = world_->addObstacle(*(sdf_result.first), sdf_result.second);
  has_esdf_obstacle_ = true;

  world_->setSdfGridValuesFromHost(
    esdf_obstacle_handle_, esdf_array.data.data(),
    cumotion_lib::Obstacle::Grid::Precision::FLOAT);

  // Update world view.
  world_view_.update();

  RCLCPP_INFO(logger_, "Updated ESDF grid successfully");
  return true;
}

std::pair<std::unique_ptr<cumotion_lib::Obstacle>, cumotion_lib::Pose3>
WorldManagerImpl::CreateSdfObstacleStructure(
  const Eigen::Vector3i & grid_shape,
  const Eigen::Vector3d & grid_origin,
  double voxel_size_m)
{
  // Create pose for SDF obstacle: place origin at minimal voxel corner
  Eigen::Quaterniond identity_quat(1, 0, 0, 0);
  cumotion_lib::Pose3 grid_pose(
    cumotion_lib::Rotation3(identity_quat),
    grid_origin
  );

  /**
   * Create SDF obstacle. Note that here we are only creating the structure; the actual data is
   * set separately via setSdfGridValuesFromHost.
   */
  auto sdf_obstacle = cumotion_lib::CreateObstacle(cumotion_lib::Obstacle::Type::SDF);
  cumotion_lib::Obstacle::Grid grid(
    grid_shape.x(),
    grid_shape.y(),
    grid_shape.z(),
    voxel_size_m,
    cumotion_lib::Obstacle::Grid::Precision::FLOAT,
    cumotion_lib::Obstacle::Grid::Precision::FLOAT
  );
  sdf_obstacle->setAttribute(cumotion_lib::Obstacle::Attribute::GRID, grid);

  return {std::move(sdf_obstacle), grid_pose};
}

void WorldManagerImpl::AddGroundPlane()
{
  // Create ground plane as a box obstacle using configured dimensions.
  Eigen::Quaterniond identity_quat(1, 0, 0, 0);
  cumotion_lib::Pose3 ground_pose(
    cumotion_lib::Rotation3(identity_quat),
    Eigen::Vector3d(0, 0, config_.ground_plane_z_offset));

  auto ground = cumotion_lib::CreateObstacle(cumotion_lib::Obstacle::Type::CUBOID);
  ground->setAttribute(
    cumotion_lib::Obstacle::Attribute::SIDE_LENGTHS,
    Eigen::Vector3d(
      config_.ground_plane_size_x,
      config_.ground_plane_size_y,
      config_.ground_plane_thickness));

  auto handle = world_->addObstacle(*ground, ground_pose);
  obstacle_handles_["ground_plane"] = handle;
  ground_plane_added_ = true;
}

bool WorldManagerImpl::AddCollisionObjects(
  const std::vector<moveit_msgs::msg::CollisionObject> & objects)
{
  bool all_supported = true;

  for (const auto & obj : objects) {
    auto obstacles = ToCuMotionObstacles(obj);

    if (obstacles.empty() &&
      (!obj.primitives.empty() || !obj.meshes.empty()))
    {
      RCLCPP_WARN(
        logger_,
        "Some primitives/meshes in object %s could not be converted",
        obj.id.c_str());
      all_supported = false;
    }

    for (auto & [obstacle, pose] : obstacles) {
      // Generate unique name for this obstacle
      std::string name = obj.id + "_" + std::to_string(obstacle_handles_.size());
      auto handle = world_->addObstacle(*obstacle, pose);
      obstacle_handles_[name] = handle;
    }
  }

  return all_supported;
}

// Calculate AABBs to clear from ESDF for an object at the given pose
std::unique_ptr<EsdfClearingObjects> WorldManagerImpl::CalculateAabbsToClear(
  const geometry_msgs::msg::Pose & world_pose_object,
  const std::string & mesh_resource,
  const std::vector<double> & object_esdf_clearing_padding,
  const std::string & object_shape,
  const geometry_msgs::msg::Vector3 & object_scale)
{
  auto clearing_objects = std::make_unique<EsdfClearingObjects>();

  // Validate input
  if (object_esdf_clearing_padding.size() != 3) {
    RCLCPP_ERROR(logger_, "object_esdf_clearing_padding must have 3 elements");
    return clearing_objects;
  }

  // Convert padding to Eigen vector for easier math
  Eigen::Vector3d padding(
    object_esdf_clearing_padding[0],
    object_esdf_clearing_padding[1],
    object_esdf_clearing_padding[2]
  );

  // Dispatch to shape-specific clearing region calculation.
  bool success = false;
  if (object_shape == "SPHERE") {
    success = CalculateAabbsForSphere(
      world_pose_object, padding, object_scale,
      clearing_objects.get());
  } else if (object_shape == "CUBOID") {
    success = CalculateAabbsForCuboid(
      world_pose_object, padding, object_scale,
      clearing_objects.get());
  } else if (object_shape == "CUSTOM_MESH") {
    success = CalculateAabbsForMesh(
      world_pose_object, mesh_resource, padding, object_scale, clearing_objects.get());
  } else {
    RCLCPP_WARN(logger_, "Unknown object shape: %s", object_shape.c_str());
  }

  if (!success) {
    RCLCPP_WARN(
      logger_, "Failed to calculate clearing regions for shape: %s", object_shape.c_str());
  } else {
    RCLCPP_INFO(
      logger_,
      "Successfully calculated %zu AABB(s) and %zu sphere(s) for clearing",
      clearing_objects->aabbs_min.size(), clearing_objects->spheres_center.size());
  }

  // Free the loaded scene if a mesh was loaded.
  if (object_shape == "CUSTOM_MESH") {
    mesh_importer_->FreeScene();
  }

  return clearing_objects;
}

bool WorldManagerImpl::CalculateAabbsForSphere(
  const geometry_msgs::msg::Pose & world_pose_object,
  const Eigen::Vector3d & padding,
  const geometry_msgs::msg::Vector3 & object_scale,
  EsdfClearingObjects * clearing_objects)
{
  RCLCPP_DEBUG(logger_, "Calculating clearing regions for SPHERE object");

  // For sphere, create a single sphere to clear.
  double radius = std::max({object_scale.x, object_scale.y, object_scale.z}) / 2.0;
  radius += padding.maxCoeff();  // Add max padding to radius

  // Extract center position in world frame.
  geometry_msgs::msg::Point center;
  center.x = world_pose_object.position.x;
  center.y = world_pose_object.position.y;
  center.z = world_pose_object.position.z;

  clearing_objects->spheres_center.push_back(center);
  clearing_objects->spheres_radius.push_back(radius);

  RCLCPP_DEBUG(
    logger_, "Clearing sphere: center=(%.3f, %.3f, %.3f), radius=%.3f",
    center.x, center.y, center.z, radius);

  return true;
}

bool WorldManagerImpl::CalculateAabbsForCuboid(
  const geometry_msgs::msg::Pose & world_pose_object,
  const Eigen::Vector3d & padding,
  const geometry_msgs::msg::Vector3 & object_scale,
  EsdfClearingObjects * clearing_objects)
{
  RCLCPP_DEBUG(logger_, "Calculating clearing regions for CUBOID object");

  // Generate cuboid vertices in local frame with actual dimensions.
  std::vector<Eigen::Vector3d> vertices = GenerateCuboidVertices(
    object_scale.x, object_scale.y, object_scale.z);

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
      padding,
      aabb_min,
      aabb_size))
  {
    return false;
  }

  clearing_objects->aabbs_min.push_back(aabb_min);
  clearing_objects->aabbs_size.push_back(aabb_size);

  RCLCPP_DEBUG(
    logger_,
    "Clearing AABB for cuboid: min=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)",
    aabb_min.x, aabb_min.y, aabb_min.z,
    aabb_size.x, aabb_size.y, aabb_size.z);

  return true;
}

bool WorldManagerImpl::CalculateAabbsForMesh(
  const geometry_msgs::msg::Pose & world_pose_object,
  const std::string & mesh_resource,
  const Eigen::Vector3d & padding,
  const geometry_msgs::msg::Vector3 & object_scale,
  EsdfClearingObjects * clearing_objects)
{
  RCLCPP_DEBUG(logger_, "Calculating clearing regions for CUSTOM_MESH object");

  // Load mesh using helper function.
  if (!LoadMesh(mesh_resource)) {
    return false;
  }

  // Extract vertices from the loaded scene.
  const aiScene * const scene = mesh_importer_->GetScene();
  if (scene == nullptr) {
    RCLCPP_ERROR(logger_, "Scene is null after loading mesh");
    return false;
  }

  std::vector<Eigen::Vector3d> vertices;
  for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
    const aiMesh * mesh = scene->mMeshes[m];
    if (mesh == nullptr) {
      RCLCPP_ERROR(logger_, "Null mesh encountered at index %u; aborting mesh processing", m);
      return false;
    }
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      const aiVector3D & v = mesh->mVertices[i];
      vertices.emplace_back(v.x, v.y, v.z);
    }
  }

  if (vertices.empty()) {
    RCLCPP_ERROR(logger_, "Mesh has no valid vertices");
    return false;
  }

  RCLCPP_DEBUG(logger_, "Extracted %zu vertices from mesh", vertices.size());

  // Build transformation matrix: T * R * S (scale, then rotate, then translate).
  // Note: Methods are called in reverse order because they post-multiply.
  Eigen::Affine3d world_transform = Eigen::Affine3d::Identity();
  world_transform.translate(
    Eigen::Vector3d(
      world_pose_object.position.x,
      world_pose_object.position.y,
      world_pose_object.position.z));
  world_transform.rotate(
    Eigen::Quaterniond(
      world_pose_object.orientation.w,
      world_pose_object.orientation.x,
      world_pose_object.orientation.y,
      world_pose_object.orientation.z));
  world_transform.scale(
    Eigen::Vector3d(
      object_scale.x,
      object_scale.y,
      object_scale.z));

  geometry_msgs::msg::Point aabb_min;
  geometry_msgs::msg::Vector3 aabb_size;
  if (!ComputeAABBFromVertices(
      vertices,
      world_transform,
      padding,
      aabb_min,
      aabb_size))
  {
    return false;
  }

  clearing_objects->aabbs_min.push_back(aabb_min);
  clearing_objects->aabbs_size.push_back(aabb_size);

  RCLCPP_DEBUG(
    logger_,
    "Clearing AABB for custom mesh: min=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)",
    aabb_min.x, aabb_min.y, aabb_min.z,
    aabb_size.x, aabb_size.y, aabb_size.z);

  return true;
}

std::vector<Eigen::Vector3d> WorldManagerImpl::GenerateCuboidVertices(
  double size_x, double size_y, double size_z)
{
  // Calculate half-extents from full dimensions.
  const double hx = size_x / 2.0;
  const double hy = size_y / 2.0;
  const double hz = size_z / 2.0;

  // Define the 8 vertices of the cuboid in local object frame.
  return {
    Eigen::Vector3d(-hx, -hy, -hz),
    Eigen::Vector3d(hx, -hy, -hz),
    Eigen::Vector3d(hx, hy, -hz),
    Eigen::Vector3d(-hx, hy, -hz),
    Eigen::Vector3d(-hx, -hy, hz),
    Eigen::Vector3d(hx, -hy, hz),
    Eigen::Vector3d(hx, hy, hz),
    Eigen::Vector3d(-hx, hy, hz)
  };
}

bool WorldManagerImpl::LoadMesh(const std::string & mesh_resource)
{
  if (mesh_resource.empty()) {
    RCLCPP_ERROR(logger_, "Mesh resource path is empty");
    return false;
  }

  // Extract file path from resource URI (handle "file://", "package://", etc.).
  std::string mesh_path = mesh_resource;
  if (mesh_path.find(kFileUriPrefix) == 0) {
    mesh_path = mesh_path.substr(kFileUriPrefixLength);
  } else if (mesh_path.find("package://") == 0) {
    RCLCPP_ERROR(
      logger_,
      "package:// URIs not yet supported for ESDF clearing. Please use absolute file paths.");
    return false;
  }

  // Use std::filesystem to validate the file path.
  std::filesystem::path file_path(mesh_path);
  if (!std::filesystem::exists(file_path)) {
    RCLCPP_ERROR(logger_, "Mesh file does not exist: %s", mesh_path.c_str());
    return false;
  }

  RCLCPP_DEBUG(logger_, "Loading mesh for AABB calculation: %s", mesh_path.c_str());

  // Load mesh into the importer.
  mesh_importer_->ReadFile(mesh_path, kAssimpPostProcessFlags);

  const aiScene * const scene = mesh_importer_->GetScene();
  if (!scene || !scene->HasMeshes()) {
    RCLCPP_ERROR(
      logger_, "Failed to load mesh from %s: %s",
      mesh_path.c_str(), mesh_importer_->GetErrorString());
    return false;
  }

  RCLCPP_DEBUG(logger_, "Loaded mesh with %d submeshes", scene->mNumMeshes);

  return true;
}

bool WorldManagerImpl::ComputeAABBFromVertices(
  const std::vector<Eigen::Vector3d> & vertices,
  const Eigen::Affine3d & world_transform,
  const Eigen::Vector3d & padding,
  geometry_msgs::msg::Point & aabb_min,
  geometry_msgs::msg::Vector3 & aabb_size)
{
  if (vertices.empty()) {
    return false;
  }

  // Transform each vertex to world frame and track component-wise min/max.
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

  // Apply object_esdf_clearing_padding per-side: each AABB face is pushed
  // outward by the full padding value so the clearance between the object
  // surface and the clearing boundary equals object_esdf_clearing_padding.
  min_corner -= padding;
  max_corner += padding;

  // Pack into output messages.
  aabb_min.x = min_corner(0);
  aabb_min.y = min_corner(1);
  aabb_min.z = min_corner(2);

  aabb_size.x = max_corner(0) - min_corner(0);
  aabb_size.y = max_corner(1) - min_corner(1);
  aabb_size.z = max_corner(2) - min_corner(2);

  return true;
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia
