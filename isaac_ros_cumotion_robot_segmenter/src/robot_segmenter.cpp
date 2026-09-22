// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_cumotion_robot_segmenter/robot_segmenter.hpp"

#include <cmath>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace manipulator
{

namespace
{
constexpr const char kDefaultQoS[] = "DEFAULT";
constexpr int kTfLookupTimeoutSeconds = 30;
}  // namespace

RobotSegmenter::RobotSegmenter(const rclcpp::NodeOptions & options)
: rclcpp::Node("robot_segmenter", options),
  sync_queue_size_(declare_parameter<int>("sync_queue_size", 10)),
  input_qos_{::isaac_ros::common::AddQosParameter(
      *this, kDefaultQoS, "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(
      *this, kDefaultQoS, "output_qos")},
  enable_performance_logging_(declare_parameter<bool>("enable_performance_logging", false)),
  additional_buffer_distance_(declare_parameter<double>("additional_buffer_distance", 0.0)),
  robot_base_frame_(declare_parameter<std::string>("robot_base_frame", "base_link")),
  robot_description_service_name_(declare_parameter<std::string>(
      "robot_description_service_name",
      "/cumotion/get_robot_description"))
{
  // Load robot description from URDF and XRDF files
  const std::string urdf_path = declare_parameter<std::string>("urdf_path", "");
  const std::string xrdf_path = declare_parameter<std::string>("xrdf_path", "");

  if (urdf_path.empty() || xrdf_path.empty()) {
    throw std::runtime_error("urdf_path and xrdf_path parameters must be provided");
  }

  if (robot_base_frame_.empty()) {
    throw std::runtime_error("robot_base_frame parameter must be provided");
  }

  RCLCPP_INFO(
    get_logger(), "Loading robot description from URDF: %s, XRDF: %s",
    urdf_path.c_str(), xrdf_path.c_str());

  robot_description_ = cumotion::LoadRobotFromFile(xrdf_path, urdf_path);
  if (!robot_description_) {
    throw std::runtime_error("Failed to load robot description from URDF and XRDF files");
  }

  RCLCPP_INFO(
    get_logger(), "Robot description loaded successfully with %d joints",
    robot_description_->numCSpaceCoords());

  auto cspace_coords = robot_description_->numCSpaceCoords();
  for (int i = 0; i < cspace_coords; i++) {
    RCLCPP_INFO(
      get_logger(), "Cspace coord %d: %s", i, robot_description_->cSpaceCoordName(
        i).c_str());
    ordered_joint_names_.push_back(robot_description_->cSpaceCoordName(i));
  }

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options.acceptable_buffer_backends = "any";
  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "depth_image", input_qos_,
    std::bind(&RobotSegmenter::DepthCallback, this, std::placeholders::_1), sub_options);
  depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info_depth", input_qos_,
    std::bind(&RobotSegmenter::DepthCameraInfoCallback, this, std::placeholders::_1), sub_options);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream(
    "isaac_ros_cumotion_robot_segmenter_stream");

  rclcpp::SubscriptionOptions sub_options_joint_state;
  sub_options_joint_state.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options_joint_state.callback_group = create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);

  joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", input_qos_,
    std::bind(&RobotSegmenter::JointStateCallback, this, std::placeholders::_1),
    sub_options_joint_state);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  robot_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "robot_mask", output_qos_, pub_options);
  robot_depth_pub_ = create_publisher<sensor_msgs::msg::Image>(
    "robot_depth", output_qos_, pub_options);

  // Topic name for receiving reload robot description signal
  const std::string reload_topic_name = declare_parameter<std::string>(
    "reload_robot_description_topic", "cumotion/robot_description_updated");

  // Create callback group for service client
  service_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create service client for getting robot description
  get_robot_description_client_ =
    create_client<isaac_ros_cumotion_interfaces::srv::GetRobotDescription>(
    robot_description_service_name_,
    rclcpp::ServicesQoS(),
    service_cb_group_);

  // Subscribe to reload robot description topic
  rclcpp::SubscriptionOptions reload_sub_options;
  reload_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
  reload_robot_description_sub_ = create_subscription<std_msgs::msg::UInt64>(
    reload_topic_name,
    rclcpp::QoS(rclcpp::KeepLast(1)).transient_local(),
    std::bind(&RobotSegmenter::ReloadRobotDescriptionCallback, this, std::placeholders::_1),
    reload_sub_options);

  RCLCPP_INFO(
    get_logger(),
    "Robot segmenter initialized. Listening for reload signal on topic '%s', "
    "will fetch description from service '%s'",
    reload_topic_name.c_str(), robot_description_service_name_.c_str());
}

void RobotSegmenter::InitializeCameraPose(const sensor_msgs::msg::CameraInfo & depth_camera_info)
{
  const std::string camera_frame = depth_camera_info.header.frame_id;

  RCLCPP_INFO(
    get_logger(),
    "Waiting for transform from %s to %s...",
    camera_frame.c_str(),
    robot_base_frame_.c_str());

  // lookupTransform will block until transform is available or timeout expires
  geometry_msgs::msg::TransformStamped stamped;
  try {
    stamped = tf_buffer_->lookupTransform(
      robot_base_frame_, camera_frame,
      tf2::TimePointZero,
      std::chrono::seconds(kTfLookupTimeoutSeconds));
  } catch (const tf2::TransformException & ex) {
    throw std::runtime_error(
            "Failed to lookup transform from '" + camera_frame + "' to '" +
            robot_base_frame_ + "' after " + std::to_string(kTfLookupTimeoutSeconds) +
            " seconds: " + ex.what());
  }

  const Eigen::Isometry3d eig_transform = tf2::transformToEigen(stamped);
  Eigen::Matrix4d robot_pose_camera = eig_transform.matrix();

  std::stringstream ss;
  ss << robot_pose_camera;
  RCLCPP_INFO(get_logger(), "Camera pose in robot frame:\n%s", ss.str().c_str());
  robot_pose_camera_ = cumotion::Pose3(robot_pose_camera);
}

RobotSegmenter::~RobotSegmenter()
{
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "Failed to synchronize CUDA stream");
}

void RobotSegmenter::ComputeAndPublishRobotMask(
  const sensor_msgs::msg::Image & depth_msg,
  const sensor_msgs::msg::JointState & joint_state)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  RCLCPP_DEBUG(get_logger(), "Starting robot segmenter computation");

  const int depth_w = static_cast<int>(depth_msg.width);
  const int depth_h = static_cast<int>(depth_msg.height);

  // Check if buffers are allocated and have correct size
  if (buffer_width_ != depth_w || buffer_height_ != depth_h) {
    RCLCPP_WARN(
      get_logger(), "Buffer size mismatch. Expected %dx%d, got %dx%d. Skipping frame.",
      buffer_width_, buffer_height_, depth_w, depth_h);
    return;
  }

  if (!robot_segmenter_) {
    RCLCPP_WARN(get_logger(), "Robot segmenter not initialized yet, dropping frame");
    return;
  }

  const std::string & encoding = depth_msg.encoding;
  const bool is_uint16 = encoding == sensor_msgs::image_encodings::TYPE_16UC1;
  if (!is_uint16 && encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
    RCLCPP_ERROR(
      get_logger(), "Unsupported depth image encoding '%s'; expected 16UC1 or 32FC1",
      encoding.c_str());
    return;
  }
  if (depth_msg.is_bigendian != 0) {
    RCLCPP_ERROR(get_logger(), "Big-endian depth images are not supported");
    return;
  }

  const size_t num_pixels = static_cast<size_t>(depth_w) * static_cast<size_t>(depth_h);
  const size_t bytes_per_pixel = is_uint16 ? sizeof(uint16_t) : sizeof(float);
  const size_t row_size = static_cast<size_t>(depth_w) * bytes_per_pixel;
  const size_t buffer_size = row_size * static_cast<size_t>(depth_h);
  if (depth_msg.step != row_size || depth_msg.data.size() != buffer_size) {
    RCLCPP_ERROR(
      get_logger(), "Depth image must be tightly packed: step=%u, expected=%zu, size=%zu, "
      "expected size=%zu", depth_msg.step, row_size, depth_msg.data.size(), buffer_size);
    return;
  }

  const size_t current_size = is_uint16 ?
    cpu_input_buffer_uint16_.size() :
    cpu_input_buffer_float_.size();

  if (num_pixels != current_size) {
    // Resize CPU buffers (element count, not bytes).
    if (is_uint16) {
      cpu_input_buffer_uint16_.resize(num_pixels);
      cpu_output_buffer_uint16_.resize(num_pixels);
      cpu_mask_buffer_uint16_.resize(num_pixels);
    } else {
      cpu_input_buffer_float_.resize(num_pixels);
      cpu_output_buffer_float_.resize(num_pixels);
      cpu_mask_buffer_float_.resize(num_pixels);
    }
    RCLCPP_INFO(
      get_logger(), "Resized CPU buffers to %zu elements (%zu bytes)",
      num_pixels, buffer_size);
  }

  // Create joint positions vector (needed for segmentation lambda)
  Eigen::VectorXd joint_positions(static_cast<Eigen::Index>(ordered_joint_names_.size()));
  for (size_t i = 0; i < ordered_joint_names_.size(); ++i) {
    joint_positions[static_cast<Eigen::Index>(i)] =
      joint_state.position[name_index_cache_[ordered_joint_names_[i]]];
  }

  // Generic lambda to handle both uint16_t and float types
  // This avoids code duplication - compiler instantiates two versions
  auto process_depth = [&](auto * input_buf, auto * output_buf, auto * mask_buf) {
      // Copy input data from GPU to CPU buffer
      RCLCPP_DEBUG(
        get_logger(), "Copying depth image from GPU to CPU, buffer size: %zu bytes",
        buffer_size);

      auto depth_read_handle = cuda_buffer_backend::from_input_buffer(
        depth_msg.data, *cuda_stream_);
      CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
          input_buf, depth_read_handle.get_ptr(),
          buffer_size, cudaMemcpyDeviceToHost, *cuda_stream_),
        "Failed to copy depth image from GPU to CPU");
      CHECK_CUDA_ERROR(
        cudaStreamSynchronize(*cuda_stream_),
        "Failed to synchronize CUDA stream after copying depth image");

      auto depth_output_image = std::make_unique<sensor_msgs::msg::Image>();
      depth_output_image->header = depth_msg.header;
      depth_output_image->height = depth_msg.height;
      depth_output_image->width = depth_msg.width;
      depth_output_image->encoding = encoding;
      depth_output_image->is_bigendian = 0;
      depth_output_image->step = static_cast<uint32_t>(row_size);
      depth_output_image->data = cuda_buffer_backend::allocate_buffer(buffer_size);

      auto mask_output_image = std::make_unique<sensor_msgs::msg::Image>();
      mask_output_image->header = depth_msg.header;
      mask_output_image->height = depth_msg.height;
      mask_output_image->width = depth_msg.width;
      mask_output_image->encoding = encoding;
      mask_output_image->is_bigendian = 0;
      mask_output_image->step = static_cast<uint32_t>(row_size);
      mask_output_image->data = cuda_buffer_backend::allocate_buffer(buffer_size);

      // Create CPU DepthImage views with HOST residency
      const auto input_depth_image = cumotion::CreateDepthImageView(
        input_buf, depth_w, depth_h, std::nullopt, cumotion::BufferResidency::HOST);
      auto output_depth_image = cumotion::CreateDepthImageView(
        output_buf, depth_w, depth_h, std::nullopt, cumotion::BufferResidency::HOST);
      auto output_mask = cumotion::CreateDepthImageView(
        mask_buf, depth_w, depth_h, std::nullopt, cumotion::BufferResidency::HOST);

      // Call robot segmenter (operates on CPU buffers)
      robot_segmenter_->segmentDepthImage(
        robot_pose_camera_,
        joint_positions,
        input_depth_image,
        &output_depth_image,
        &output_mask
      );

      {
        auto depth_write_handle = cuda_buffer_backend::from_output_buffer(
          depth_output_image->data, *cuda_stream_);
        auto mask_write_handle = cuda_buffer_backend::from_output_buffer(
          mask_output_image->data, *cuda_stream_);
        CHECK_CUDA_ERROR(
          cudaMemcpyAsync(
            mask_write_handle.get_ptr(), mask_buf, buffer_size,
            cudaMemcpyHostToDevice, *cuda_stream_),
          "Failed to copy segmented mask from CPU to GPU");
        CHECK_CUDA_ERROR(
          cudaMemcpyAsync(
            depth_write_handle.get_ptr(), output_buf, buffer_size,
            cudaMemcpyHostToDevice, *cuda_stream_),
          "Failed to copy segmented depth from CPU to GPU");
        CHECK_CUDA_ERROR(
          cudaStreamSynchronize(*cuda_stream_),
          "Failed to synchronize CUDA stream after copying segmented outputs");
      }

      // Note that for uint16_t 0 = robot, max uint16_t value = background.
      // For float32 0 = robot, 1 = background.
      robot_mask_pub_->publish(std::move(mask_output_image));
      robot_depth_pub_->publish(std::move(depth_output_image));
    };

  // Call with appropriate buffer types based on encoding
  if (is_uint16) {
    process_depth(
      cpu_input_buffer_uint16_.data(),
      cpu_output_buffer_uint16_.data(),
      cpu_mask_buffer_uint16_.data());
  } else {
    process_depth(
      cpu_input_buffer_float_.data(),
      cpu_output_buffer_float_.data(),
      cpu_mask_buffer_float_.data());
  }

  RCLCPP_DEBUG(get_logger(), "Published robot mask");
  if (enable_performance_logging_) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(
      get_logger(), "Robot segmenter computation took %ld milliseconds", duration.count());
  }
}

void RobotSegmenter::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(node_mutex_);

  if (!depth_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received depth image but don't have depth camera info !");
    return;
  } else if (!joint_state_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Have not received joint state yet !");
    return;
  } else {
    ComputeAndPublishRobotMask(*msg, joint_state_.value());
  }
}

void RobotSegmenter::DepthCameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(node_mutex_);

  if (depth_camera_info_.has_value()) {
    RCLCPP_DEBUG_ONCE(get_logger(), "Received depth camera info but already have it !");
    return;
  }

  const sensor_msgs::msg::CameraInfo & depth_camera_info = *msg;

  // Create and cache cuMotion CameraIntrinsics from the ROS CameraInfo K matrix
  // K is a 3x3 matrix in row-major order: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const double fx = depth_camera_info.k[0];
  const double cx = depth_camera_info.k[2];
  const double fy = depth_camera_info.k[4];
  const double cy = depth_camera_info.k[5];
  depth_camera_intrinsics_ = cumotion::CameraIntrinsics(fx, fy, cx, cy);

  InitializeCameraPose(depth_camera_info);

  // Create the robot segmenter now that we have camera intrinsics
  if (!robot_segmenter_ && robot_description_) {
    RCLCPP_INFO(
      get_logger(), "Creating robot segmenter with camera intrinsics: "
      "fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx, fy, cx, cy);

    robot_segmenter_ = cumotion::CreateRobotSegmenter(
      *robot_description_,
      depth_camera_intrinsics_.value(),
      additional_buffer_distance_,
      cumotion::RobotSegmenter::RobotGeometryKind::WORLD_COLLISION_SPHERES);

    if (robot_segmenter_) {
      RCLCPP_INFO(get_logger(), "Robot segmenter created successfully");
    } else {
      throw std::runtime_error("Failed to create robot segmenter");
    }

    // Preallocate CPU buffers for input and output images
    // The segmenter requires HOST residency buffers
    buffer_width_ = static_cast<int>(depth_camera_info.width);
    buffer_height_ = static_cast<int>(depth_camera_info.height);
  }

  depth_camera_info_ = depth_camera_info;
}

void RobotSegmenter::JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(node_mutex_);
  joint_state_ = *msg;

  if (!name_index_cached_) {
    for (size_t i = 0; i < msg->name.size(); i++) {
      name_index_cache_[msg->name[i]] = static_cast<int>(i);
    }
    name_index_cached_ = true;
  }
}

void RobotSegmenter::ReloadRobotDescriptionCallback(const std_msgs::msg::UInt64::SharedPtr msg)
{
  RCLCPP_INFO(
    get_logger(),
    "Received robot description update %lu, fetching from service...",
    msg->data);
  FetchAndReinitializeRobotDescription();
}

void RobotSegmenter::FetchAndReinitializeRobotDescription()
{
  // Check if service is available (non-blocking check)
  if (!get_robot_description_client_->service_is_ready()) {
    RCLCPP_ERROR(
      get_logger(),
      "Service '%s' not available",
      robot_description_service_name_.c_str());
    return;
  }

  // Create request
  auto request =
    std::make_shared<isaac_ros_cumotion_interfaces::srv::GetRobotDescription::Request>();

  RCLCPP_INFO(get_logger(), "Sending async request to GetRobotDescription service...");

  // Call service asynchronously with a callback to avoid deadlock
  // (synchronous wait from within a callback blocks the executor)
  auto future = get_robot_description_client_->async_send_request(
    request,
    [this](rclcpp::Client<isaac_ros_cumotion_interfaces::srv::GetRobotDescription>::
    SharedFuture response_future) {
      try {
        auto response = response_future.get();
        if (response->urdf.empty() || response->xrdf.empty()) {
          RCLCPP_ERROR(
            get_logger(),
            "Service returned empty URDF or XRDF, cannot reinitialize robot segmenter");
          return;
        }

        RCLCPP_INFO(
          get_logger(),
          "Received robot description from service (URDF: %zu bytes, XRDF: %zu bytes)",
          response->urdf.size(), response->xrdf.size());

        // Reinitialize with the new robot description
        InitializeRobotSegmenterWithCustomDescription(response->urdf, response->xrdf);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "Service call failed: %s", e.what());
      }
    });
}

void RobotSegmenter::InitializeRobotSegmenterWithCustomDescription(
  const std::string & urdf,
  const std::string & xrdf)
{
  std::lock_guard<std::mutex> lock(node_mutex_);

  RCLCPP_INFO(get_logger(), "Reinitializing robot description from strings");

  // Load robot description from URDF and XRDF strings
  auto new_robot_description = cumotion::LoadRobotFromMemory(xrdf, urdf);
  if (!new_robot_description) {
    RCLCPP_ERROR(get_logger(), "Failed to load robot description from provided URDF and XRDF");
    return;
  }

  // Update robot description
  robot_description_ = std::move(new_robot_description);

  RCLCPP_INFO(
    get_logger(), "Robot description reloaded successfully with %d joints",
    robot_description_->numCSpaceCoords());

  // Update ordered joint names
  ordered_joint_names_.clear();
  auto cspace_coords = robot_description_->numCSpaceCoords();
  for (int i = 0; i < cspace_coords; i++) {
    RCLCPP_INFO(
      get_logger(), "Cspace coord %d: %s", i,
      robot_description_->cSpaceCoordName(i).c_str());
    ordered_joint_names_.push_back(robot_description_->cSpaceCoordName(i));
  }

  // Reset joint name cache so it gets rebuilt on next joint state
  name_index_cached_ = false;
  name_index_cache_.clear();

  // Recreate robot segmenter if we have camera intrinsics
  if (depth_camera_intrinsics_.has_value()) {
    RCLCPP_INFO(get_logger(), "Recreating robot segmenter with existing camera intrinsics");

    robot_segmenter_ = cumotion::CreateRobotSegmenter(
      *robot_description_,
      depth_camera_intrinsics_.value(),
      additional_buffer_distance_,
      cumotion::RobotSegmenter::RobotGeometryKind::WORLD_COLLISION_SPHERES);

    if (robot_segmenter_) {
      RCLCPP_INFO(get_logger(), "Robot segmenter recreated successfully");
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to recreate robot segmenter");
    }
  } else {
    RCLCPP_INFO(
      get_logger(),
      "Camera intrinsics not yet available, robot segmenter will be created when they arrive");
    robot_segmenter_.reset();
  }
}

}  // namespace manipulator
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::manipulator::RobotSegmenter)
