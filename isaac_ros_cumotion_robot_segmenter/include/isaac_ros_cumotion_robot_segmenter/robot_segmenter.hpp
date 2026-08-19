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

#ifndef ISAAC_ROS_CUMOTION_ROBOT_SEGMENTER__ROBOT_SEGMENTER_HPP_
#define ISAAC_ROS_CUMOTION_ROBOT_SEGMENTER__ROBOT_SEGMENTER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <isaac_ros_cumotion_interfaces/srv/get_robot_description.hpp>

#include "cumotion/cumotion.h"
#include "cumotion/robot_description.h"
#include "cumotion/robot_segmenter.h"
#include "cumotion/vision.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"


namespace nvidia
{
namespace isaac_ros
{
namespace manipulator
{

namespace Nitros = nvidia::isaac_ros::nitros;

class RobotSegmenter : public rclcpp::Node
{
public:
  explicit RobotSegmenter(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~RobotSegmenter();

private:
  // QoS parameters
  int sync_queue_size_;
  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  int64_t memory_pool_num_blocks_;

  // Subscribers for individual callbacks
  rclcpp::Subscription<Nitros::NitrosImage>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_;
  // Subscribe to joint state message, used to determine robot pose
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  // Publisher for robot mask
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr robot_mask_pub_;

  // Publisher for robot depth
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr robot_depth_pub_;

  // TF buffer and listener for receiving extrinsics
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 4x4 transformation matrix from depth to color frame
  std::optional<Eigen::Matrix4d> color_pose_depth_;

  // Performance logging flag
  bool enable_performance_logging_;

  // Mutex for node operations
  mutable std::mutex node_mutex_;

  // Cached camera infos used when use_cached_camera_info_ is true
  std::optional<sensor_msgs::msg::CameraInfo> depth_camera_info_;

  // Cached cuMotion camera intrinsics
  std::optional<cumotion::CameraIntrinsics> depth_camera_intrinsics_;

  // Cached joint state
  std::optional<sensor_msgs::msg::JointState> joint_state_;

  // Robot description loaded from URDF and XRDF
  std::unique_ptr<cumotion::RobotDescription> robot_description_;

  // Robot segmenter for masking robot from depth images
  std::unique_ptr<cumotion::RobotSegmenter> robot_segmenter_;

  // Additional buffer distance for robot geometry inflation
  double additional_buffer_distance_;

  // Cache of joint name to index
  std::unordered_map<std::string, int> name_index_cache_;
  bool name_index_cached_{false};

  // Robot segmenter joint confifuration
  std::vector<std::string> ordered_joint_names_;

  // Preallocated CPU buffers for robot segmentation (segmenter requires HOST residency)
  std::vector<uint16_t> cpu_input_buffer_uint16_;
  std::vector<float> cpu_input_buffer_float_;
  std::vector<uint16_t> cpu_output_buffer_uint16_;
  std::vector<float> cpu_output_buffer_float_;
  std::vector<uint16_t> cpu_mask_buffer_uint16_;
  std::vector<float> cpu_mask_buffer_float_;
  int buffer_width_{0};
  int buffer_height_{0};

  bool is_mono_{true};
  std::string robot_base_frame_;

  // Camera pose in robot frame
  cumotion::Pose3 robot_pose_camera_;

  // Preallocated GPU output buffer for publishing
  float * gpu_output_buffer_{nullptr};
  size_t gpu_output_buffer_size_{0};

  // Service client for fetching robot description from cumotion_planner
  rclcpp::Client<isaac_ros_cumotion_interfaces::srv::GetRobotDescription>::SharedPtr
    get_robot_description_client_;

  // Subscriber for robot description reload signal
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reload_robot_description_sub_;

  // Service name for getting robot description
  std::string robot_description_service_name_;

  // CUDA resources
  ::nvidia::isaac_ros::common::CudaStreamPtr cuda_stream_;
  nvidia::isaac_ros::nitros::CUDAMemoryPool pool_;

  // Callback group for service client
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;

  void DepthCallback(const Nitros::NitrosImage::ConstSharedPtr & msg);
  void DepthCameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

  // Callback for robot description reload signal
  void ReloadRobotDescriptionCallback(const std_msgs::msg::Bool::SharedPtr msg);

  // Initialize camera pose from camera info
  void InitializeCameraPose(const sensor_msgs::msg::CameraInfo & depth_camera_info);

  // Fetch robot description from service and reinitialize segmenter
  void FetchAndReinitializeRobotDescription();

  // Initialize robot segmenter from URDF/XRDF strings
  void InitializeRobotSegmenterWithCustomDescription(
    const std::string & urdf, const std::string & xrdf);

  // Shared computation used by both synchronized and individual callbacks
  void ComputeAndPublishRobotMask(
    const Nitros::NitrosImage & depth_msg,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    const sensor_msgs::msg::JointState & joint_state);
};

}  // namespace manipulator
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_ROBOT_SEGMENTER__ROBOT_SEGMENTER_HPP_
