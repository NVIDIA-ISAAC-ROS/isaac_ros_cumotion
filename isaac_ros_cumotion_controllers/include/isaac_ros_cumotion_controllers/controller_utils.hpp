// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ISAAC_ROS_CUMOTION_CONTROLLERS__CONTROLLER_UTILS_HPP_
#define ISAAC_ROS_CUMOTION_CONTROLLERS__CONTROLLER_UTILS_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "geometry_msgs/msg/pose.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion_controllers
{

using PoseData = std::pair<Eigen::Vector3d, Eigen::Quaterniond>;

/// True iff both the translation and the quaternion components of `pose` are
/// finite. Gating `setPoseTarget` on this prevents NaN from RMPflow → integrators
/// → command interfaces if a malformed reference pose slips through.
inline bool IsPoseDataFinite(const PoseData & pose)
{
  return pose.first.allFinite() && pose.second.coeffs().allFinite();
}

/// True iff `pose` is non-trivial (not the "unset" sentinel: position near
/// origin AND orientation near identity) AND has a usable orientation. A
/// degenerate (near-zero or non-finite) quaternion would NaN out when
/// normalized, so we reject those up front instead of normalizing first.
inline bool IsReferencePoseValid(const geometry_msgs::msg::Pose & pose)
{
  constexpr double kPosTolerance = 1e-3;
  constexpr double kRotTolerance = 1e-3;
  constexpr double kMinQuatNorm = 1e-6;
  const Eigen::Vector3d position(pose.position.x, pose.position.y, pose.position.z);
  const Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x,
    pose.orientation.y, pose.orientation.z);
  if (!position.allFinite() || !q.coeffs().allFinite() ||
    q.coeffs().norm() < kMinQuatNorm)
  {
    return false;
  }
  const auto rot_err = q.normalized().angularDistance(Eigen::Quaterniond::Identity());
  return !(position.isZero(kPosTolerance) && rot_err < kRotTolerance);
}

/// Look up the index in `interfaces` of each entry in `names` (matched by
/// LoanedCommandInterface::get_name / LoanedStateInterface::get_name).
/// Returns std::nullopt as soon as any name is not found.
inline std::optional<std::vector<size_t>> FindInterfaceIndices(
  const std::vector<std::string> & names,
  const auto & interfaces)
{
  std::vector<size_t> indices;
  indices.reserve(names.size());
  for (const auto & name : names) {
    bool found = false;
    for (size_t i = 0; i < interfaces.size(); ++i) {
      if (interfaces[i].get_name() == name) {
        indices.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) {return std::nullopt;}
  }
  return indices;
}

}  // namespace cumotion_controllers
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_CUMOTION_CONTROLLERS__CONTROLLER_UTILS_HPP_
