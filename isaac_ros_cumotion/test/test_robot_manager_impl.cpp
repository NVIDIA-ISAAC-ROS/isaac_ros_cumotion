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

#include "test/test_robot_manager_impl.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "isaac_ros_cumotion/impl/robot_manager_impl.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace cumotion
{

// Test fixture that initializes ROS 2 context and creates a test node for logging.
class RobotManagerImplTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    test_node_ = std::make_shared<rclcpp::Node>("test_robot_manager_node");

    // Create minimal valid test URDF and XRDF files.
    CreateTestRobotFiles();
  }

  void TearDown() override
  {
    test_node_.reset();
    rclcpp::shutdown();
  }

  // Create minimal test URDF and XRDF files in /tmp.
  void CreateTestRobotFiles()
  {
    test_urdf_path_ = "/tmp/test_robot.urdf";
    test_xrdf_path_ = "/tmp/test_robot.xrdf";

    // Minimal valid URDF with 2 revolute joints.
    std::string urdf_content =
      R"(<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  <link name="link2"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  <link name="tool_frame"/>
  <joint name="tool_joint" type="fixed">
    <parent link="link2"/>
    <child link="tool_frame"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
  </joint>
</robot>
)";

    // Minimal valid XRDF matching the URDF.
    std::string xrdf_content =
      R"(format: xrdf
format_version: 1.0
cspace:
  joint_names:
    - joint1
    - joint2
  default_cspace_position: [0.0, 0.0]
  acceleration_limits: [1.0, 1.0]
  jerk_limits: [10.0, 10.0]
  velocity_limits: [1.0, 1.0]
tool_frames: ["tool_frame"]
)";

    std::ofstream urdf_file(test_urdf_path_);
    urdf_file << urdf_content;
    urdf_file.close();

    std::ofstream xrdf_file(test_xrdf_path_);
    xrdf_file << xrdf_content;
    xrdf_file.close();
  }

  // Helpers to provide valid configuration parameters.
  std::string GetValidUrdfPath() const {return test_urdf_path_;}
  std::string GetValidXrdfPath() const {return test_xrdf_path_;}
  std::shared_ptr<rclcpp::Node> test_node_;
  std::string test_urdf_path_;
  std::string test_xrdf_path_;
};

// Tests basic construction of RobotManagerImpl.
TEST_F(RobotManagerImplTest, Construction)
{
  EXPECT_NO_THROW(
        {
          RobotManagerImpl manager(
            GetValidUrdfPath(),
            GetValidXrdfPath(),
            test_node_->get_logger());
        });
}

// Tests initialization with valid robot files.
TEST_F(RobotManagerImplTest, InitializeSuccess)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  // Verify robot description was loaded.
  EXPECT_NE(manager.GetRobotDescription(), nullptr);
  EXPECT_NE(manager.GetKinematics(), nullptr);
}

// Tests initialization fails with invalid URDF path.
TEST_F(RobotManagerImplTest, InitializeInvalidURDF)
{
  EXPECT_THROW(
        {
          RobotManagerImpl manager(
            "/nonexistent/robot.urdf",
            GetValidXrdfPath(),
            test_node_->get_logger());
        }, std::runtime_error);
}

// Tests initialization fails with empty URDF path.
TEST_F(RobotManagerImplTest, InitializeEmptyURDFPath)
{
  EXPECT_THROW(
        {
          RobotManagerImpl manager(
            "",
            GetValidXrdfPath(),
            test_node_->get_logger());
        }, std::runtime_error);
}

// Tests initialization fails with invalid XRDF path.
TEST_F(RobotManagerImplTest, InitializeInvalidXRDF)
{
  EXPECT_THROW(
        {
          RobotManagerImpl manager(
            GetValidUrdfPath(),
            "/nonexistent/robot.xrdf",
            test_node_->get_logger());
        }, std::runtime_error);
}

// Tests getting joint names after initialization.
TEST_F(RobotManagerImplTest, GetJointNames)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  auto joint_names = manager.GetJointNames();
  EXPECT_EQ(joint_names.size(), 2);
  EXPECT_EQ(joint_names[0], "joint1");
  EXPECT_EQ(joint_names[1], "joint2");
}

// Tests getting number of joints.
TEST_F(RobotManagerImplTest, GetNumJoints)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  EXPECT_EQ(manager.GetNumJoints(), 2);
}

// Tests getting tool frame name.
TEST_F(RobotManagerImplTest, GetToolFrame)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  EXPECT_EQ(manager.GetToolFrame(), "tool_frame");
}

// Tests getting base frame name.
TEST_F(RobotManagerImplTest, GetBaseFrame)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  // Base frame should be "base_link" from the URDF.
  EXPECT_EQ(manager.GetBaseFrame(), "base_link");
}

// Tests joint state subscription and retrieval.
TEST_F(RobotManagerImplTest, JointStateSubscription)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  // Initially no valid joint state.
  EXPECT_FALSE(manager.HasValidJointState());

  // Provide a joint state update (subscription is owned by CumotionPlanner, not RobotManagerImpl).
  std::vector<std::string> names = {"joint1", "joint2"};
  std::vector<double> positions = {0.1, 0.2};
  std::vector<double> velocities = {0.0, 0.0};
  manager.UpdateJointState(names, positions, velocities);

  // Now should have valid joint state.
  EXPECT_TRUE(manager.HasValidJointState());

  Eigen::VectorXd state;
  EXPECT_TRUE(manager.GetCurrentJointState(state));
  EXPECT_EQ(state.size(), 2);
  EXPECT_NEAR(state[0], 0.1, 1e-6);
  EXPECT_NEAR(state[1], 0.2, 1e-6);
}

// Tests that GetCurrentJointState returns false when no joint state received.
TEST_F(RobotManagerImplTest, GetCurrentJointStateNoData)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  Eigen::VectorXd state;
  EXPECT_FALSE(manager.GetCurrentJointState(state));
}

// Tests using default tool frame when none specified.
TEST_F(RobotManagerImplTest, DefaultToolFrame)
{
  RobotManagerImpl manager(
    GetValidUrdfPath(),
    GetValidXrdfPath(),
    test_node_->get_logger());

  // Should use the first tool frame from robot description.
  EXPECT_EQ(manager.GetToolFrame(), "tool_frame");
}

// A full robot state includes non-controlled joints. Passing all positions to cuMotion
// produces the wrong c-space dimension, even though the arm start state is valid.
TEST_F(RobotManagerImplTest, StartStateFiltersExtraJointsAndUsesCSpaceOrder)
{
  RobotManagerImpl manager(GetValidUrdfPath(), GetValidXrdfPath(), test_node_->get_logger());
  moveit_msgs::msg::RobotState state;
  state.joint_state.name = {"finger_joint1", "joint2", "joint1", "finger_joint2"};
  state.joint_state.position = {0.04, -0.4, 0.3, 0.04};

  Eigen::VectorXd position;
  std::string error;
  ASSERT_TRUE(manager.ResolveMoveItStartCSpacePosition(state, position, error)) << error;
  ASSERT_EQ(position.size(), manager.GetNumJoints());
  EXPECT_DOUBLE_EQ(position(0), 0.3);
  EXPECT_DOUBLE_EQ(position(1), -0.4);
}

// Joint constraints can arrive in a different order than the XRDF. Copying positions
// without consulting names sends the robot toward a different joint configuration.
TEST_F(RobotManagerImplTest, JointGoalUsesCSpaceOrder)
{
  RobotManagerImpl manager(GetValidUrdfPath(), GetValidXrdfPath(), test_node_->get_logger());
  std::vector<moveit_msgs::msg::JointConstraint> constraints(2);
  constraints[0].joint_name = "joint2";
  constraints[0].position = -0.4;
  constraints[1].joint_name = "joint1";
  constraints[1].position = 0.3;

  Eigen::VectorXd position;
  std::string error;
  ASSERT_TRUE(manager.ResolveMoveItGoalCSpacePosition(constraints, position, error)) << error;
  ASSERT_EQ(position.size(), manager.GetNumJoints());
  EXPECT_DOUBLE_EQ(position(0), 0.3);
  EXPECT_DOUBLE_EQ(position(1), -0.4);
}

TEST_F(RobotManagerImplTest, StartDiffRejectsNamedJointWithoutPosition)
{
  RobotManagerImpl manager(GetValidUrdfPath(), GetValidXrdfPath(), test_node_->get_logger());
  manager.UpdateJointState({"joint1", "joint2"}, {0.3, -0.4}, {});
  moveit_msgs::msg::RobotState state;
  state.is_diff = true;
  state.joint_state.name = {"joint1", "joint2"};
  state.joint_state.position = {0.8};

  Eigen::VectorXd position;
  std::string error;
  EXPECT_FALSE(manager.ResolveMoveItStartCSpacePosition(state, position, error));
  EXPECT_NE(error.find("joint2"), std::string::npos);
}

TEST_F(RobotManagerImplTest, StartStateRejectsPartialAndEmptyRequests)
{
  RobotManagerImpl manager(GetValidUrdfPath(), GetValidXrdfPath(), test_node_->get_logger());
  manager.UpdateJointState({"joint1", "joint2"}, {0.3, -0.4}, {});

  for (const bool is_diff : {false, true}) {
    moveit_msgs::msg::RobotState state;
    state.is_diff = is_diff;
    state.joint_state.name = {"joint1"};
    state.joint_state.position = {0.8};
    Eigen::VectorXd position;
    std::string error;
    EXPECT_FALSE(manager.ResolveMoveItStartCSpacePosition(state, position, error));
    EXPECT_NE(error.find("joint2"), std::string::npos);

    state.joint_state.name.clear();
    state.joint_state.position.clear();
    EXPECT_FALSE(manager.ResolveMoveItStartCSpacePosition(state, position, error));
    EXPECT_FALSE(error.empty());

    state.joint_state.name = {"joint2", "joint1"};
    state.joint_state.position = {-0.4, 0.8};
    ASSERT_TRUE(manager.ResolveMoveItStartCSpacePosition(state, position, error)) << error;
    EXPECT_DOUBLE_EQ(position(0), 0.8);
    EXPECT_DOUBLE_EQ(position(1), -0.4);
  }
}

}  // namespace cumotion
}  // namespace isaac_ros
}  // namespace nvidia

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
