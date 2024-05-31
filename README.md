# Isaac ROS cuMotion

NVIDIA accelerated packages for arm motion planning and control

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_cumotion/cumotion_ur10_demo.gif/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_cumotion/cumotion_ur10_demo.gif/" width="600px"/></a></div>

## Overview

[Isaac ROS cuMotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion) provides CUDA-accelerated manipulation
capabilities for robots in ROS 2, enabling faster processing speeds and real-time performance
that are crucial to complex robotics tasks such as motion planning.
It provides two main capabilities, motion generation for robot
arms via an integration of cuMotion into MoveIt 2 and segmentation of robot from depth streams
using cuMotion’s kinematics and geometry processing functions to accurately identify and filter robot parts.
This allows one to reconstruct obstacles in the environment without spurious contributions from the robot itself.

The key advantages of using Isaac ROS cuMotion are:

* Increased Efficiency: CUDA acceleration significantly speeds up processing times,
  allowing for complex computation, such as collision avoidance, occurring at real-time.
* Enhanced Precision: Accurate motion planning and segmentation allow for better
  performance in tasks requiring fine manipulation and detailed environmental interaction.
* Improved Flexibility: Modular design allows easy integration with existing ROS 2 setups,
  such as configurations using MoveIt 2, enabling customization and scalability using familiar
  tooling.

The Isaac ROS cuMotion repository currently contains the following packages:

`isaac_ros_cumotion`:
: This package contains the cuMotion planner node and the robot segmentation node.

`isaac_ros_cumotion_examples`:
: This package contains various examples demonstrating use of cuMotion with MoveIt.

`isaac_ros_cumotion_moveit`:
: This package provides a plugin for MoveIt 2 that exposes cuMotion as an external planner, leveraging `isaac_ros_cumotion`.

Isaac ROS cuMotion is also featured as part of [Isaac Manipulator](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_manipulator/index.html).

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_cumotion`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html)
  * [Motion Generation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html#motion-generation)
  * [Robot Segmentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html#robot-segmentation)
* [`isaac_ros_cumotion_moveit`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#troubleshooting)
* [`isaac_ros_esdf_visualizer`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_esdf_visualizer/index.html)
  * [Overview](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_esdf_visualizer/index.html#overview)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_esdf_visualizer/index.html#api)
* [`isaac_ros_moveit_goal_setter`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_moveit_goal_setter/index.html)
  * [Overview](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_moveit_goal_setter/index.html#overview)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_moveit_goal_setter/index.html#api)
* [`isaac_ros_moveit_goal_setter_interfaces`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_moveit_goal_setter_interfaces/index.html)

## Latest

Update 2024-05-30: Initial release
