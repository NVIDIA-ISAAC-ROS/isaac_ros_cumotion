# Isaac ROS cuMotion

NVIDIA accelerated packages for arm motion planning and control

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_cumotion/cumotion_ur10_demo.gif/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.6/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_cumotion/cumotion_ur10_demo.gif/" width="600px"/></a></div>

## Overview

Isaac ROS cuMotion provides CUDA-accelerated manipulation
capabilities for robots in ROS 2. It integrates [cuMotion](https://nvidia-isaac-ros.github.io/concepts/manipulation/index.html#concept-cumotion) into MoveIt 2
to generate smooth, collision-free trajectories, and also exposes cuMotion inverse kinematics and
trajectory generation capabilities through ROS 2 actions/services. Support is provided for
segmenting and filtering the robot itself from the depth streams.

The key advantages of using Isaac ROS cuMotion are:

* **Improved cycle times:** cuMotion produces smooth, optimal-time trajectories in the
  presence of obstacles, generally reducing motion times compared to previous
  state-of-the-art planners.  In cluttered environments and other challenging scenarios,
  cuMotion can often produce a valid trajectory when other planners might fail altogether.
* **Improved planning times:** cuMotion takes advantage of CUDA acceleration to produce
  collision-free, optimal-time trajectories in a fraction of a second.
* **Avoidance of obstacles captured by depth cameras:** cuMotion optionally leverages
  [nvblox](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/index.html) to perform 3D reconstruction of an environment from one or more depth image
  streams.  The environment is represented as a signed distance field (SDF) for
  efficient obstacle-aware planning.  Support is provided for segmenting and filtering
  the robot itself from the depth streams.
* **Flexibility:** A modular design simplifies integration with existing ROS 2 workflows,
  especially those already using MoveIt 2.

> [!Warning]
> Before using or developing with cuMotion or other Isaac for Manipulation software,
> familiarize yourself with the associated safety information that is provided by your robot
> manufacturer.

> In addition, we provide the following best practices:

> 1. Familiarize yourself with the location of the emergency stop buttons, and be prepared to apply if necessary.
> 2. Before operation, ensure the working area is free of any persons or other potential hazards.
> 3. Before operation, alert all persons near the working area that robot operation is about to begin.
> 4. Before and during operation, be aware of any persons entering the vicinity around the working area,
>    and be prepared to give necessary warnings, instructions, or take other necessary actions.
> 5. Take extra caution when testing or deploying new features or code.

Examples are provided for three modes of operation:

* **Standalone MoveIt:** MoveIt’s RViz interface allows trajectories to be planned and visualized even without a physical
  robot or external simulator.
* **Simulated robot (Isaac Sim):** Trajectories may be planned and executed on a simulated robot in
  [Isaac Sim](https://developer.nvidia.com/isaac/sim), allowing convenient development and rapid iteration without
  use of physical hardware.  Simulated sensors enable testing of perception, for example, for the purpose of collision avoidance.
* **Physical robot:** For on-robot testing and final deployment, trajectories are planned and executed on a physical
  robot.

The Isaac ROS cuMotion repository currently contains the following packages:

`isaac_ros_cumotion`:
: This package contains the cuMotion planner node that provides the ROS 2 interfaces for
  inverse kinematics and motion generation using cuMotion.

`isaac_ros_cumotion_examples`:
: This package contains various examples demonstrating use of cuMotion with MoveIt.

`isaac_ros_cumotion_moveit`:
: This package provides a plugin for MoveIt 2 that exposes cuMotion as an external planner, leveraging `isaac_ros_cumotion`.

Isaac ROS cuMotion is also featured as part of [Isaac for Manipulation](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html).

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_cumotion`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html)
  * [Motion Generation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html#motion-generation)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html#api)
* [`isaac_ros_cumotion_moveit`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#try-more-examples)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#troubleshooting)
* [`isaac_ros_cumotion_object_attachment`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html)
  * [Object Attachment](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html#object-attachment)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html#api)
* [`isaac_ros_cumotion_robot_segmenter`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_robot_segmenter/index.html)
  * [Overview](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_robot_segmenter/index.html#overview)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_robot_segmenter/index.html#quickstart)
  * [Running Tests](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_robot_segmenter/index.html#running-tests)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_robot_segmenter/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_robot_segmenter/index.html#api)
* [`isaac_ros_goal_setter_interfaces`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_goal_setter_interfaces/index.html)

## Latest

Update 2026-08-18: Compatibility and integration updates for the Isaac ROS 4.6.0 release
