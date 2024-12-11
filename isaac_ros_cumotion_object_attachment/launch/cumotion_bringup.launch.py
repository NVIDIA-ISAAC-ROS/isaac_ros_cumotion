# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import SetParameter


def generate_launch_description():

    # The 'robot' argument can accept:
    # - XRDF filename
    # - YAML filename
    # - Absolute paths for XRDF or YAML files

    # Default URDF file path (currently absolute, should be updated as needed).
    default_urdf_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_robot_description'),
        'urdf',
        'ur5e_robotiq_2f_140.urdf'
    )

    # Declare launch arguments with full paths
    launch_args = [
        DeclareLaunchArgument(
            'robot',
            default_value='ur5e_robotiq_2f_140.xrdf',
            description='Robot file (XRDF or YAML)'
        ),
        DeclareLaunchArgument(
            'urdf_file_path',
            default_value=default_urdf_file_path,
            description='Full path to the URDF file'
        ),
        DeclareLaunchArgument(
            'yml_file_path',
            default_value='',
            description='Path to the YAML file containing robot configurations'
        ),
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value='/joint_states',
            description='Joint states topic'
        ),
        DeclareLaunchArgument(
            'depth_camera_info_topics',
            default_value="['/camera_1/color/camera_info']",
            description='Depth camera info topic'
        ),
        DeclareLaunchArgument(
            'depth_image_topics',
            default_value="['/camera_1/aligned_depth_to_color/image_raw']",
            description='Depth image topic for robot segmenter'
        ),
        DeclareLaunchArgument(
            'object_link_name',
            default_value='attached_object',
            description='Object link name for object attachment'
        ),
        DeclareLaunchArgument(
            'search_radius',
            default_value='0.2',
            description='Search radius for object attachment'
        ),
        DeclareLaunchArgument(
            'surface_sphere_radius',
            default_value='0.01',
            description='Radius for object surface collision spheres'
        ),
        DeclareLaunchArgument(
            'update_link_sphere_server_segmenter',
            default_value='segmenter_attach_object',
            description='Update link sphere server for robot segmenter'
        ),
        DeclareLaunchArgument(
            'update_link_sphere_server_planner',
            default_value='planner_attach_object',
            description='Update link sphere server for cumotion planner'
        ),
        DeclareLaunchArgument(
            'clustering_bypass',
            default_value='False',
            description='Whether to bypass clustering'
        ),
        DeclareLaunchArgument(
            'update_esdf_on_request',
            default_value='False',
            description='Whether object attachment should request an updated ESDF from nvblox '
                        'as part of the service call'
        ),
        DeclareLaunchArgument(
            'action_names',
            default_value="['segmenter_attach_object', 'planner_attach_object']",
            description='List of action names for the object attachment'
        ),
        DeclareLaunchArgument(
            'time_sync_slop',
            default_value='0.1',
            description='Time synchronization slop'
        ),
        DeclareLaunchArgument(
            'distance_threshold',
            default_value='0.02',
            description='Distance threshold for segmentation'
        ),
        DeclareLaunchArgument(
            'clustering_hdbscan_min_samples',
            default_value='20',
            description='HDBSCAN min samples for clustering'
        ),
        DeclareLaunchArgument(
            'clustering_hdbscan_min_cluster_size',
            default_value='30',
            description='HDBSCAN min cluster size for clustering'
        ),
        DeclareLaunchArgument(
            'clustering_hdbscan_cluster_selection_epsilon',
            default_value='0.5',
            description='HDBSCAN cluster selection epsilon'
        ),
        DeclareLaunchArgument(
            'clustering_num_top_clusters_to_select',
            default_value='3',
            description='Number of top clusters to select'
        ),
        DeclareLaunchArgument(
            'clustering_group_clusters',
            default_value='False',
            description='Whether to group clusters'
        ),
        DeclareLaunchArgument(
            'clustering_min_points',
            default_value='100',
            description='Minimum points for clustering'
        ),
    ]

    # LaunchConfiguration objects to pass to the launch files
    robot = LaunchConfiguration('robot')
    urdf_path = LaunchConfiguration('urdf_file_path')
    yml_file_path = LaunchConfiguration('yml_file_path')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    depth_camera_info_topics = LaunchConfiguration('depth_camera_info_topics')
    depth_image_topics = LaunchConfiguration('depth_image_topics')
    object_link_name = LaunchConfiguration('object_link_name')
    search_radius = LaunchConfiguration('search_radius')
    surface_sphere_radius = LaunchConfiguration('surface_sphere_radius')
    update_esdf_on_request = LaunchConfiguration('update_esdf_on_request')
    update_link_sphere_server_segmenter = LaunchConfiguration(
        'update_link_sphere_server_segmenter')
    update_link_sphere_server_planner = LaunchConfiguration(
        'update_link_sphere_server_planner')
    clustering_bypass = LaunchConfiguration('clustering_bypass')
    action_names = LaunchConfiguration('action_names')
    time_sync_slop = LaunchConfiguration('time_sync_slop')
    distance_threshold = LaunchConfiguration('distance_threshold')
    clustering_hdbscan_min_samples = LaunchConfiguration(
        'clustering_hdbscan_min_samples')
    clustering_hdbscan_min_cluster_size = LaunchConfiguration(
        'clustering_hdbscan_min_cluster_size')
    clustering_hdbscan_cluster_selection_epsilon = LaunchConfiguration(
        'clustering_hdbscan_cluster_selection_epsilon')
    clustering_num_top_clusters_to_select = LaunchConfiguration(
        'clustering_num_top_clusters_to_select')
    clustering_group_clusters = LaunchConfiguration(
        'clustering_group_clusters')
    clustering_min_points = LaunchConfiguration('clustering_min_points')

    # Shared world depth topic as a string array
    world_depth_topic = "['/cumotion/camera_1/world_depth']"

    # Paths to the launch files
    cumotion_launch_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion'),
        'launch',
        'isaac_ros_cumotion.launch.py')

    robot_segmenter_launch_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion'),
        'launch',
        'robot_segmentation.launch.py')

    object_attachment_launch_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_object_attachment'),
        'launch',
        'object_attachment.launch.py')

    # Include the launch files with updated arguments
    cumotion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cumotion_launch_path),
        launch_arguments={
            'cumotion_planner.robot': robot,
            'cumotion_planner.urdf_path': urdf_path,
            'cumotion_planner.yml_file_path': yml_file_path,
            'cumotion_planner.update_link_sphere_server':
                update_link_sphere_server_planner,
        }.items()
    )

    robot_segmenter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_segmenter_launch_path),
        launch_arguments={
            'robot_segmenter.robot': robot,
            'robot_segmenter.urdf_path': urdf_path,
            'cumotion_planner.yml_file_path': yml_file_path,
            'robot_segmenter.depth_image_topics': depth_image_topics,
            'robot_segmenter.depth_camera_infos': depth_camera_info_topics,
            'robot_segmenter.joint_states_topic': joint_states_topic,
            'robot_segmenter.time_sync_slop': time_sync_slop,
            'robot_segmenter.distance_threshold': distance_threshold,
            'robot_segmenter.update_link_sphere_server':
                update_link_sphere_server_segmenter,
            'robot_segmenter.world_depth_publish_topics': world_depth_topic,
            'robot_segmenter.depth_qos': 'SENSOR_DATA',
            'robot_segmenter.depth_info_qos': 'SENSOR_DATA',
            'robot_segmenter.mask_qos': 'SENSOR_DATA',
            'robot_segmenter.world_depth_qos': 'SENSOR_DATA',

        }.items()
    )

    object_attachment_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(object_attachment_launch_path),
        launch_arguments={
            'object_attachment.robot': robot,
            'object_attachment.urdf_path': urdf_path,
            'object_attachment.time_sync_slop': time_sync_slop,
            'object_attachment.joint_states_topic': joint_states_topic,
            'object_attachment.depth_image_topics': world_depth_topic,
            'object_attachment.depth_camera_infos': depth_camera_info_topics,
            'object_attachment.object_link_name': object_link_name,
            'object_attachment.action_names': action_names,
            'object_attachment.search_radius': search_radius,
            'object_attachment.surface_sphere_radius': surface_sphere_radius,
            'object_attachment.update_esdf_on_request': update_esdf_on_request,
            'object_attachment.clustering_bypass_clustering':
                clustering_bypass,
            'object_attachment.clustering_hdbscan_min_samples':
                clustering_hdbscan_min_samples,
            'object_attachment.clustering_hdbscan_min_cluster_size':
                clustering_hdbscan_min_cluster_size,
            'object_attachment.clustering_hdbscan_cluster_selection_epsilon':
                clustering_hdbscan_cluster_selection_epsilon,
            'object_attachment.clustering_num_top_clusters_to_select':
                clustering_num_top_clusters_to_select,
            'object_attachment.clustering_group_clusters':
                clustering_group_clusters,
            'object_attachment.clustering_min_points':
                clustering_min_points,
            'object_attachment.depth_qos': 'SENSOR_DATA',
            'object_attachment.depth_info_qos': 'SENSOR_DATA',
        }.items()
    )
    # Add sim time parameter as we always run this launch file with a ROSbag which needs this
    # parameter so that object attachment can filter depth images correctly.
    use_sim_time_param = [SetParameter(name='use_sim_time', value=True)]
    # Return the LaunchDescription with all included launch files
    return LaunchDescription(launch_args + use_sim_time_param + [
        cumotion_launch,
        robot_segmenter_launch,
        object_attachment_launch
    ])
