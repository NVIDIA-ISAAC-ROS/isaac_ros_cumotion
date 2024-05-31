# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from curobo.geom.types import Sphere
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def get_spheres_marker(
        robot_spheres, base_frame: str, time, rgb=[0.1, 0.1, 0.1, 0.5], start_idx: int = 0):
    m_arr = MarkerArray()

    for i in range(len(robot_spheres)):
        r_s = Sphere(
            name='sphere',
            radius=robot_spheres[i, -1],
            pose=robot_spheres[i, :3].tolist() + [1, 0, 0, 0],
        )
        # print(rgb[i])
        m = get_marker_sphere(r_s, base_frame, time, start_idx + i, rgb)
        m_arr.markers.append(m)
    return m_arr


def get_marker_sphere(sphere: Sphere, base_frame: str, time, idx=0, rgb=[0.4, 0.4, 0.8, 1.0]):
    marker = Marker()
    marker.header.frame_id = base_frame
    marker.header.stamp = time
    marker.id = idx
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.scale.x = sphere.radius * 2
    marker.scale.y = sphere.radius * 2
    marker.scale.z = sphere.radius * 2
    marker.color.r = rgb[0]
    marker.color.g = rgb[1]
    marker.color.b = rgb[2]
    marker.color.a = rgb[3]

    # pose:
    marker.pose.position.x = sphere.position[0]
    marker.pose.position.y = sphere.position[1]
    marker.pose.position.z = sphere.position[2]
    marker.pose.orientation.w = 1.0
    return marker
