# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
import threading
import time

from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose as CuPose
from curobo.types.state import JointState as CuJointState
from curobo.util_file import get_robot_configs_path
from curobo.util_file import join_path
from curobo.util_file import load_yaml
from curobo.wrap.model.robot_segmenter import RobotSegmenter
from cv_bridge import CvBridge
from isaac_ros_cumotion.util import get_spheres_marker
from isaac_ros_cumotion.xrdf_utils import convert_xrdf_to_curobo
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import torch
from visualization_msgs.msg import MarkerArray


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CumotionRobotSegmenter(Node):
    """This node filters out depth pixels assosiated with a robot body using a mask."""

    def __init__(self):
        super().__init__('cumotion_robot_segmentation')
        self.declare_parameter('robot', 'ur5e.yml')
        self.declare_parameter('cuda_device', 0)
        self.declare_parameter('distance_threshold', 0.1)
        self.declare_parameter('time_sync_slop', 0.1)
        self.declare_parameter('tf_lookup_duration', 5.0)

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('debug_robot_topic', '/cumotion/robot_segmenter/robot_spheres')

        self.declare_parameter('depth_image_topics', ['/cumotion/depth_1/image_raw'])
        self.declare_parameter('depth_camera_infos', ['/cumotion/depth_1/camera_info'])
        self.declare_parameter('robot_mask_publish_topics', ['/cumotion/depth_1/robot_mask'])
        self.declare_parameter('world_depth_publish_topics', ['/cumotion/depth_1/world_depth'])

        self.declare_parameter('log_debug', False)
        self.declare_parameter('urdf_path', rclpy.Parameter.Type.STRING)

        try:
            self.__urdf_path = self.get_parameter('urdf_path')
            self.__urdf_path = self.__urdf_path.get_parameter_value().string_value
        except rclpy.exceptions.ParameterUninitializedException:
            self.__urdf_path = None

        distance_threshold = (
            self.get_parameter('distance_threshold').get_parameter_value().double_value)
        time_sync_slop = self.get_parameter('time_sync_slop').get_parameter_value().double_value
        self._tf_lookup_duration = (
            self.get_parameter('tf_lookup_duration').get_parameter_value().double_value
        )
        joint_states_topic = (
            self.get_parameter('joint_states_topic').get_parameter_value().string_value)
        debug_robot_topic = (
            self.get_parameter('debug_robot_topic').get_parameter_value().string_value)
        depth_image_topics = (
            self.get_parameter('depth_image_topics').get_parameter_value().string_array_value)
        depth_camera_infos = (
            self.get_parameter('depth_camera_infos').get_parameter_value().string_array_value)
        publish_mask_topics = (
            self.get_parameter(
                'robot_mask_publish_topics').get_parameter_value().string_array_value)
        world_depth_topics = (
            self.get_parameter(
                'world_depth_publish_topics').get_parameter_value().string_array_value)

        self._log_debug = self.get_parameter('log_debug').get_parameter_value().bool_value
        num_cameras = len(depth_image_topics)
        self._num_cameras = num_cameras

        if len(depth_camera_infos) != num_cameras:
            self.get_logger().error(
                'Number of topics in depth_camera_infos does not match depth_image_topics')
        if len(publish_mask_topics) != num_cameras:
            self.get_logger().error(
                'Number of topics in publish_mask_topics does not match depth_image_topics')
        if len(world_depth_topics) != num_cameras:
            self.get_logger().error(
                'Number of topics in world_depth_topics does not match depth_image_topics')

        cuda_device_id = self.get_parameter('cuda_device').get_parameter_value().integer_value

        self._tensor_args = TensorDeviceType(device=torch.device('cuda', cuda_device_id))

        # Create subscribers:
        subscribers = [Subscriber(self, Image, topic) for topic in depth_image_topics]
        subscribers.append(Subscriber(self, JointState, joint_states_topic))
        # Subscribe to topics with sync:
        self.approx_time_sync = ApproximateTimeSynchronizer(
            tuple(subscribers), queue_size=100, slop=time_sync_slop)
        self.approx_time_sync.registerCallback(self.process_depth_and_joint_state)

        self.info_subscribers = []

        for idx in range(num_cameras):
            self.info_subscribers.append(
                self.create_subscription(
                    CameraInfo, depth_camera_infos[idx],
                    lambda msg, index=idx: self.camera_info_cb(msg, index), 10)
            )

        self.mask_publishers = [
            self.create_publisher(Image, topic, 10) for topic in publish_mask_topics]
        self.segmented_publishers = [
            self.create_publisher(Image, topic, 10) for topic in world_depth_topics]

        self.debug_robot_publisher = self.create_publisher(MarkerArray, debug_robot_topic, 10)

        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create a depth mask publisher:
        robot_file = self.get_parameter('robot').get_parameter_value().string_value
        self.br = CvBridge()

        # Create buffers to store data:
        self._depth_buffers = None
        self._depth_intrinsics = [None for x in range(num_cameras)]
        self._robot_pose_camera = [None for x in range(num_cameras)]
        self._depth_encoding = None

        self._js_buffer = None
        self._timestamp = None
        self._camera_headers = []
        self.lock = threading.Lock()
        self.timer = self.create_timer(0.01, self.on_timer)

        robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))
        if robot_file.lower().endswith('.xrdf'):
            if self.__urdf_path is None:
                self.get_logger().fatal('urdf_path is required to load robot from .xrdf')
                raise SystemExit
            robot_dict = convert_xrdf_to_curobo(self.__urdf_path, robot_dict, self.get_logger())

        if self.__urdf_path is not None:
            robot_dict['robot_cfg']['kinematics']['urdf_path'] = self.__urdf_path

        self._cumotion_segmenter = RobotSegmenter.from_robot_file(
            robot_dict, distance_threshold=distance_threshold)

        self._cumotion_base_frame = self._cumotion_segmenter.base_link
        self._robot_pose_cameras = None
        self.get_logger().info(f'Node initialized with {self._num_cameras} cameras')

    def process_depth_and_joint_state(self, *msgs):
        self._depth_buffers = []
        self._depth_encoding = []
        self._camera_headers = []
        for msg in msgs:
            if (isinstance(msg, Image)):
                img = self.br.imgmsg_to_cv2(msg)
                if msg.encoding == '32FC1':
                    img = 1000.0 * img
                self._depth_buffers.append(img)
                self._camera_headers.append(msg.header)
                self._depth_encoding.append(msg.encoding)
            if (isinstance(msg, JointState)):
                self._js_buffer = {'joint_names': msg.name, 'position': msg.position}
                self._timestamp = msg.header.stamp

    def camera_info_cb(self, msg, idx):
        self._depth_intrinsics[idx] = msg.k

    def publish_robot_spheres(self, traj: CuJointState):
        kin_state = self._cumotion_segmenter.robot_world.get_kinematics(traj.position)
        spheres = kin_state.link_spheres_tensor.cpu().numpy()
        current_time = self.get_clock().now().to_msg()

        m_arr = get_spheres_marker(
            spheres[0],
            self._cumotion_base_frame,
            current_time,
            rgb=[0.0, 1.0, 0.0, 1.0],
        )

        self.debug_robot_publisher.publish(m_arr)

    def is_subscribed(self) -> bool:
        count_mask = max(
            [mask_pub.get_subscription_count() for mask_pub in self.mask_publishers]
            + [seg_pub.get_subscription_count() for seg_pub in self.segmented_publishers]
        )
        if count_mask > 0:
            return True
        return False

    def publish_images(self, depth_mask, segmented_depth, camera_header, idx: int):

        if self.mask_publishers[idx].get_subscription_count() > 0:
            depth_mask = depth_mask[idx]
            msg = self.br.cv2_to_imgmsg(depth_mask, 'mono8')
            msg.header = camera_header[idx]
            self.mask_publishers[idx].publish(msg)

        if self.segmented_publishers[idx].get_subscription_count() > 0:
            segmented_depth = segmented_depth[idx]
            if self._depth_encoding[idx] == '16UC1':
                segmented_depth = segmented_depth.astype(np.uint16)
            elif self._depth_encoding[idx] == '32FC1':
                segmented_depth = segmented_depth / 1000.0
            msg = self.br.cv2_to_imgmsg(segmented_depth, self._depth_encoding[idx])
            msg.header = camera_header[idx]
            self.segmented_publishers[idx].publish(msg)

    def on_timer(self):
        computation_time = -1.0
        node_time = -1.0

        if not self.is_subscribed():
            return

        if ((not all(isinstance(intrinsic, np.ndarray) for intrinsic in self._depth_intrinsics))
                or (len(self._camera_headers) == 0) or (self._timestamp is None)):
            return

        timestamp = self._timestamp

        # Read camera transforms
        if self._robot_pose_cameras is None:
            self.get_logger().info('Reading TF from cameras')

            with self.lock:
                camera_headers = deepcopy(self._camera_headers)

            for i in range(self._num_cameras):
                if self._robot_pose_camera[i] is None:
                    try:
                        t = self.tf_buffer.lookup_transform(
                            self._cumotion_base_frame,
                            camera_headers[i].frame_id,
                            timestamp,
                            rclpy.duration.Duration(seconds=self._tf_lookup_duration),
                        )
                        self._robot_pose_camera[i] = CuPose.from_list(
                            [
                                t.transform.translation.x,
                                t.transform.translation.y,
                                t.transform.translation.z,
                                t.transform.rotation.w,
                                t.transform.rotation.x,
                                t.transform.rotation.y,
                                t.transform.rotation.z,
                            ]
                        )
                    except TransformException as ex:
                        self.get_logger().debug(f'Could not transform {camera_headers[i].frame_id} \
                            to { self._cumotion_base_frame}: {ex}')
                        continue
            if None not in self._robot_pose_camera:
                self._robot_pose_cameras = CuPose.cat(self._robot_pose_camera)
                self.get_logger().info('Received TF from cameras to robot')

        # Check if all camera transforms have been received
        if self._robot_pose_cameras is None:
            return

        with self.lock:
            timestamp = self._timestamp
            depth_image = np.copy(np.stack((self._depth_buffers)))
            intrinsics = np.copy(np.stack(self._depth_intrinsics))
            js = np.copy(self._js_buffer['position'])
            j_names = deepcopy(self._js_buffer['joint_names'])
            camera_headers = deepcopy(self._camera_headers)
            self._timestamp = None
            self._camera_headers = []
        start_node_time = time.time()

        depth_image = self._tensor_args.to_device(depth_image.astype(np.float32))
        depth_image = depth_image.view(
            self._num_cameras, depth_image.shape[-2], depth_image.shape[-1])

        if not self._cumotion_segmenter.ready:
            intrinsics = self._tensor_args.to_device(intrinsics).view(self._num_cameras, 3, 3)
            cam_obs = CameraObservation(depth_image=depth_image, intrinsics=intrinsics)
            self._cumotion_segmenter.update_camera_projection(cam_obs)
            self.get_logger().info('Updated Projection Matrices')
        cam_obs = CameraObservation(depth_image=depth_image, pose=self._robot_pose_cameras)
        q = CuJointState.from_numpy(
            position=js, joint_names=j_names, tensor_args=self._tensor_args).unsqueeze(0)
        q = self._cumotion_segmenter.robot_world.get_active_js(q)

        start_segmentation_time = time.time()
        depth_mask, segmented_depth = self._cumotion_segmenter.get_robot_mask_from_active_js(
            cam_obs, q)
        if self._log_debug:
            torch.cuda.synchronize()
            computation_time = time.time() - start_segmentation_time
        depth_mask = depth_mask.cpu().numpy().astype(np.uint8) * 255
        segmented_depth = segmented_depth.cpu().numpy()

        for x in range(depth_mask.shape[0]):
            self.publish_images(depth_mask, segmented_depth, camera_headers, x)

        if self.debug_robot_publisher.get_subscription_count() > 0:
            self.publish_robot_spheres(q)
        if self._log_debug:
            node_time = time.time() - start_node_time
            self.get_logger().info(f'Node Time(ms), Computation Time(ms): {node_time * 1000.0},\
                                    {computation_time * 1000.0}')


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    cumotion_segmenter = CumotionRobotSegmenter()

    # Spin the node so the callback function is called.
    rclpy.spin(cumotion_segmenter)

    # Destroy the node explicitly
    cumotion_segmenter.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()
