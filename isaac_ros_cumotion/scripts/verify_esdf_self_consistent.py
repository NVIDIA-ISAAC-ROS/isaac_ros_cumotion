#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""
Verify nvblox returns a self-consistent ESDF (no sentinel-like voxels).

This is a diagnostic for the ESDF contract consumed by cuMotion: once nvblox is
configured with ``unobserved_esdf_policy: free``, the integrator should write
valid distances upstream instead of the unobserved sentinel value.

This script calls ``/nvblox_node/get_esdf_and_gradient`` directly and reports
how many voxels in the response carry a sentinel-like value.

Usage::

    ros2 run isaac_ros_cumotion verify_esdf_self_consistent.py
    # or, with options:
    ros2 run isaac_ros_cumotion verify_esdf_self_consistent.py \\
        --service /nvblox_node/get_esdf_and_gradient \\
        --frame-id base_link \\
        --num-queries 5 \\
        --interval-sec 1.0 \\
        --use-aabb --aabb-size 2.0 2.0 2.0

Exit code is 0 when every queried response is sentinel-free, 1 when any
sentinel-like voxels were observed (or no successful response was received).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List

from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
import rclpy
from rclpy.node import Node
from rclpy.task import Future


# nvblox's default ESDF response value for unobserved voxels is -1000.0.
DEFAULT_UNOBSERVED_ESDF_VALUE = -1000.0
DEFAULT_SENTINEL_TOLERANCE_M = 1e-3


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        '--service',
        default='/nvblox_node/get_esdf_and_gradient',
        help='Fully qualified ESDF service name to call.',
    )
    parser.add_argument(
        '--frame-id',
        default='base_link',
        help='Frame the (optional) AABB is expressed in. Matches cumotion default.',
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=3,
        help='Number of consecutive service calls to make.',
    )
    parser.add_argument(
        '--interval-sec',
        type=float,
        default=1.0,
        help='Seconds to wait between successive queries.',
    )
    parser.add_argument(
        '--service-timeout-sec',
        type=float,
        default=15.0,
        help='Per-call timeout when waiting for the service response.',
    )
    parser.add_argument(
        '--service-discovery-timeout-sec',
        type=float,
        default=10.0,
        help='How long to wait for the service to appear before giving up.',
    )
    parser.add_argument(
        '--unobserved-value',
        type=float,
        default=DEFAULT_UNOBSERVED_ESDF_VALUE,
        help=(
            "Value nvblox uses for unobserved voxels. Defaults to nvblox's "
            'esdf_and_gradients_unobserved_value default.'
        ),
    )
    parser.add_argument(
        '--sentinel-tolerance-m',
        type=float,
        default=DEFAULT_SENTINEL_TOLERANCE_M,
        help='Absolute tolerance used when matching --unobserved-value.',
    )
    parser.add_argument(
        '--max-expected-distance-m',
        type=float,
        default=None,
        help=(
            'Maximum plausible absolute ESDF distance. If omitted, this is inferred '
            'from the AABB size or response grid size.'
        ),
    )
    parser.add_argument(
        '--update-esdf',
        action='store_true',
        default=True,
        help='Tell nvblox to update the ESDF before responding (default: true).',
    )
    parser.add_argument(
        '--no-update-esdf',
        action='store_false',
        dest='update_esdf',
        help='Do not request an ESDF update on each call.',
    )
    parser.add_argument(
        '--use-aabb',
        action='store_true',
        help=(
            'Set request.use_aabb=True. cuMotion sends AABB-bounded requests after '
            'startup workspace-bounds caching; the default no-AABB request exercises '
            'the initial unbounded request shape.'
        ),
    )
    parser.add_argument(
        '--aabb-min',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=[-1.0, -1.0, -0.2],
        help='Minimum corner of the AABB (only used with --use-aabb).',
    )
    parser.add_argument(
        '--aabb-size',
        type=float,
        nargs=3,
        metavar=('SX', 'SY', 'SZ'),
        default=[2.0, 2.0, 1.5],
        help='Size of the AABB (only used with --use-aabb).',
    )
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> EsdfAndGradients.Request:
    """Construct an ESDF request shaped like the one cumotion sends."""
    request = EsdfAndGradients.Request()
    request.update_esdf = bool(args.update_esdf)
    request.visualize_esdf = True
    request.frame_id = args.frame_id
    request.use_aabb = bool(args.use_aabb)
    if args.use_aabb:
        request.aabb_min_m = Point(
            x=float(args.aabb_min[0]),
            y=float(args.aabb_min[1]),
            z=float(args.aabb_min[2]),
        )
        request.aabb_size_m = Vector3(
            x=float(args.aabb_size[0]),
            y=float(args.aabb_size[1]),
            z=float(args.aabb_size[2]),
        )
    return request


def infer_distance_bound_m(
    args: argparse.Namespace,
    response: EsdfAndGradients.Response,
) -> float | None:
    """Infer a plausible maximum ESDF distance for sentinel-like value detection."""
    if args.max_expected_distance_m is not None:
        return float(args.max_expected_distance_m)
    if args.use_aabb:
        return max(float(side_length_m) for side_length_m in args.aabb_size)

    dims = [d.size for d in response.esdf_and_gradients.layout.dim[:3]]
    if len(dims) == 3 and response.voxel_size_m > 0.0:
        return max(dims) * float(response.voxel_size_m)
    return None


def is_sentinel_like(
    value: float,
    unobserved_value: float,
    tolerance_m: float,
    distance_bound_m: float | None,
    voxel_size_m: float,
) -> bool:
    """Return whether an ESDF value is outside the valid signed-distance range."""
    if not math.isfinite(value):
        return True
    if math.isclose(value, unobserved_value, rel_tol=0.0, abs_tol=tolerance_m):
        return True
    if distance_bound_m is not None:
        margin_m = max(float(voxel_size_m), tolerance_m)
        return abs(value) > distance_bound_m + margin_m
    return False


def count_sentinels(
    values: List[float],
    response: EsdfAndGradients.Response,
    args: argparse.Namespace,
) -> int:
    distance_bound_m = infer_distance_bound_m(args, response)
    return sum(
        1
        for value in values
        if is_sentinel_like(
            value,
            args.unobserved_value,
            args.sentinel_tolerance_m,
            distance_bound_m,
            response.voxel_size_m,
        )
    )


def summarize_response(
    response: EsdfAndGradients.Response,
    sentinel_count: int,
    args: argparse.Namespace,
) -> str:
    arr = response.esdf_and_gradients
    dims = [d.size for d in arr.layout.dim] if arr.layout.dim else []
    total = len(arr.data)
    if total == 0:
        return 'empty grid'
    distance_bound_m = infer_distance_bound_m(args, response)
    finite = [
        v for v in arr.data
        if not is_sentinel_like(
            v,
            args.unobserved_value,
            args.sentinel_tolerance_m,
            distance_bound_m,
            response.voxel_size_m,
        )
    ]
    if finite:
        vmin = min(finite)
        vmax = max(finite)
    else:
        vmin = float('nan')
        vmax = float('nan')
    return (
        f'shape={dims} voxels={total} voxel_size={response.voxel_size_m:.3f}m '
        f'origin=({response.origin_m.x:.3f}, {response.origin_m.y:.3f}, '
        f'{response.origin_m.z:.3f}) sentinels={sentinel_count} '
        f'observed_range=[{vmin:.3f}, {vmax:.3f}] '
        f'distance_bound={distance_bound_m if distance_bound_m is not None else "unknown"}'
    )


class EsdfSelfConsistencyChecker(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('esdf_self_consistency_checker')
        self._args = args
        self._client = self.create_client(EsdfAndGradients, args.service)

    def wait_for_service(self) -> bool:
        timeout = self._args.service_discovery_timeout_sec
        self.get_logger().info(
            f"Waiting up to {timeout:.1f}s for service '{self._args.service}' ..."
        )
        if not self._client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service '{self._args.service}' not available; is nvblox running?"
            )
            return False
        return True

    def send_one(self) -> EsdfAndGradients.Response | None:
        request = build_request(self._args)
        future: Future = self._client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self._args.service_timeout_sec,
        )
        if not future.done():
            self.get_logger().error('ESDF service call timed out.')
            return None
        exception = future.exception()
        if exception is not None:
            self.get_logger().error(f'ESDF service call failed: {exception}')
            return None
        response = future.result()
        if response is None:
            self.get_logger().error('ESDF service call returned no response.')
            return None
        if not response.success:
            self.get_logger().error('ESDF service reported success=false.')
            return None
        return response


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    rclpy.init()
    try:
        node = EsdfSelfConsistencyChecker(args)
        if not node.wait_for_service():
            return 1

        total_sentinels = 0
        total_voxels = 0
        successful_queries = 0
        any_response_seen = False

        for i in range(args.num_queries):
            response = node.send_one()
            if response is None:
                node.get_logger().warn(f'Query {i + 1}/{args.num_queries} failed; retrying.')
                time.sleep(args.interval_sec)
                continue
            any_response_seen = True
            successful_queries += 1
            values = list(response.esdf_and_gradients.data)
            sentinel_count = count_sentinels(values, response, args)
            total_sentinels += sentinel_count
            total_voxels += len(response.esdf_and_gradients.data)
            line = summarize_response(response, sentinel_count, args)
            if sentinel_count == 0:
                node.get_logger().info(f'Query {i + 1}: clean -- {line}')
            else:
                node.get_logger().warn(
                    f'Query {i + 1}: SENTINEL-LIKE VALUES PRESENT -- {line}'
                )
            if i + 1 < args.num_queries:
                time.sleep(args.interval_sec)

        if not any_response_seen:
            node.get_logger().error(
                'No successful ESDF responses received. Cannot conclude self-consistency.'
            )
            return 1

        pct = (
            100.0 * total_sentinels / total_voxels if total_voxels else 0.0
        )
        node.get_logger().info(
            'Summary: '
            f'{successful_queries}/{args.num_queries} successful queries, '
            f'{total_sentinels}/{total_voxels} sentinel-like voxels '
            f'({pct:.4f}%).'
        )
        if total_sentinels == 0:
            node.get_logger().info(
                'Result: SELF-CONSISTENT. nvblox returned sentinel-free ESDF responses.'
            )
            return 0
        node.get_logger().warn(
            'Result: NOT SELF-CONSISTENT. nvblox is still emitting sentinel-like voxels; '
            'verify "unobserved_esdf_policy" is set to "free" in nvblox_manipulator_base.yaml '
            'and that the new nvblox build is loaded.'
        )
        return 1
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
