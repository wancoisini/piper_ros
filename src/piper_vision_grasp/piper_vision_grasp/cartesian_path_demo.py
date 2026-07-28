#!/usr/bin/env python3
"""MoveIt 多路点笛卡尔轨迹规划与执行示例。"""
import math
import threading
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from pymoveit2 import MoveIt2, MoveIt2State
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


class CartesianPathDemo(Node):
    def __init__(self):
        super().__init__('cartesian_path_demo')

        self.declare_parameter('group_name', 'piper_arm')
        self.declare_parameter('base_link', 'arm_base')
        self.declare_parameter('end_effector', 'link6')
        self.declare_parameter('start_joints', [0.0] * 6)
        self.declare_parameter('move_to_start', True)
        self.declare_parameter('frame_id', 'arm_base')
        self.declare_parameter('relative_waypoints', [
            0.005, 0.0, 0.0,
            0.010, 0.0, 0.0,
            0.010, 0.0, 0.005,
        ])
        self.declare_parameter('max_step', 0.005)
        self.declare_parameter('jump_threshold', 0.0)
        self.declare_parameter('prismatic_jump_threshold', 0.0)
        self.declare_parameter('revolute_jump_threshold', 0.0)
        self.declare_parameter('avoid_collisions', True)
        self.declare_parameter('min_fraction', 0.95)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('service_timeout_sec', 10.0)
        self.declare_parameter('motion_timeout_sec', 30.0)

        g = self.get_parameter
        self.group_name = g('group_name').value
        self.base_link = g('base_link').value
        self.end_effector = g('end_effector').value
        self.start_joints = list(g('start_joints').value)
        self.move_to_start = g('move_to_start').value
        self.frame_id = g('frame_id').value
        self.relative_waypoint_values = list(g('relative_waypoints').value)
        self.max_step = g('max_step').value
        self.jump_threshold = g('jump_threshold').value
        self.prismatic_jump_threshold = g('prismatic_jump_threshold').value
        self.revolute_jump_threshold = g('revolute_jump_threshold').value
        self.avoid_collisions = g('avoid_collisions').value
        self.min_fraction = g('min_fraction').value
        self.service_timeout = g('service_timeout_sec').value
        self.motion_timeout = g('motion_timeout_sec').value

        self.cb_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ARM_JOINTS,
            base_link_name=self.base_link,
            end_effector_name=self.end_effector,
            group_name=self.group_name,
            callback_group=self.cb_group,
            use_move_group_action=True,
        )
        self.moveit2.planning_time = g('planning_time').value
        self.cartesian_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path',
            callback_group=self.cb_group,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _make_relative_waypoints(self):
        values = self.relative_waypoint_values
        if not values or len(values) % 3 != 0:
            self.get_logger().error(
                'relative_waypoints must be a non-empty flat list of '
                '[dx, dy, dz] groups')
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.end_effector,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.service_timeout),
            ).transform
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'failed to get {self.frame_id}->{self.end_effector}: {exc}')
            return None

        q = transform.rotation
        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if q_norm < 1e-9:
            self.get_logger().error('current end-effector quaternion is invalid')
            return None
        waypoints = []
        for index in range(0, len(values), 3):
            dx, dy, dz = [float(value) for value in values[index:index + 3]]
            pose = Pose()
            pose.position.x = transform.translation.x + dx
            pose.position.y = transform.translation.y + dy
            pose.position.z = transform.translation.z + dz
            pose.orientation.x = q.x / q_norm
            pose.orientation.y = q.y / q_norm
            pose.orientation.z = q.z / q_norm
            pose.orientation.w = q.w / q_norm
            waypoints.append(pose)
        return waypoints

    def _wait_motion_done(self):
        start = time.time()
        while self.moveit2.query_state() == MoveIt2State.IDLE:
            if time.time() - start > 2.0:
                self.get_logger().error('MoveIt did not accept the motion goal')
                return False
            time.sleep(0.02)
        while self.moveit2.query_state() != MoveIt2State.IDLE:
            if time.time() - start > self.motion_timeout:
                self.get_logger().error('motion execution timed out')
                return False
            time.sleep(0.02)
        return self.moveit2.motion_suceeded

    def _move_to_start(self):
        if len(self.start_joints) != len(ARM_JOINTS):
            self.get_logger().error('start_joints must contain six values')
            return False
        self.get_logger().info(f'moving to start joints: {self.start_joints}')
        self.moveit2.move_to_configuration(self.start_joints)
        return self._wait_motion_done()

    def _wait_for_joint_state(self):
        start = time.time()
        while self.moveit2.joint_state is None:
            if time.time() - start > self.service_timeout:
                self.get_logger().error('joint states are not available')
                return False
            time.sleep(0.05)
        return True

    def _plan(self, waypoints):
        if not self.cartesian_client.wait_for_service(
                timeout_sec=self.service_timeout):
            self.get_logger().error('/compute_cartesian_path is unavailable')
            return None
        if not self._wait_for_joint_state():
            return None

        request = GetCartesianPath.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = self.frame_id
        request.start_state.joint_state = self.moveit2.joint_state
        request.group_name = self.group_name
        request.link_name = self.end_effector
        request.waypoints = waypoints
        request.max_step = self.max_step
        request.jump_threshold = self.jump_threshold
        request.prismatic_jump_threshold = self.prismatic_jump_threshold
        request.revolute_jump_threshold = self.revolute_jump_threshold
        request.avoid_collisions = self.avoid_collisions

        self.get_logger().info(
            f'planning Cartesian path with {len(waypoints)} waypoints')
        future = self.cartesian_client.call_async(request)
        start = time.time()
        while not future.done():
            if time.time() - start > self.service_timeout:
                self.get_logger().error('Cartesian planning service timed out')
                return None
            time.sleep(0.02)
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Cartesian planning call failed: {exc}')
            return None

    def run(self):
        if self.move_to_start and not self._move_to_start():
            self.get_logger().error('failed to reach start joints')
            return False
        waypoints = self._make_relative_waypoints()
        if waypoints is None:
            return False

        response = self._plan(waypoints)
        if response is None:
            return False
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'Cartesian planning failed, error={response.error_code.val}')
            return False
        self.get_logger().info(f'Cartesian path fraction={response.fraction:.3f}')
        if response.fraction < self.min_fraction:
            self.get_logger().error(
                f'fraction {response.fraction:.3f} is below '
                f'min_fraction {self.min_fraction:.3f}; path will not execute')
            return False
        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            self.get_logger().error('Cartesian planner returned an empty trajectory')
            return False

        self.get_logger().info('executing Cartesian trajectory')
        self.moveit2.execute(trajectory)
        if not self._wait_motion_done():
            self.get_logger().error('Cartesian trajectory execution failed')
            return False
        self.get_logger().info('Cartesian path demo completed')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPathDemo()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(2.0)
    try:
        if rclpy.ok():
            node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
