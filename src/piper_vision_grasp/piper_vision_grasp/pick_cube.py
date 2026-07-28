#!/usr/bin/env python3
"""抓取节点：用 pymoveit2 驱动 piper_arm 规划到目标方块并执行 pick。

流程：等待目标位姿 -> 预抓取位(上方) -> 张开夹爪 -> 下降到抓取位
-> 闭合夹爪 -> 抬升。手臂用 MoveIt2 (piper_arm 组) 规划；夹爪走
piper_gripper_controller 的 follow_joint_trajectory。
"""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pymoveit2 import MoveIt2, MoveIt2State


ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
GRIPPER_JOINTS = ['joint7']


class PickCube(Node):
    def __init__(self):
        super().__init__('pick_cube')

        self.declare_parameter('target_color', 'red')
        self.declare_parameter('target_pose_topic', '/red_cube/pose')
        self.declare_parameter('base_link', 'arm_base')
        self.declare_parameter('end_effector', 'link6')
        self.declare_parameter('pregrasp_height', 0.12)
        self.declare_parameter('grasp_z_offset', 0.02)
        self.declare_parameter('lift_height', 0.15)
        self.declare_parameter('gripper_open', 0.05)
        self.declare_parameter('gripper_close', 0.0)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('tip_offset', 0.135)
        self.declare_parameter('grasp_qx', 0.0)
        self.declare_parameter('grasp_qy', 1.0)
        self.declare_parameter('grasp_qz', 0.0)
        self.declare_parameter('grasp_qw', 0.0)
        self.declare_parameter('observe_joints', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('home_joints', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('place_x', 0.30)
        self.declare_parameter('place_y', 0.15)
        self.declare_parameter('place_z', 0.035)
        self.declare_parameter('place_pregrasp_height', 0.05)

        g = self.get_parameter
        default_place = {
            'x': g('place_x').value,
            'y': g('place_y').value,
            'z': g('place_z').value,
        }
        for color in ('red', 'green', 'blue'):
            for axis, value in default_place.items():
                self.declare_parameter(f'{color}_place_{axis}', value)

        self.target_color = str(g('target_color').value).lower()
        if self.target_color not in ('red', 'green', 'blue'):
            raise ValueError(
                f'unsupported target_color={self.target_color!r}; '
                'expected red, green, or blue')
        self.pose_topic = g('target_pose_topic').value
        self.base_link = g('base_link').value
        self.eef = g('end_effector').value
        self.pregrasp_h = g('pregrasp_height').value
        self.grasp_z = g('grasp_z_offset').value
        self.lift_h = g('lift_height').value
        self.grip_open = g('gripper_open').value
        self.grip_close = g('gripper_close').value
        self.tip_offset = g('tip_offset').value
        self.quat = (g('grasp_qx').value, g('grasp_qy').value,
                     g('grasp_qz').value, g('grasp_qw').value)
        self.observe_joints = list(g('observe_joints').value)
        self.home_joints = list(g('home_joints').value)
        self.place_x = g(f'{self.target_color}_place_x').value
        self.place_y = g(f'{self.target_color}_place_y').value
        self.place_z = g(f'{self.target_color}_place_z').value
        self.place_pregrasp_h = g('place_pregrasp_height').value

        self.cb_group = ReentrantCallbackGroup()

        # use_move_group_action=True: 走 MoveGroup action 的纯异步路径。
        # 默认(False)的 plan()/wait_until_executed() 内部会调用
        # rclpy.spin_once(self._node)，与后台 MultiThreadedExecutor 争抢同一
        # 节点的 wait set，导致消息(含目标位姿)全部停摆且 goal 卡死。
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ARM_JOINTS,
            base_link_name=self.base_link,
            end_effector_name=self.eef,
            group_name='piper_arm',
            callback_group=self.cb_group,
            use_move_group_action=True,
        )
        self.moveit2.planning_time = g('planning_time').value

        # 夹爪直接发 JointTrajectory 到控制器（避免第二个 MoveIt2 action
        # client 与 arm 的共享执行器产生 wait set 越界，导致 spin 线程崩溃）。
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/piper_gripper_controller/joint_trajectory', 10)

        self.latest_pose = None
        self.pose_lock = threading.Lock()
        # pose 订阅放到独立回调组，避免被 MoveIt2 的回调占用而收不到消息
        self.pose_cb_group = ReentrantCallbackGroup()
        self.create_subscription(PoseStamped, self.pose_topic,
                                 self.pose_cb, 10,
                                 callback_group=self.pose_cb_group)

        self.get_logger().info(
            f'pick_cube ready. color={self.target_color} place=('
            f'{self.place_x:.3f}, {self.place_y:.3f}, {self.place_z:.3f}); '
            'waiting for target pose...')

    def pose_cb(self, msg: PoseStamped):
        with self.pose_lock:
            self.latest_pose = msg

    def get_target(self):
        with self.pose_lock:
            return self.latest_pose

    def _link6_from_tip(self, tip_x, tip_y, tip_z):
        """给定指尖目标位置，沿抓取姿态 z 轴回退 tip_offset 求 link6 目标。

        指尖在 link6 的 +z 方向 tip_offset 处；抓取姿态旋转后 link6 的 z 轴
        在 base 系的方向为 z_dir，则 link6 位置 = 指尖位置 - tip_offset * z_dir。
        """
        qx, qy, qz, qw = self.quat
        # 用四元数把局部 (0,0,1) 旋到 base 系（z 轴世界方向）
        zx = 2.0 * (qx * qz + qw * qy)
        zy = 2.0 * (qy * qz - qw * qx)
        zz = 1.0 - 2.0 * (qx * qx + qy * qy)
        return (tip_x - self.tip_offset * zx,
                tip_y - self.tip_offset * zy,
                tip_z - self.tip_offset * zz)

    def _wait_motion_done(self, timeout_sec=30.0):
        """轮询 query_state() 等运动结束。不能用 moveit2.wait_until_executed()，
        它内部 spin_once 会和后台执行器冲突。运动由后台执行器异步推进。"""
        t0 = time.time()
        # 先等 goal 被接受（离开 IDLE），最多等 2s
        while self.moveit2.query_state() == MoveIt2State.IDLE:
            if time.time() - t0 > 2.0:
                self.get_logger().warn('MoveIt did not accept the motion goal')
                return False
            time.sleep(0.02)
        # 再等回到 IDLE（执行完成）
        while self.moveit2.query_state() != MoveIt2State.IDLE:
            if time.time() - t0 > timeout_sec:
                self.get_logger().warn('motion wait timed out')
                return False
            time.sleep(0.02)
        return self.moveit2.motion_suceeded

    def move_tip_to(self, tip_x, tip_y, tip_z):
        lx, ly, lz = self._link6_from_tip(tip_x, tip_y, tip_z)
        self.get_logger().info(
            f'tip target ({tip_x:.3f}, {tip_y:.3f}, {tip_z:.3f}) '
            f'-> link6 ({lx:.3f}, {ly:.3f}, {lz:.3f})')
        self.moveit2.move_to_pose(
            position=[lx, ly, lz],
            quat_xyzw=list(self.quat),
            frame_id=self.base_link,
        )
        return self._wait_motion_done()

    def move_to_joints(self, joints, label=''):
        self.get_logger().info(f'moving to joint config {label}: {joints}')
        self.moveit2.move_to_configuration(list(joints))
        return self._wait_motion_done()

    def set_gripper(self, position, settle_sec=1.5):
        """直接发布 joint7 目标到夹爪控制器，joint8 由 joint8_ctrl 镜像。"""
        traj = JointTrajectory()
        traj.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(position)]
        pt.time_from_start.sec = 1
        traj.points = [pt]
        self.get_logger().info(f'[seq] set gripper joint7={position}')
        # 多发几次确保控制器收到
        for _ in range(3):
            self.gripper_pub.publish(traj)
            time.sleep(0.1)
        time.sleep(settle_sec)

    def wait_for_target(self, timeout_sec=30.0):
        """在当前(观察)位姿下等待检测到方块目标位姿。"""
        t0 = time.time()
        while rclpy.ok():
            tgt = self.get_target()
            if tgt is not None:
                return tgt
            if time.time() - t0 > timeout_sec:
                return None
            self.get_logger().info(
                f'[seq] waiting for {self.pose_topic} ...',
                throttle_duration_sec=2.0)
            time.sleep(0.2)
        return None

    def run_sequence(self):
        self.get_logger().info('[seq] start: opening gripper')
        # 0. 张开夹爪
        self.set_gripper(self.grip_open)

        # 1. 移动到观察位，让相机能看到方块
        self.get_logger().info('[seq] moving to observe pose')
        if not self.move_to_joints(self.observe_joints, 'observe'):
            self.get_logger().error('failed to reach observe pose')
            return False
        self.get_logger().info('[seq] reached observe pose, waiting for detection')

        # 2. 在观察位等待检测结果（清掉旧位姿，确保是当前视角的检测）
        with self.pose_lock:
            self.latest_pose = None
        target = self.wait_for_target()
        if target is None:
            self.get_logger().warn('no target pose detected at observe pose')
            return False
        px = target.pose.position.x
        py = target.pose.position.y
        pz = target.pose.position.z
        self.get_logger().info(f'target cube at ({px:.3f}, {py:.3f}, {pz:.3f})')

        # 3. 预抓取位（指尖在方块正上方）
        if not self.move_tip_to(px, py, pz + self.pregrasp_h):
            self.get_logger().error('failed to reach pregrasp pose')
            return False
        # 4. 下降到抓取位（指尖对准方块中心）
        if not self.move_tip_to(px, py, pz + self.grasp_z):
            self.get_logger().error('failed to reach grasp pose')
            return False
        # 5. 闭合夹爪抓取
        self.set_gripper(self.grip_close)
        # 6. 抬升
        if not self.move_tip_to(px, py, pz + self.lift_h):
            self.get_logger().error('failed to lift target')
            return False

        # 7. 移动到放置点上方
        if not self.move_tip_to(
                self.place_x, self.place_y,
                self.place_z + self.place_pregrasp_h):
            self.get_logger().error('failed to reach place pregrasp pose')
            return False
        # 8. 下降到放置点
        if not self.move_tip_to(self.place_x, self.place_y, self.place_z):
            self.get_logger().error('failed to reach place pose')
            return False
        # 9. 松开夹爪放下方块
        self.set_gripper(self.grip_open)
        # 10. 抬起离开放置点
        if not self.move_tip_to(
                self.place_x, self.place_y,
                self.place_z + self.place_pregrasp_h):
            self.get_logger().error('failed to leave place pose')
            return False

        # 11. 回到 home 姿态
        if not self.move_to_joints(self.home_joints, 'home'):
            self.get_logger().error('failed to return home')
            return False

        self.get_logger().info('pick-and-place sequence done')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PickCube()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 等待 MoveIt2 与关节状态就绪后执行完整取放流程
    time.sleep(2.0)
    try:
        if rclpy.ok():
            node.run_sequence()
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
