#!/usr/bin/env python3
"""可配置颜色方块检测节点。

订阅 RGB / 深度 / camera_info，用 HSV 阈值分割指定颜色区域，取最大连通域质心，
结合该像素处的深度值与相机内参反投影出光学坐标系下的 3D 点，再用 tf2
变换到规划基座坐标系，发布为 PoseStamped。
"""
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped

from cv_bridge import CvBridge

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class CubeDetector(Node):
    COLOR_H_RANGES = {
        'red': [(0, 10), (160, 179)],
        'green': [(35, 85)],
        'blue': [(90, 130)],
    }

    def __init__(self):
        super().__init__('cube_detector')

        self.declare_parameter('target_color', 'red')
        self.declare_parameter('rgb_topic', '/camera/image_raw/image')
        self.declare_parameter('depth_topic', '/camera/image_raw/depth_image')
        self.declare_parameter('camera_info_topic', '/camera/image_raw/camera_info')
        self.declare_parameter('optical_frame', 'camera_link_optical')
        self.declare_parameter('fallback_optical_frame', 'camera_link')
        self.declare_parameter('target_frame', 'arm_base')
        self.declare_parameter('h_low1', 0)
        self.declare_parameter('h_high1', 10)
        self.declare_parameter('h_low2', 160)
        self.declare_parameter('h_high2', 179)
        self.declare_parameter('s_min', 100)
        self.declare_parameter('v_min', 60)
        self.declare_parameter('min_area_px', 200)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('target_pose_topic', '/red_cube/pose')
        self.declare_parameter('debug_image_topic', '/red_cube/debug_image')

        g = self.get_parameter
        self.target_color = str(g('target_color').value).lower()
        if self.target_color not in self.COLOR_H_RANGES:
            raise ValueError(
                f'unsupported target_color={self.target_color!r}; '
                'expected red, green, or blue')
        self.rgb_topic = g('rgb_topic').value
        self.depth_topic = g('depth_topic').value
        self.info_topic = g('camera_info_topic').value
        self.optical_frame = g('optical_frame').value
        self.fallback_frame = g('fallback_optical_frame').value
        self.target_frame = g('target_frame').value
        self.h_low1 = g('h_low1').value
        self.h_high1 = g('h_high1').value
        self.h_low2 = g('h_low2').value
        self.h_high2 = g('h_high2').value
        self.s_min = g('s_min').value
        self.v_min = g('v_min').value
        self.min_area = g('min_area_px').value
        self.publish_debug = g('publish_debug_image').value
        self.pose_topic = g('target_pose_topic').value
        self.debug_topic = g('debug_image_topic').value
        h_ranges = self.COLOR_H_RANGES[self.target_color]
        if self.target_color == 'red':
            h_ranges = [(self.h_low1, self.h_high1),
                        (self.h_low2, self.h_high2)]
        self.hsv_bounds = [
            (np.array([h_low, self.s_min, self.v_min], dtype=np.uint8),
             np.array([h_high, 255, 255], dtype=np.uint8))
            for h_low, h_high in h_ranges
        ]
        self.morphology_kernel = np.ones((5, 5), np.uint8)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        self.resolved_optical_frame = None  # 运行时确定实际可用的 optical frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, self.info_topic,
                                 self.info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic,
                                 self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.rgb_topic,
                                 self.rgb_cb, qos_profile_sensor_data)

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, self.debug_topic, 1)

        self.get_logger().info(
            f'cube_detector started. color={self.target_color} rgb={self.rgb_topic} '
            f'depth={self.depth_topic} target_frame={self.target_frame} '
            f'pose_topic={self.pose_topic}')

    def info_cb(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        # 只接受明确的 optical frame；普通 camera_link 不符合反投影坐标约定。
        if (self.resolved_optical_frame is None
                and 'optical' in msg.header.frame_id.lower()):
            self.resolved_optical_frame = msg.header.frame_id

    def depth_cb(self, msg: Image):
        # Gazebo 深度通常是 32FC1（米）
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'depth convert failed: {e}')
            return
        self.latest_depth = np.asarray(depth, dtype=np.float32)

    def _color_mask(self, hsv):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_bounds:
            mask |= cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self.morphology_kernel)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self.morphology_kernel)
        return mask

    def _sample_depth(self, u, v):
        """在 (u,v) 附近取有效深度中值，抗噪声/空洞。"""
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape[:2]
        u0, u1 = max(0, u - 3), min(w, u + 4)
        v0, v1 = max(0, v - 3), min(h, v + 4)
        patch = self.latest_depth[v0:v1, u0:u1].reshape(-1)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def rgb_cb(self, msg: Image):
        if self.fx is None:
            return  # 等待 camera_info
        if self.latest_depth is None:
            return  # 等待深度
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'rgb convert failed: {e}')
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = self._color_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._publish_debug(bgr, None)
            return
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            self._publish_debug(bgr, None)
            return
        M = cv2.moments(c)
        if M['m00'] == 0:
            return
        u = int(M['m10'] / M['m00'])
        v = int(M['m01'] / M['m00'])

        z = self._sample_depth(u, v)
        if z is None:
            self.get_logger().warn('no valid depth at cube centroid', throttle_duration_sec=2.0)
            self._publish_debug(bgr, (u, v))
            return

        # 反投影到光学坐标系 (REP 103: z 向前, x 右, y 下)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        src_frame = self.resolved_optical_frame or self.optical_frame
        pt = PointStamped()
        pt.header.stamp = msg.header.stamp
        pt.header.frame_id = src_frame
        pt.point.x = x
        pt.point.y = y
        pt.point.z = z

        world_pt = self._transform_point(pt)
        if world_pt is None:
            # 尝试 fallback frame（万一 optical frame 不在 TF 树里）
            pt.header.frame_id = self.fallback_frame
            world_pt = self._transform_point(pt)
            if world_pt is None:
                self._publish_debug(bgr, (u, v))
                return

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.target_frame
        ps.pose.position = world_pt.point
        ps.pose.orientation.w = 1.0
        self.pose_pub.publish(ps)
        self.get_logger().info(
            f'{self.target_color} cube @ px({u},{v}) z={z:.3f}m -> '
            f'{self.target_frame} ({world_pt.point.x:.3f}, '
            f'{world_pt.point.y:.3f}, {world_pt.point.z:.3f})',
            throttle_duration_sec=1.0)
        self._publish_debug(bgr, (u, v))

    def _transform_point(self, pt: PointStamped):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, pt.header.frame_id,
                rclpy.time.Time(),  # latest available
                timeout=rclpy.duration.Duration(seconds=0.2))
            return do_transform_point(pt, tf)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {pt.header.frame_id}->{self.target_frame} failed: {e}',
                throttle_duration_sec=2.0)
            return None

    def _publish_debug(self, bgr, centroid):
        if not self.publish_debug:
            return
        img = bgr.copy()
        if centroid is not None:
            draw_colors = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
            }
            color = draw_colors[self.target_color]
            cv2.circle(img, centroid, 6, color, 2)
            cv2.putText(img, f'{self.target_color} cube',
                        (centroid[0] + 8, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'debug image publish failed: {exc}',
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CubeDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
