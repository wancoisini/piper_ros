import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('piper_vision_grasp')
    params = os.path.join(pkg, 'config', 'grasp_params.yaml')

    # Gazebo 只发布 camera_link，未发布光学坐标系 camera_link_optical。
    # 相机反投影的 3D 点在光学系（REP 103: z 向前, x 右, y 下）下，
    # 因此补一个 camera_link -> camera_link_optical 的静态变换。
    # 旋转 rpy = (-pi/2, 0, -pi/2) 将常规相机系(x前)转为光学系(z前)。
    optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_optical_tf',
        output='screen',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '-1.5707963', '--pitch', '0', '--yaw', '-1.5707963',
                   '--frame-id', 'camera_link',
                   '--child-frame-id', 'camera_link_optical'],
        parameters=[{'use_sim_time': True}],
    )

    target_color = LaunchConfiguration('target_color')
    target_color_arg = DeclareLaunchArgument(
        'target_color',
        default_value='red',
        choices=['red', 'green', 'blue'],
        description='Target cube color: red, green, or blue',
    )
    show_debug_view = LaunchConfiguration('show_debug_view')
    show_debug_view_arg = DeclareLaunchArgument(
        'show_debug_view',
        default_value='true',
        choices=['true', 'false'],
        description='Open rqt_image_view for the annotated detection image',
    )

    detector = Node(
        package='piper_vision_grasp',
        executable='cube_detector',
        name='cube_detector',
        output='screen',
        parameters=[params, {
            'use_sim_time': True,
            'target_color': target_color,
        }],
    )

    picker = Node(
        package='piper_vision_grasp',
        executable='pick_cube',
        name='pick_cube',
        output='screen',
        parameters=[params, {
            'use_sim_time': True,
            'target_color': target_color,
        }],
    )

    debug_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='target_cube_debug_view',
        output='screen',
        arguments=['/target_cube/debug_image'],
        condition=IfCondition(show_debug_view),
    )

    return LaunchDescription([
        target_color_arg,
        show_debug_view_arg,
        optical_tf,
        detector,
        picker,
        debug_view,
    ])
