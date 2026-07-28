import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('piper_vision_grasp')
    params = os.path.join(pkg, 'config', 'cartesian_path_demo.yaml')

    return LaunchDescription([
        Node(
            package='piper_vision_grasp',
            executable='cartesian_path_demo',
            name='cartesian_path_demo',
            output='screen',
            parameters=[params, {'use_sim_time': True}],
        ),
    ])
