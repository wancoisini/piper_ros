import os
from glob import glob
from setuptools import setup

package_name = 'piper_vision_grasp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Configurable color-cube detection, grasping, and Cartesian path demos for Piper.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'red_cube_detector = piper_vision_grasp.red_cube_detector:main',
            'cube_detector = piper_vision_grasp.red_cube_detector:main',
            'pick_cube = piper_vision_grasp.pick_cube:main',
            'cartesian_path_demo = piper_vision_grasp.cartesian_path_demo:main',
        ],
    },
)
