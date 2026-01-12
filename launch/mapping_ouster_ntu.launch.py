import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_fast_livo = get_package_share_directory('fast_livo')
    
    rviz_use = LaunchConfiguration('rviz')
    
    ntu_config = os.path.join(pkg_fast_livo, 'config', 'NTU_VIRAL.yaml')
    camera_config = os.path.join(pkg_fast_livo, 'config', 'camera_NTU_VIRAL.yaml')
    rviz_config = os.path.join(pkg_fast_livo, 'rviz_cfg', 'livo.rviz')

    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    fast_livo_node = Node(
        package='fast_livo',
        executable='fastlivo_mapping',
        name='laserMapping',
        output='screen',
        parameters=[
            ntu_config,
            camera_config
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(rviz_use)
    )

    return LaunchDescription([
        declare_rviz_arg,
        fast_livo_node,
        rviz_node
    ])