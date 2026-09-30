from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    kortex_bringup_dir = get_package_share_directory('kortex_bringup')
    
    # Launch the robot arm
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([kortex_bringup_dir, '/launch/gen3.launch.py']),
        launch_arguments={
            'robot_ip': '192.168.1.10',
            'gripper': 'robotiq_2f_140',
            'gripper_joint_name': 'finger_joint'
        }.items()
    )
    
    # Static transform from robot wrist to arm camera
    arm_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='arm_camera_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'bracelet_link', 'camera_link']
    )
    
    # Launch Kinova wrist camera (slave camera)
    kinova_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('kinova_vision'),
            '/launch/kinova_vision.launch.py'
        ])
    )
    
    return LaunchDescription([
        robot_launch,
        arm_camera_tf,
        kinova_vision_launch
    ])
