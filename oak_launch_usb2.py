import os
os.environ["DEPTHAI_USB2_MODE"] = "1"

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("depthai_ros_driver"),
                    "launch", "camera.launch.py"
                )
            ),
            launch_arguments={
                "name":           "oak",
                "parent_frame":   "base_link",
                "use_rviz":       "false",
                "enableRgb":      "true",
                "enableDepth":    "false",
                "rgbResolution":  "THE_400_P",
            }.items(),
        )
    ])
