cmake_minimum_required(VERSION 3.8)
project(global_camera_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

# Generate action interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/FindGlassPosition.action"
  DEPENDENCIES geometry_msgs
)

set(dependencies
  rclcpp
  rclcpp_action
  rclcpp_components
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  pcl_ros
  pcl_conversions
)

include_directories(include)

add_library(glass_perception SHARED
  src/glass_perception.cpp
)

ament_target_dependencies(glass_perception ${dependencies})

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(glass_perception "${cpp_typesupport_target}")

rclcpp_components_register_node(glass_perception
  PLUGIN "global_camera_perception::GlassPerception"
  EXECUTABLE glass_perception_node
)

install(TARGETS glass_perception
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY action/
  DESTINATION share/${PROJECT_NAME}/action
)

ament_package()
