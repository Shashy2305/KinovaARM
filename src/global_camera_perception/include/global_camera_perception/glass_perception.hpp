#ifndef GLOBAL_CAMERA_PERCEPTION__GLASS_PERCEPTION_HPP_
#define GLOBAL_CAMERA_PERCEPTION__GLASS_PERCEPTION_HPP_

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "pcl_conversions/pcl_conversions.h"
#include "pcl_ros/transforms.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/filters/project_inliers.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/segmentation/extract_clusters.h"
#include "pcl/common/centroid.h"
#include "pcl/common/common.h"
#include "pcl/features/normal_3d.h"
#include "pcl/surface/convex_hull.h"
#include "pcl/ModelCoefficients.h"
#include <Eigen/Eigen>

#include "global_camera_perception/action/find_glass_position.hpp"

namespace global_camera_perception
{

class GlassPerception : public rclcpp::Node
{
  using FindGlassAction    = global_camera_perception::action::FindGlassPosition;
  using FindGlassGoalHandle = rclcpp_action::ServerGoalHandle<FindGlassAction>;

public:
  explicit GlassPerception(const rclcpp::NodeOptions & options);

private:
  // Parameters
  std::string camera_topic_;
  std::string world_frame_;
  bool debug_;
  int loop_set_;
  int count_callback_;
  bool find_objects_;

  // Results
  geometry_msgs::msg::Point glass_center_;
  float glass_radius_;
  geometry_msgs::msg::Point table_center_;
  std::vector<std::string> process_descriptions_;

  // TF
  std::shared_ptr<tf2_ros::Buffer> buffer_;
  std::shared_ptr<tf2_ros::TransformListener> listener_;

  // ROS
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filter_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plate_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr holes_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // Action server
  rclcpp_action::Server<FindGlassAction>::SharedPtr server_;

  // PCL filters
  pcl::PassThrough<pcl::PointXYZRGB> range_filter_x_;
  pcl::PassThrough<pcl::PointXYZRGB> range_filter_y_;
  pcl::PassThrough<pcl::PointXYZRGB> range_filter_z_;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outliers_filter_;
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid_;
  pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> segment_plane_;

  // Accumulated clouds
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr plate_cloud_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr hole_cloud_;

  // Action callbacks
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const FindGlassAction::Goal> goal);

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<FindGlassGoalHandle> goal_handle);

  void handle_accepted(const std::shared_ptr<FindGlassGoalHandle> goal_handle);
  void execute(const std::shared_ptr<FindGlassGoalHandle> goal_handle);

  // Cloud callback
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // Pipeline steps
  void segment(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr & input,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plate,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr holes);

  void extract_clusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & hole_cloud,
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloud_vector);

  void project_and_hull(
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloud_vector,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & hole_cloud);

  void estimate_circle_params(
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloud_vector,
    std::vector<float> & xc, std::vector<float> & yc,
    std::vector<float> & zc, std::vector<float> & r);

  void publish_markers(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & plate_cloud,
    std::vector<float> & xc, std::vector<float> & yc,
    std::vector<float> & zc, std::vector<float> & r,
    geometry_msgs::msg::Point & table_center);
};

}  // namespace global_camera_perception

#endif
