#include "global_camera_perception/glass_perception.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace global_camera_perception
{

static const rclcpp::Logger LOGGER = rclcpp::get_logger("GlassPerception");

using std::placeholders::_1;
using std::placeholders::_2;

GlassPerception::GlassPerception(const rclcpp::NodeOptions & options)
: rclcpp::Node("glass_perception", options), debug_(true), find_objects_(false)
{
  clock_ = this->get_clock();

  loop_set_       = this->declare_parameter<int>("loop_set", 1);
  count_callback_ = 0;
  debug_          = this->declare_parameter<bool>("debug_topics", true);
  world_frame_    = this->declare_parameter<std::string>("frame_id", "base_link");
  camera_topic_   = this->declare_parameter<std::string>(
    "camera_topic", "/global_camera/global_camera/depth/color/points");

  // Debug publishers
  if (debug_) {
    rclcpp::QoS qos(1);
    qos.best_effort();
    filter_cloud_pub_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("filter_cloud", qos);
    plate_cloud_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("plate_cloud", qos);
    holes_cloud_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("holes_cloud", qos);
    colored_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("color_hole_cloud", qos);
    marker_pub_        = this->create_publisher<visualization_msgs::msg::MarkerArray>("marker", qos);
  }

  // Range filters
  range_filter_x_.setFilterFieldName("x");
  range_filter_x_.setFilterLimits(-0.3, 0.1);

  range_filter_y_.setFilterFieldName("y");
  range_filter_y_.setFilterLimits(0.1, 0.4);

  range_filter_z_.setFilterFieldName("z");
  range_filter_z_.setFilterLimits(-0.2, 0.1);

  // Outlier removal
  outliers_filter_.setMeanK(50);
  outliers_filter_.setStddevMulThresh(2.5);

  // Voxel grid
  double leaf = 0.005;
  voxel_grid_.setLeafSize(leaf, leaf, leaf);

  // Plane segmentation
  segment_plane_.setOptimizeCoefficients(true);
  segment_plane_.setModelType(pcl::SACMODEL_PLANE);
  segment_plane_.setNormalDistanceWeight(0.005);
  segment_plane_.setMaxIterations(100);
  segment_plane_.setDistanceThreshold(0.01);

  // TF
  buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer_);

  // Accumulated clouds
  plate_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  hole_cloud_  = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  // Subscribe to point cloud
  rclcpp::QoS points_qos(10);
  points_qos.best_effort();
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    camera_topic_, points_qos,
    std::bind(&GlassPerception::cloud_callback, this, _1));

  // Action server
  server_ = rclcpp_action::create_server<FindGlassAction>(
    this->get_node_base_interface(),
    this->get_node_clock_interface(),
    this->get_node_logging_interface(),
    this->get_node_waitables_interface(),
    "find_glass",
    std::bind(&GlassPerception::handle_goal, this, _1, _2),
    std::bind(&GlassPerception::handle_cancel, this, _1),
    std::bind(&GlassPerception::handle_accepted, this, _1));

  RCLCPP_INFO(LOGGER, "GlassPerception initialized");
}

rclcpp_action::GoalResponse GlassPerception::handle_goal(
  const rclcpp_action::GoalUUID &,
  std::shared_ptr<const FindGlassAction::Goal>)
{
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse GlassPerception::handle_cancel(
  const std::shared_ptr<FindGlassGoalHandle>)
{
  return rclcpp_action::CancelResponse::ACCEPT;
}

void GlassPerception::handle_accepted(const std::shared_ptr<FindGlassGoalHandle> goal_handle)
{
  std::thread{std::bind(&GlassPerception::execute, this, _1), goal_handle}.detach();
}

void GlassPerception::execute(const std::shared_ptr<FindGlassGoalHandle> goal_handle)
{
  auto result   = std::make_shared<FindGlassAction::Result>();
  auto feedback = std::make_shared<FindGlassAction::Feedback>();
  const auto goal = goal_handle->get_goal();

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    return;
  }

  find_objects_ = goal->find_glass;

  while (find_objects_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  feedback->process_description = process_descriptions_;
  goal_handle->publish_feedback(feedback);

  result->glass_center  = glass_center_;
  result->glass_radius  = glass_radius_;
  result->table_center  = table_center_;

  goal_handle->succeed(result);
}

void GlassPerception::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!find_objects_) return;

  count_callback_++;
  RCLCPP_INFO(LOGGER, "Count: %d", count_callback_);

  process_descriptions_.clear();
  glass_center_.x = 0.0;
  glass_center_.y = 0.0;
  glass_center_.z = 0.0;
  glass_radius_   = 0.0;
  table_center_.x = 0.0;
  table_center_.y = 0.0;
  table_center_.z = 0.0;

  // Convert to PCL
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*msg, *cloud);

  // Transform to base_link
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
  if (!pcl_ros::transformPointCloud(world_frame_, *cloud, *cloud_transformed, *buffer_)) {
    RCLCPP_ERROR(LOGGER, "Error transforming to frame %s", world_frame_.c_str());
    return;
  }

  // Range filter
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  range_filter_z_.setInputCloud(cloud_transformed);
  range_filter_z_.filter(*cloud_filtered);
  range_filter_x_.setInputCloud(cloud_filtered);
  range_filter_x_.filter(*cloud_filtered);
  range_filter_y_.setInputCloud(cloud_filtered);
  range_filter_y_.filter(*cloud_filtered);

  // Outlier removal
  outliers_filter_.setInputCloud(cloud_filtered);
  outliers_filter_.filter(*cloud_filtered);

  if (debug_) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_filtered, cloud_msg);
    filter_cloud_pub_->publish(cloud_msg);
  }

  // Segment
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr plate_store(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr hole_store(new pcl::PointCloud<pcl::PointXYZRGB>);

  plate_cloud_->header.frame_id = cloud_filtered->header.frame_id;
  hole_cloud_->header.frame_id  = cloud_filtered->header.frame_id;

  segment(cloud_filtered, plate_store, hole_store);
  *plate_cloud_ += *plate_store;
  *hole_cloud_  += *hole_store;

  if (count_callback_ < loop_set_) return;

  // Extract clusters
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_vector;
  extract_clusters(hole_cloud_, cloud_vector);

  if (!cloud_vector.empty()) {
    project_and_hull(cloud_vector, hole_cloud_);
    std::vector<float> xc, yc, zc, r;
    estimate_circle_params(cloud_vector, xc, yc, zc, r);
    geometry_msgs::msg::Point table_center;
    publish_markers(plate_cloud_, xc, yc, zc, r, table_center);
    table_center_ = table_center;

    // Take the cluster with largest radius as the glass
    size_t best = 0;
    for (size_t i = 1; i < r.size(); ++i) {
      if (r[i] > r[best]) best = i;
    }
    glass_center_.x = xc[best];
    glass_center_.y = yc[best];
    glass_center_.z = zc[best];
    glass_radius_   = r[best];

    RCLCPP_INFO(LOGGER, "Glass center: (%.4f, %.4f, %.4f) radius: %.4f",
      glass_center_.x, glass_center_.y, glass_center_.z, glass_radius_);

  } else {
    RCLCPP_INFO(LOGGER, "No glass detected on table");
    process_descriptions_.push_back("No glass detected");
  }

  if (debug_) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*plate_cloud_, cloud_msg);
    plate_cloud_pub_->publish(cloud_msg);
    pcl::toROSMsg(*hole_cloud_, cloud_msg);
    holes_cloud_pub_->publish(cloud_msg);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto & c : cloud_vector) *combined += *c;
    pcl::toROSMsg(*combined, cloud_msg);
    cloud_msg.header.frame_id = cloud_filtered->header.frame_id;
    colored_cloud_pub_->publish(cloud_msg);
  }

  count_callback_ = 0;
  plate_cloud_->clear();
  hole_cloud_->clear();
  find_objects_ = false;
}

void GlassPerception::segment(
  const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr & input,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr plate,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr holes)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  voxel_grid_.setInputCloud(input);
  voxel_grid_.filter(*voxel_cloud);

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setSearchMethod(tree);
  ne.setInputCloud(voxel_cloud);
  ne.setKSearch(50);
  ne.compute(*normals);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  segment_plane_.setInputCloud(voxel_cloud);
  segment_plane_.setInputNormals(normals);
  segment_plane_.segment(*inliers, *coefficients);

  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(voxel_cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*plate);

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*plate, centroid);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr above_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
  extract.setNegative(true);
  extract.filter(*above_plane);

  float radius = 0.125;
  for (const auto & pt : above_plane->points) {
    float dx = pt.x - centroid.x();
    float dy = pt.y - centroid.y();
    if (std::sqrt(dx*dx + dy*dy) <= radius) {
      holes->points.push_back(pt);
    }
  }
}

void GlassPerception::extract_clusters(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr & hole_cloud,
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloud_vector)
{
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setInputCloud(hole_cloud);
  ec.setClusterTolerance(0.01);
  ec.setMinClusterSize(50);
  ec.setMaxClusterSize(100000);
  std::vector<pcl::PointIndices> clusters;
  ec.extract(clusters);

  RCLCPP_INFO(LOGGER, "Extracted %zu clusters", clusters.size());

  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(hole_cloud);
  for (size_t i = 0; i < clusters.size(); ++i) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
    extract.setIndices(pcl::PointIndicesPtr(new pcl::PointIndices(clusters[i])));
    extract.filter(*cluster);
    cloud_vector.push_back(cluster);
    RCLCPP_INFO(LOGGER, "Cluster %zu size: %zu", i+1, cluster->size());
  }
}

void GlassPerception::project_and_hull(
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloud_vector,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr & hole_cloud)
{
  pcl::PointXYZRGB min_pt, max_pt;
  pcl::getMinMax3D(*hole_cloud, min_pt, max_pt);

  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setStddevMulThresh(1.0);

  for (auto & cloud : cloud_vector) {
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    coeff->values.resize(4);
    coeff->values[0] = coeff->values[1] = 0;
    coeff->values[2] = 1;
    coeff->values[3] = -min_pt.z;

    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coeff);
    proj.filter(*cloud);

    pcl::ConvexHull<pcl::PointXYZRGB> hull;
    hull.setInputCloud(cloud);
    hull.setDimension(2);
    hull.reconstruct(*cloud);

    sor.setMeanK(static_cast<int>(cloud->size()));
    sor.setInputCloud(cloud);
    sor.filter(*cloud);
  }
}

void GlassPerception::estimate_circle_params(
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & cloud_vector,
  std::vector<float> & xc, std::vector<float> & yc,
  std::vector<float> & zc, std::vector<float> & r)
{
  for (size_t i = 0; i < cloud_vector.size(); ++i) {
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*cloud_vector[i], min_pt, max_pt);
    zc.push_back((2 * min_pt.z + 0.10) / 2.0);

    int N = cloud_vector[i]->size();
    Eigen::MatrixXd A(N, 3);
    Eigen::MatrixXd B(N, 1);
    for (int j = 0; j < N; ++j) {
      float x = cloud_vector[i]->points[j].x;
      float y = cloud_vector[i]->points[j].y;
      A(j, 0) = x; A(j, 1) = y; A(j, 2) = 1;
      B(j)    = x * x + y * y;
    }
    Eigen::Vector3d X = (A.transpose() * A).ldlt().solve(A.transpose() * B);
    xc.push_back(X(0) / 2);
    yc.push_back(X(1) / 2);
    r.push_back(std::sqrt(4 * X(2) + X(0)*X(0) + X(1)*X(1)) / 2);

    RCLCPP_INFO(LOGGER, "Glass %zu center: (%.4f, %.4f, %.4f) radius: %.4f",
      i+1, xc[i], yc[i], zc[i], r[i]);
  }
}

void GlassPerception::publish_markers(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr & plate_cloud,
  std::vector<float> & xc, std::vector<float> & yc,
  std::vector<float> & zc, std::vector<float> & r,
  geometry_msgs::msg::Point & table_center)
{
  (void)zc;
  visualization_msgs::msg::MarkerArray marker_array;

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*plate_cloud, centroid);
  table_center.x = centroid.x();
  table_center.y = centroid.y();
  table_center.z = centroid.z();

  // Table center marker — red sphere
  visualization_msgs::msg::Marker center_marker;
  center_marker.header.frame_id = world_frame_;
  center_marker.ns     = "table_center";
  center_marker.id     = 100;
  center_marker.type   = visualization_msgs::msg::Marker::SPHERE;
  center_marker.action = visualization_msgs::msg::Marker::ADD;
  center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.02;
  center_marker.color.r = 1.0; center_marker.color.a = 1.0;
  center_marker.pose.position.x = centroid.x();
  center_marker.pose.position.y = centroid.y();
  center_marker.pose.position.z = centroid.z();
  marker_array.markers.push_back(center_marker);

  for (size_t i = 0; i < xc.size(); ++i) {
    // Glass center — yellow sphere
    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = world_frame_;
    sphere.ns     = "glass_center_" + std::to_string(i);
    sphere.id     = i;
    sphere.type   = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02;
    sphere.color.r = 1.0; sphere.color.g = 1.0; sphere.color.a = 1.0;
    sphere.pose.position.x = xc[i];
    sphere.pose.position.y = yc[i];
    sphere.pose.position.z = centroid.z();
    marker_array.markers.push_back(sphere);

    // Glass circle outline — blue line strip
    visualization_msgs::msg::Marker circle;
    circle.header.frame_id = world_frame_;
    circle.ns     = "glass_circle_" + std::to_string(i);
    circle.id     = i;
    circle.type   = visualization_msgs::msg::Marker::LINE_STRIP;
    circle.action = visualization_msgs::msg::Marker::ADD;
    circle.scale.x = 0.005;
    circle.color.b = 1.0; circle.color.a = 1.0;
    const int pts = 72;
    for (int j = 0; j <= pts; ++j) {
      float angle = j * 2.0 * M_PI / pts;
      geometry_msgs::msg::Point p;
      p.x = xc[i] + r[i] * std::cos(angle);
      p.y = yc[i] + r[i] * std::sin(angle);
      p.z = centroid.z();
      circle.points.push_back(p);
    }
    marker_array.markers.push_back(circle);
  }

  marker_pub_->publish(marker_array);
}

}  // namespace global_camera_perception

RCLCPP_COMPONENTS_REGISTER_NODE(global_camera_perception::GlassPerception)
