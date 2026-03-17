#pragma once

#include <any>
#include <deque>
#include <mutex>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#ifdef BUILD_WITH_CV_BRIDGE
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#endif
#include <visualization_msgs/msg/marker_array.hpp>
#include <glim/odometry/estimation_frame.hpp>



namespace glim {
class TimeKeeper;
class CloudPreprocessor;
class AsyncOdometryEstimation;
class AsyncSubMapping;
class AsyncGlobalMapping;
class AsyncDynamicObjectRejection;
class DynamicObjectRejectionCPU;
class PoseKalmanFilter;
class ExtensionModule;
class GenericTopicSubscription;

class GlimROS : public rclcpp::Node {
public:
  GlimROS(const rclcpp::NodeOptions& options);
  ~GlimROS();

  bool needs_wait();
  void timer_callback();

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);
#ifdef BUILD_WITH_CV_BRIDGE
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
#endif
  size_t points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);


#ifdef GLIM_USE_DYNAMIC_REJECTION_BBOX
  void bbox_callback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg);
#endif
  void wait(bool auto_quit = false);
  void save(const std::string& path);

  const std::vector<std::shared_ptr<GenericTopicSubscription>>& extension_subscriptions();

private:
  std::unique_ptr<glim::TimeKeeper> time_keeper;
  std::unique_ptr<glim::CloudPreprocessor> preprocessor;
#ifdef GLIM_USE_DYNAMIC_REJECTION_BBOX
  std::shared_ptr<glim::DynamicBBoxRejection> dynamic_bbox_rejection;
#endif
#ifdef GLIM_USE_DYNAMIC_REJECTION_VOXEL
  std::shared_ptr<glim::AsyncDynamicObjectRejection> dynamic_object_rejection;
#endif
  std::shared_ptr<glim::PoseKalmanFilter> pose_kalman_filter;
  double last_imu_stamp_;
  struct KfImuData { Eigen::Vector3d acc; Eigen::Vector3d gyro; double dt; };
  std::mutex kf_imu_mutex_;
  std::deque<KfImuData> kf_imu_queue_;
  std::shared_ptr<glim::AsyncOdometryEstimation> odometry_estimation;
  std::unique_ptr<glim::AsyncSubMapping> sub_mapping;
  std::unique_ptr<glim::AsyncGlobalMapping> global_mapping;
  
  bool keep_raw_points;
  double imu_time_offset;
  double points_time_offset;
  double acc_scale;
  bool dump_on_unload;

  std::string intensity_field, ring_field;

  // Extension modulles
  std::vector<std::shared_ptr<ExtensionModule>> extension_modules;
  std::vector<std::shared_ptr<GenericTopicSubscription>> extension_subs;

  // ROS-related
  rclcpp::TimerBase::SharedPtr timer;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_sub;
#ifdef GLIM_USE_DYNAMIC_REJECTION_BBOX
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_points_bbox_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_points_bbox_pub;
#endif
#ifdef GLIM_USE_DYNAMIC_REJECTION_VOXEL
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_points_voxel_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_points_voxel_pub;
#endif
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr filtered_pose_pub;

#ifdef BUILD_WITH_CV_BRIDGE
  image_transport::Subscriber image_sub;
#endif
};

}  // namespace glim
