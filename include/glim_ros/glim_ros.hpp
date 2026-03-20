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
#include <visualization_msgs/msg/marker.hpp>

#include <std_msgs/msg/header.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/dynamic_rejection/voxel_filtering.hpp>
#include <glim/dynamic_rejection/wall_bbox.hpp>

namespace glim {

class TimeKeeper;
class CloudPreprocessor;
class AsyncOdometryEstimation;
class AsyncSubMapping;
class AsyncGlobalMapping;
class AsyncDynamicObjectRejection;
class DynamicBBoxRejection;
class DynamicObjectRejectionCPU;
class WallFilter;
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

  void bbox_callback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg);
  void wait(bool auto_quit = false);
  void save(const std::string& path);

  const std::vector<std::shared_ptr<GenericTopicSubscription>>& extension_subscriptions();

private:
  void publish_voxelmap(const std_msgs::msg::Header& header,
                        const gtsam_points::DynamicVoxelMapCPU& voxelmap);

  /// Publish a PointCloud2 containing only the wall-classified voxel centroids.
  void publish_wall_voxelmap(const std_msgs::msg::Header& header,
                             const WallFilterResult& wf_result);

  // ---------------------------------------------------------------------------
  // Core modules
  // ---------------------------------------------------------------------------
  std::unique_ptr<glim::TimeKeeper>            time_keeper;
  std::unique_ptr<glim::CloudPreprocessor>     preprocessor;
  std::shared_ptr<glim::AsyncOdometryEstimation> odometry_estimation;
  std::unique_ptr<glim::AsyncSubMapping>       sub_mapping;
  std::unique_ptr<glim::AsyncGlobalMapping>    global_mapping;

  // ---------------------------------------------------------------------------
  // Dynamic rejection — BBOX mode
  // ---------------------------------------------------------------------------
  std::shared_ptr<glim::DynamicBBoxRejection>  dynamic_bbox_rejection;

  // ---------------------------------------------------------------------------
  // Dynamic rejection — VOXEL mode
  // Two-stage pipeline: WallFilter → DynamicObjectRejectionCPU,
  // wrapped in an async thread by AsyncDynamicObjectRejection.
  // ---------------------------------------------------------------------------
  std::shared_ptr<glim::WallFilter>               wall_filter;
  std::shared_ptr<glim::AsyncDynamicObjectRejection> dynamic_object_rejection;

  std::shared_ptr<glim::WallBBoxRegistry> wall_bbox_registry_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr wall_bbox_pub_;
  // ---------------------------------------------------------------------------
  // Pose Kalman filter (shared with odometry / dynamic rejection)
  // ---------------------------------------------------------------------------
  std::shared_ptr<glim::PoseKalmanFilter> pose_kalman_filter;
  double last_imu_stamp_;
  struct KfImuData { Eigen::Vector3d acc; Eigen::Vector3d gyro; double dt; };
  std::mutex              kf_imu_mutex_;
  std::deque<KfImuData>   kf_imu_queue_;

  // ---------------------------------------------------------------------------
  // Config / flags
  // ---------------------------------------------------------------------------
  bool        keep_raw_points;
  double      imu_time_offset;
  double      points_time_offset;
  double      acc_scale;
  bool        dump_on_unload;
  std::string dynamic_rejection_type;
  std::string intensity_field;
  std::string ring_field;

  // ---------------------------------------------------------------------------
  // Extension modules
  // ---------------------------------------------------------------------------
  std::vector<std::shared_ptr<ExtensionModule>>          extension_modules;
  std::vector<std::shared_ptr<GenericTopicSubscription>> extension_subs;

  // ---------------------------------------------------------------------------
  // ROS subscribers / publishers / timer
  // ---------------------------------------------------------------------------
  rclcpp::TimerBase::SharedPtr timer;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr         imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;

  // BBOX mode
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_points_bbox_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_points_bbox_pub;

  // VOXEL mode
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr     filtered_points_voxel_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr     dynamic_points_voxel_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxelmap_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr     wall_points_pub;

  // Kalman-filtered pose
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr filtered_pose_pub;

#ifdef BUILD_WITH_CV_BRIDGE
  image_transport::Subscriber image_sub;
#endif
};

}  // namespace glim