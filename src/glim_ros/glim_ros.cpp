#include <glim_ros/glim_ros.hpp>

#define GLIM_ROS2

#include <cstring>
#include <deque>
#include <sstream>
#include <thread>
#include <iostream>
#include <functional>
#include <boost/format.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <jo_msgs/msg/obstacle_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <gtsam_points/optimizers/linearization_hook.hpp>
#include <gtsam_points/cuda/nonlinear_factor_set_gpu_create.hpp>

#include <glim/util/debug.hpp>
#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/util/ros_cloud_converter.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <glim/mapping/async_sub_mapping.hpp>
#include <glim/mapping/async_global_mapping.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>
#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>
#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>
#include <glim/dynamic_rejection/dynamic_bounding_box_rejection.hpp>
#include <glim/dynamic_rejection/voxel_filtering.hpp>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim_ros/ros_compatibility.hpp>
#include <glim_ros/ros_qos.hpp>
#include <glim_ros/pointcloud2_msg.hpp>

namespace glim {

// =============================================================================
// Construction
// =============================================================================

GlimROS::GlimROS(const rclcpp::NodeOptions& options) : Node("glim_ros", options) {

  // ---------------------------------------------------------------------------
  // Logger
  // ---------------------------------------------------------------------------
  auto logger = spdlog::stdout_color_mt("glim");
  logger->sinks().push_back(get_ringbuffer_sink());
  spdlog::set_default_logger(logger);

  bool debug = false;
  this->declare_parameter<bool>("debug", false);
  this->get_parameter<bool>("debug", debug);

  if (debug) {
    spdlog::info("debug logging enabled");
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/glim_log.log", true);
    logger->sinks().push_back(file_sink);
    logger->set_level(spdlog::level::trace);
    print_system_info(logger);
  }

  // ---------------------------------------------------------------------------
  // Global config
  // ---------------------------------------------------------------------------
  dump_on_unload = false;
  this->declare_parameter<bool>("dump_on_unload", false);
  this->get_parameter<bool>("dump_on_unload", dump_on_unload);

  std::string config_path;
  this->declare_parameter<std::string>("config_path", "config");
  this->get_parameter<std::string>("config_path", config_path);

  if (config_path.empty() || config_path[0] != '/') {
    config_path = ament_index_cpp::get_package_share_directory("glim") + "/" + config_path;
  }
  logger->info("config_path: {}", config_path);
  glim::GlobalConfig::instance(config_path);

  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  keep_raw_points    = config_ros.param<bool>  ("glim_ros", "keep_raw_points",    false);
  imu_time_offset    = config_ros.param<double>("glim_ros", "imu_time_offset",    0.0);
  points_time_offset = config_ros.param<double>("glim_ros", "points_time_offset", 0.0);
  acc_scale          = config_ros.param<double>("glim_ros", "acc_scale",          0.0);

  glim::Config config_sensors(glim::GlobalConfig::get_config_path("config_sensors"));
  intensity_field = config_sensors.param<std::string>("sensors", "intensity_field", "intensity");
  ring_field      = config_sensors.param<std::string>("sensors", "ring_field",      "");

  dynamic_rejection_type = config_ros.param<std::string>("glim_ros", "dynamic_rejection_type", "NONE");
  lidar_frame_id_ = config_ros.param<std::string>("glim_ros", "lidar_frame_id", "velodyne");

  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ---------------------------------------------------------------------------
  // GPU linearization hook
  // ---------------------------------------------------------------------------
#ifdef BUILD_GTSAM_POINTS_GPU
  gtsam_points::LinearizationHook::register_hook(
      []() { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

  // ---------------------------------------------------------------------------
  // Preprocessing
  // ---------------------------------------------------------------------------
  time_keeper.reset(new glim::TimeKeeper);
  preprocessor.reset(new glim::CloudPreprocessor);

  // ---------------------------------------------------------------------------
  // Pose Kalman filter  (shared across IMU callback and timer callback)
  // ---------------------------------------------------------------------------
  pose_kalman_filter = std::make_shared<glim::PoseKalmanFilter>();
  last_imu_stamp_    = -1.0;
  

  // ---------------------------------------------------------------------------
  // Dynamic rejection — BBOX mode
  // ---------------------------------------------------------------------------
  spdlog::info("dynamic_rejection_type: {}", dynamic_rejection_type);
  if (dynamic_rejection_type == "BBOX") {
    spdlog::info("dynamic rejection: BBOX mode");
    dynamic_bbox_rejection = std::make_shared<glim::DynamicBBoxRejection>();
    glim::Config config_bbox(glim::GlobalConfig::get_config_path("config_bbox_rejection"));
    const int64_t bbox_min_track_age_default =
        config_bbox.param<int>("param_bbox_rejection", "bbox_min_track_age", 5);
    this->declare_parameter<int64_t>("bbox_min_track_age", bbox_min_track_age_default);
    this->declare_parameter<int64_t>(
        "bbox_potentially_dynamic_min_track_age",
        config_bbox.param<int>(
            "param_bbox_rejection",
            "bbox_potentially_dynamic_min_track_age",
            static_cast<int>(bbox_min_track_age_default)));
    this->declare_parameter<int64_t>(
        "bbox_static_min_track_age",
        config_bbox.param<int>("param_bbox_rejection", "bbox_static_min_track_age", 8));
    this->declare_parameter<double>(
        "bbox_min_static_velocity",
        config_bbox.param<double>("param_bbox_rejection", "bbox_min_static_velocity", 0.1));
  }

  // ---------------------------------------------------------------------------
  // Dynamic rejection — VOXEL mode
  //
  // Two objects are built independently and injected into the async wrapper:
  //   1. WallFilter            — owns voxelization + wall marking
  //   2. DynamicObjectRejectionCPU — owns voxel scoring + history
  // ---------------------------------------------------------------------------
  if (dynamic_rejection_type == "VOXEL") {
    spdlog::info("dynamic rejection: VOXEL mode");

    // Wall filter (reads its own config section internally)
    wall_bbox_registry_ = std::make_shared<glim::WallBBoxRegistry>();

    wall_filter = std::make_shared<glim::WallFilter>(
        glim::WallFilterConfig{}, wall_bbox_registry_, pose_kalman_filter);
    
    cluster_extractor = std::make_shared<glim::DynamicClusterExtractor>(pose_kalman_filter);
    // Dynamic scorer
    auto dyn_rejection = std::make_shared<glim::DynamicObjectRejectionCPU>(
        glim::DynamicObjectRejectionParamsCPU(),
        pose_kalman_filter);

    // Async wrapper owns both
    dynamic_object_rejection =
        std::make_shared<glim::AsyncDynamicObjectRejection>(dyn_rejection, wall_filter, cluster_extractor);
  }

  // ---------------------------------------------------------------------------
  // Odometry estimation
  // ---------------------------------------------------------------------------
  glim::Config config_odometry(glim::GlobalConfig::get_config_path("config_odometry"));
  const std::string odom_so =
      config_odometry.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");
  spdlog::info("load {}", odom_so);

  auto odom = OdometryEstimationBase::load_module(odom_so);
  if (!odom) {
    spdlog::critical("failed to load odometry estimation module");
    abort();
  }
  odometry_estimation.reset(new glim::AsyncOdometryEstimation(odom, odom->requires_imu()));

  // ---------------------------------------------------------------------------
  // Sub mapping
  // ---------------------------------------------------------------------------
  if (config_ros.param<bool>("glim_ros", "enable_local_mapping", true)) {
    const std::string so = glim::Config(glim::GlobalConfig::get_config_path("config_sub_mapping"))
                               .param<std::string>("sub_mapping", "so_name", "libsub_mapping.so");
    if (!so.empty()) {
      spdlog::info("load {}", so);
      auto sub = SubMappingBase::load_module(so);
      if (sub) sub_mapping.reset(new AsyncSubMapping(sub));
    }
  }

  // ---------------------------------------------------------------------------
  // Global mapping
  // ---------------------------------------------------------------------------
  if (config_ros.param<bool>("glim_ros", "enable_global_mapping", true)) {
    const std::string so = glim::Config(glim::GlobalConfig::get_config_path("config_global_mapping"))
                               .param<std::string>("global_mapping", "so_name", "libglobal_mapping.so");
    if (!so.empty()) {
      spdlog::info("load {}", so);
      auto global = GlobalMappingBase::load_module(so);
      if (global) global_mapping.reset(new AsyncGlobalMapping(global));
    }
  }

  // ---------------------------------------------------------------------------
  // Extension modules
  // ---------------------------------------------------------------------------
  const auto extensions = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules");
  if (extensions && !extensions->empty()) {
    for (const auto& ext : *extensions) {
      if (ext.find("viewer") == std::string::npos && ext.find("monitor") == std::string::npos) {
        spdlog::warn("Extension modules are enabled — check their licenses carefully");
        try {
          const std::string ext_cfg =
              ament_index_cpp::get_package_share_directory("glim_ext") + "/config";
          glim::GlobalConfig::instance()->override_param<std::string>("global", "config_ext", ext_cfg);
        } catch (ament_index_cpp::PackageNotFoundError&) {
          spdlog::warn("glim_ext package path not found");
        }
        break;
      }
    }
    for (const auto& ext : *extensions) {
      spdlog::info("load {}", ext);
      auto mod = ExtensionModule::load_module(ext);
      if (!mod) { spdlog::error("failed to load {}", ext); continue; }
      extension_modules.push_back(mod);
      auto mod_ros = std::dynamic_pointer_cast<ExtensionModuleROS2>(mod);
      if (mod_ros) {
        const auto subs = mod_ros->create_subscriptions(*this);
        extension_subs.insert(extension_subs.end(), subs.begin(), subs.end());
      }
    }
  }

  // ---------------------------------------------------------------------------
  // ROS subscribers and publishers
  // ---------------------------------------------------------------------------
  using std::placeholders::_1;

  const std::string imu_topic    = config_ros.param<std::string>("glim_ros", "imu_topic",    "");
  const std::string points_topic = config_ros.param<std::string>("glim_ros", "points_topic", "");
  const std::string image_topic  = config_ros.param<std::string>("glim_ros", "image_topic",  "");
  const std::string bbox_topic   = config_ros.param<std::string>("glim_ros", "bbox_topic",   "");
  rclcpp::SensorDataQoS default_imu_qos;
  default_imu_qos.get_rmw_qos_profile().depth = 1000;
  auto qos = get_qos_settings(config_ros, "glim_ros", "imu_qos", default_imu_qos);
  imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, qos, std::bind(&GlimROS::imu_callback, this, _1));

  qos = get_qos_settings(config_ros, "glim_ros", "points_qos");


  
  

  if (dynamic_rejection_type == "BBOX") {
    auto bbox_qos = get_qos_settings(config_ros, "glim_ros", "bbox_qos");
    bbox_sub = this->create_subscription<jo_msgs::msg::ObstacleArray>(
        bbox_topic, bbox_qos, std::bind(&GlimROS::bbox_callback, this, _1));

    spdlog::info("advertise ~/velodyne_points_dynamic, ~/bbox_markers, /velodyne_points_filtered");
    bbox_markers_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/bbox_markers", 10);

    // Raw-cloud filter: points_callback pushes {cloud, bboxes} to this thread
    // after reject() so the bboxes are guaranteed to be in sync with GLIM.
    raw_filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points_filtered", 10);
    raw_filter_running_ = true;
    raw_cloud_filter_thread_ = std::thread(&GlimROS::raw_cloud_filter_worker, this);
  }

  if (dynamic_rejection_type == "VOXEL") {
    spdlog::info("advertise ~/velodyne_points_dynamic, ~/voxelmap, ~/wall_points");
    
    voxelmap_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/voxelmap", 10);
    wall_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/wall_points", 10);
    wall_bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/wall_bboxes", 10);
    floor_bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/floor_bboxes", 10);
    dynamic_cluster_bboxes_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/cluster_bboxes", 10);
    cluster_history_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/cluster_history", 10);
  }

  // Ellipsoid markers are available in both BBOX and VOXEL modes.
  if (dynamic_rejection_type == "BBOX" || dynamic_rejection_type == "VOXEL") {
    inflate_params_ = glim::VelocityInflationParams::from_config();
    ellipsoid_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/bbox_ellipsoids", 10);

    dynamic_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/velodyne_points_dynamic", 10);
  }

  points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic, qos, std::bind(&GlimROS::points_callback, this, _1));
  spdlog::debug("subscribe to {} and {}", imu_topic, points_topic);
  
  filtered_pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/filtered_pose", 10);
  filtered_pose_marker_pub = this->create_publisher<visualization_msgs::msg::Marker>(
      "~/filtered_pose_marker", 10);

#ifdef BUILD_WITH_CV_BRIDGE
  qos = get_qos_settings(config_ros, "glim_ros", "image_qos");
  image_sub = image_transport::create_subscription(
      this, image_topic,
      std::bind(&GlimROS::image_callback, this, _1),
      "raw", qos.get_rmw_qos_profile());
#endif

  for (const auto& sub : this->extension_subscriptions()) {
    spdlog::debug("subscribe to {}", sub->topic);
    sub->create_subscriber(*this);
  }

  timer = this->create_wall_timer(
      std::chrono::milliseconds(1), [this]() { timer_callback(); });

  spdlog::debug("GlimROS initialized");
}

// =============================================================================
// Destructor
// =============================================================================

GlimROS::~GlimROS() {
  spdlog::debug("GlimROS shutting down");

  if (raw_filter_running_) {
    raw_filter_running_ = false;
    raw_queue_cv_.notify_all();
    if (raw_cloud_filter_thread_.joinable()) raw_cloud_filter_thread_.join();
  }

  extension_modules.clear();

  if (dump_on_unload) {
    wait(true);
    save("/tmp/dump");
  }
}

// =============================================================================
// extension_subscriptions()
// =============================================================================

const std::vector<std::shared_ptr<GenericTopicSubscription>>&
GlimROS::extension_subscriptions() {
  return extension_subs;
}

// =============================================================================
// imu_callback()
// =============================================================================

void GlimROS::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  spdlog::trace("IMU: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);

  if (!GlobalConfig::instance()->has_param("meta", "imu_frame_id")) {
    GlobalConfig::instance()->override_param<std::string>("meta", "imu_frame_id", msg->header.frame_id);
  }

  // Auto-detect acceleration unit
  if (std::abs(acc_scale) < 1e-6) {
    const double norm = Eigen::Vector3d(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z).norm();
    if      (norm > 7.0  && norm < 12.0) { acc_scale = 1.0;      }
    else if (norm > 0.8  && norm < 1.2)  { acc_scale = 9.80665;  }
    else                                  { acc_scale = 1.0;
      spdlog::warn("unexpected acc norm {:.2f}, assuming [m/s^2]", norm); }
  }

  const double          imu_stamp   = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9 + imu_time_offset;
  const Eigen::Vector3d linear_acc  = acc_scale * Eigen::Vector3d(
      msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d angular_vel(
      msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (!time_keeper->validate_imu_stamp(imu_stamp)) {
    spdlog::warn("skip invalid IMU stamp={}", imu_stamp);
    return;
  }

  odometry_estimation->insert_imu(imu_stamp, linear_acc, angular_vel);

  if (pose_kalman_filter && last_imu_stamp_ > 0.0) {
    const double dt = imu_stamp - last_imu_stamp_;
    if (dt > 0.0 && dt < 1.0) {
      std::lock_guard<std::mutex> lock(kf_imu_mutex_);
      kf_imu_queue_.push_back({linear_acc, angular_vel, dt});
    }
  }
  last_imu_stamp_ = imu_stamp;

  if (sub_mapping)    sub_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  if (global_mapping) global_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
}

// =============================================================================
// image_callback()
// =============================================================================

#ifdef BUILD_WITH_CV_BRIDGE
void GlimROS::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  spdlog::trace("image: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "image_frame")) {
    GlobalConfig::instance()->override_param<std::string>("meta", "image_frame", msg->header.frame_id);
  }
  auto cv_image = cv_bridge::toCvCopy(msg, "bgr8");
  const double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
  odometry_estimation->insert_image(stamp, cv_image->image);
  if (sub_mapping)    sub_mapping->insert_image(stamp, cv_image->image);
  if (global_mapping) global_mapping->insert_image(stamp, cv_image->image);
}
#endif

// =============================================================================
// points_callback()
// =============================================================================

size_t GlimROS::points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  spdlog::debug("points callback");

  if (!GlobalConfig::instance()->has_param("meta", "lidar_frame_id")) {
    GlobalConfig::instance()->override_param<std::string>("meta", "lidar_frame_id", msg->header.frame_id);
  }
  spdlog::debug("points: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);

  auto raw_points = glim::extract_raw_points(*msg, intensity_field, ring_field);
  if (!raw_points) {
    spdlog::warn("failed to extract points from message");
    return 0;
  }

  raw_points->stamp += points_time_offset;
  if (!time_keeper->process(raw_points)) {
    spdlog::warn("skip invalid point cloud (stamp={})", raw_points->stamp);
    return 0;
  }

  auto preprocessed = preprocessor->preprocess(raw_points);
  if (keep_raw_points) {
    preprocessed->raw_points = raw_points;
  }
  spdlog::debug("preprocessed: {} points", preprocessed->points.size());
  // ---------------------------------------------------------------------------
  // BBOX mode
  // ---------------------------------------------------------------------------
  if (dynamic_rejection_type == "BBOX") {
    auto filtered = dynamic_bbox_rejection->reject(preprocessed);
    spdlog::debug("BBOX filtered: {} → {} points",
        preprocessed->points.size(), filtered->points.size());

    odometry_estimation->insert_frame(filtered);

    // Push raw cloud + the bboxes just used by reject() to the filter thread.
    // Doing this here (after reject) guarantees temporal alignment with GLIM.
    if (raw_filter_running_ && !raw_bboxes_.empty()) {
      std::lock_guard<std::mutex> lock(raw_queue_mutex_);
      raw_cloud_queue_.push({msg, raw_bboxes_});
      raw_queue_cv_.notify_one();
    } else if (raw_filter_running_) {
      raw_filtered_pub_->publish(*msg);
    }

    auto dyn = dynamic_bbox_rejection->get_last_dynamic_frame();
    if (dyn && !dyn->points.empty()) {
      auto dyn_msg = glim_ros_utils::create_pointcloud2_msg(msg->header, dyn);
      dynamic_points_pub->publish(std::move(dyn_msg));
    }

  // ---------------------------------------------------------------------------
  // VOXEL mode
  // Pipeline: WallFilter → DynamicObjectRejectionCPU (both run on async thread)
  // ---------------------------------------------------------------------------
  } else if (dynamic_rejection_type == "VOXEL") {
    // Enqueue frame — WallFilter + rejection run on the background thread
    dynamic_object_rejection->insert_frame(preprocessed);

    // Drain static frames and feed to odometry
    for (const auto& filtered : dynamic_object_rejection->get_results()) {
      odometry_estimation->insert_frame(filtered);
    }

    // Publish latest voxelmap for visualization
    auto voxelmap = dynamic_object_rejection->get_last_voxelmap();
    if (voxelmap) {
      publish_voxelmap(msg->header, *voxelmap);
    }

    // Drain dynamic frames
    for (const auto& dyn : dynamic_object_rejection->get_dynamic_results()) {
      if (!dyn || dyn->points.empty()) continue;
      auto dyn_msg = glim_ros_utils::create_pointcloud2_msg(msg->header, dyn);
      dynamic_points_pub->publish(std::move(dyn_msg));
    }

    // Drain wall results and publish wall point cloud + floor bbox
    for (const auto& wf : dynamic_object_rejection->get_wall_results()) {
      if (wf.num_wall_voxels > 0) {
        publish_wall_voxelmap(msg->header, wf);
      }

      if (!wf.floor_bboxes.empty()) {
        visualization_msgs::msg::MarkerArray floor_marker_array;
        int id = 0;
        for (const auto& bbox : wf.floor_bboxes) {
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "velodyne";
          m.header.stamp    = msg->header.stamp;
          m.ns              = "floor_bbox";
          m.id              = id++;
          m.type            = visualization_msgs::msg::Marker::CUBE;
          m.action          = visualization_msgs::msg::Marker::ADD;
          m.lifetime        = rclcpp::Duration::from_seconds(0.5);
          const Eigen::Quaterniond q(bbox.get_rotation());
          m.pose.position.x    = bbox.get_center().x();
          m.pose.position.y    = bbox.get_center().y();
          m.pose.position.z    = bbox.get_center().z();
          m.pose.orientation.w = q.w();
          m.pose.orientation.x = q.x();
          m.pose.orientation.y = q.y();
          m.pose.orientation.z = q.z();
          m.scale.x = bbox.get_size().x();
          m.scale.y = bbox.get_size().y();
          m.scale.z = bbox.get_size().z();
          m.color.r = 0.2f; m.color.g = 0.8f; m.color.b = 0.2f; m.color.a = 0.35f;
          floor_marker_array.markers.push_back(m);
        }
        floor_bbox_pub_->publish(floor_marker_array);
      }
    }



    // ── Pubblica bounding box delle pareti dal registry ───────────────────────
    if (wall_bbox_registry_ && !wall_bbox_registry_->bboxes().empty()) {
      visualization_msgs::msg::MarkerArray wall_marker_array;
      
    
      const auto& bboxes = wall_bbox_registry_->bboxes();
      publish_bounding_boxes(
          msg->header, bboxes, "wall_bboxes",
          true, wall_bbox_pub_);
    }

    // Helper: costruisce un marker wireframe (LINE_LIST) da una BoundingBox
    auto make_bbox_marker = [&](const glim::BoundingBox& bbox,
                                const std::string& ns, int id,
                                float r, float g, float b, float a,
                                double line_width = 0.04)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "velodyne";
      marker.header.stamp    = this->now();
      marker.ns              = ns;
      marker.id              = id;
      marker.type            = visualization_msgs::msg::Marker::LINE_LIST;
      marker.action          = visualization_msgs::msg::Marker::ADD;
      marker.lifetime        = rclcpp::Duration::from_seconds(0.5);
      marker.scale.x         = line_width;
      marker.color.r = r; marker.color.g = g;
      marker.color.b = b; marker.color.a = a;

      const Eigen::Quaterniond q(bbox.get_rotation());
      marker.pose.position.x    = bbox.get_center().x();
      marker.pose.position.y    = bbox.get_center().y();
      marker.pose.position.z    = bbox.get_center().z();
      marker.pose.orientation.w = q.w();
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();

      const Eigen::Vector3d h = bbox.get_size() * 0.5;
      const std::array<Eigen::Vector3d, 8> v = {{
        {-h.x(),-h.y(),-h.z()}, { h.x(),-h.y(),-h.z()},
        { h.x(), h.y(),-h.z()}, {-h.x(), h.y(),-h.z()},
        {-h.x(),-h.y(), h.z()}, { h.x(),-h.y(), h.z()},
        { h.x(), h.y(), h.z()}, {-h.x(), h.y(), h.z()},
      }};
      const std::array<std::pair<int,int>, 12> edges = {{
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7},
      }};
      auto to_pt = [](const Eigen::Vector3d& p) {
        geometry_msgs::msg::Point pt;
        pt.x = p.x(); pt.y = p.y(); pt.z = p.z();
        return pt;
      };
      for (const auto& [a, b] : edges) {
        marker.points.push_back(to_pt(v[a]));
        marker.points.push_back(to_pt(v[b]));
      }
      return marker;
    };

    // Helper: TEXT_VIEW_FACING label above a bbox showing track id
    auto make_label_marker = [&](const glim::BoundingBox& bbox,
                                 const std::string& ns, int id,
                                 float r, float g, float b)
    {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "velodyne";
      m.header.stamp    = this->now();
      m.ns              = ns;
      m.id              = id;
      m.type            = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      m.action          = visualization_msgs::msg::Marker::ADD;
      m.lifetime        = rclcpp::Duration::from_seconds(0.5);
      m.scale.z         = 0.25;  // text height [m]
      m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0f;
      // Place label at the top face of the bbox
      m.pose.position.x = bbox.get_center().x();
      m.pose.position.y = bbox.get_center().y();
      m.pose.position.z = bbox.get_center().z() + bbox.get_size().z() * 0.5 + 0.15;
      m.pose.orientation.w = 1.0;
      const int tid = bbox.get_track_id();
      m.text = (tid >= 0) ? ("T" + std::to_string(tid)) : "?";
      return m;
    };

    // --- Bounding box dei cluster del frame corrente ---
    auto cluster_bbox_sets = dynamic_object_rejection->get_cluster_bbox_results();
    for (const auto& bboxes : cluster_bbox_sets) {
      visualization_msgs::msg::MarkerArray bbox_array;
      int id = 0;
      for (const auto& bbox : bboxes) {
        const bool dyn = bbox.is_dynamic_bbox();
        const float r = dyn ? 1.0f : 0.0f;
        const float g = dyn ? 0.5f : 1.0f;
        bbox_array.markers.push_back(make_bbox_marker(bbox, "cluster_current", id, r, g, 0.0f, 0.9f));
        bbox_array.markers.push_back(make_label_marker(bbox, "cluster_labels",  id, r, g, 0.0f));
        ++id;
      }
      dynamic_cluster_bboxes_pub->publish(bbox_array);
      spdlog::debug("[glim_ros] published {} cluster bboxes (current frame)", bboxes.size());

      // Ellipsoid markers for confirmed-dynamic clusters only.
      std::vector<glim::BoundingBox> dyn_bboxes;
      dyn_bboxes.reserve(bboxes.size());
      for (const auto& bbox : bboxes)
        if (bbox.is_dynamic_bbox()) dyn_bboxes.push_back(bbox);
      if (!dyn_bboxes.empty()) {
        std_msgs::msg::Header cluster_header;
        cluster_header.stamp    = this->now();
        cluster_header.frame_id = lidar_frame_id_;
        publish_ellipsoid_markers(cluster_header, dyn_bboxes);
      }
    }

    // --- Historia dei cluster: un colore per età del frame ---
    // age 0 = frame corrente (ciano), age N-1 = più vecchio (blu scuro, quasi trasparente)
    auto cluster_history_sets = dynamic_object_rejection->get_cluster_history_results();
    for (const auto& history : cluster_history_sets) {
      visualization_msgs::msg::MarkerArray history_array;
      int id = 0;
      const int num_frames = static_cast<int>(history.size());
      for (int age = 0; age < num_frames; ++age) {
        // Gradiente: age=0 → ciano (0,1,1), age=N-1 → blu (0,0,1)
        // Alpha: da 0.7 (recente) a 0.2 (vecchio)
        const float t     = (num_frames > 1) ? static_cast<float>(age) / (num_frames - 1) : 0.0f;
        const float green = 1.0f - t;      // 1→0
        const float alpha = 0.7f - 0.5f * t; // 0.7→0.2
        for (const auto& bbox : history[age]) {
          history_array.markers.push_back(
            make_bbox_marker(bbox, "cluster_history", id++, 0.0f, green, 1.0f, alpha, 0.03));
        }
      }
      cluster_history_pub->publish(history_array);
      spdlog::debug("[glim_ros] published cluster history ({} frames)", num_frames);
    }
    
  // ---------------------------------------------------------------------------
  // No dynamic rejection
  // ---------------------------------------------------------------------------
  } else {
    odometry_estimation->insert_frame(preprocessed);
  }
  Eigen::Isometry3d new_pose = pose_kalman_filter->getPose();
  Eigen::Isometry3d T_delta = new_pose * last_kf_pose_.inverse();
  
  // current_pose = T_delta * last_kf_pose_; // Applica la trasformazione delta al filtro di Kalman
  current_pose = new_pose; // Usa direttamente la nuova posa filtrata
  last_kf_pose_ = new_pose; // Aggiorna la posa del filtro di Kalman
  // Publish filtered pose as a marker
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = this->now();
  marker.ns = "filtered_pose";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  Eigen::Quaterniond q(current_pose.rotation());

  marker.scale.x = 0.05;
  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;

  const size_t MAX_POINTS = 1000;

  geometry_msgs::msg::Point p;
  p.x = current_pose.translation().x();
  p.y = current_pose.translation().y();
  p.z = current_pose.translation().z();

  traj_points_.push_back(p);

  if (traj_points_.size() > MAX_POINTS) {
      traj_points_.erase(traj_points_.begin());
  }
  marker.points = traj_points_;
  filtered_pose_marker_pub->publish(marker);
  spdlog::debug("published filtered pose marker with {} points", marker.points.size());
  return odometry_estimation->workload();
}

// =============================================================================
// publish_voxelmap()  —  private helper
// =============================================================================


void GlimROS::publish_bounding_boxes(
    const std_msgs::msg::Header& header,
    const std::vector<BoundingBox>& bboxes,
    const std::string& ns,
    bool wall,
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub)
{
  visualization_msgs::msg::MarkerArray marker_array;

  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto& bbox = bboxes[i];

    visualization_msgs::msg::Marker m;
    m.header.frame_id = header.frame_id;
    m.header.stamp    = this->now();
    m.ns              = ns;
    m.id              = static_cast<int>(i);
    m.type            = visualization_msgs::msg::Marker::CUBE;
    m.action          = visualization_msgs::msg::Marker::ADD;
    m.lifetime = rclcpp::Duration::from_seconds(0.2);

    // Posizione
    const Eigen::Vector3d& c = bbox.get_center();
    m.pose.position.x = c.x();
    m.pose.position.y = c.y();
    m.pose.position.z = c.z();

    // Orientamento
    const Eigen::Quaterniond q(bbox.get_rotation());
    m.pose.orientation.w = q.w();
    m.pose.orientation.x = q.x();
    m.pose.orientation.y = q.y();
    m.pose.orientation.z = q.z();

    // Dimensioni
    const Eigen::Vector3d& s = bbox.get_size();
    m.scale.x = s.x();
    m.scale.y = s.y();
    m.scale.z = s.z();

    // Colore
    if (wall) {
      m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 0.3f; // green for walls
    } else {
      m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 0.9f; // Red for dynamic clusters
    }

    marker_array.markers.push_back(m);
  }

  pub->publish(marker_array);
}

void GlimROS::publish_ellipsoid_markers(
    const std_msgs::msg::Header& header,
    const std::vector<BoundingBox>& bboxes)
{
  if (!ellipsoid_markers_pub_) return;

  visualization_msgs::msg::MarkerArray arr;
  int id = 0;

  for (const auto& bbox : bboxes) {
    const double speed_xy  = bbox.get_speed_xy();
    const double eff_speed =
      speed_xy > inflate_params_.v_min ? std::min(speed_xy, inflate_params_.v_max_speed) : 0.0;
    const double hx        = bbox.get_size().x() * 0.5;
    const double hy        = bbox.get_size().y() * 0.5;
    const double hz        = std::max(bbox.get_size().z() * 0.5, 1e-6);
    const double cover_scale = std::max(inflate_params_.ellipse_box_cover_scale, 1.0);

    // Quaternion that rotates X-axis to align with velocity heading
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d heading = Eigen::Vector3d::UnitX();
    if (speed_xy > inflate_params_.v_min) {
      const Eigen::Vector3d& vel = bbox.get_velocity();
      const double yaw = std::atan2(vel.y(), vel.x());
      q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
      heading = Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0.0);
    }
    const Eigen::Vector3d lateral(-heading.y(), heading.x(), 0.0);
    const Eigen::Vector3d box_x_axis = bbox.get_rotation().col(0);
    const Eigen::Vector3d box_y_axis = bbox.get_rotation().col(1);
    const double bbox_half_u =
      std::abs(heading.dot(box_x_axis)) * hx +
      std::abs(heading.dot(box_y_axis)) * hy;
    const double bbox_half_v =
      std::abs(lateral.dot(box_x_axis)) * hx +
      std::abs(lateral.dot(box_y_axis)) * hy;
    const double semi_u = cover_scale * bbox_half_u +
      0.5 * eff_speed * (inflate_params_.v_fwd_k + inflate_params_.v_rear_k);
    const double semi_lat = cover_scale * bbox_half_v + eff_speed * inflate_params_.v_lat_k;

    const Eigen::Vector3d& c = bbox.get_center();
    const Eigen::Vector3d egg_center =
      c + heading * (0.5 * eff_speed * (inflate_params_.v_fwd_k - inflate_params_.v_rear_k));

    // 2D velocity egg footprint, extruded only to the inflated bbox height.
    {
      visualization_msgs::msg::Marker m;
      m.header       = header;
      m.header.stamp = this->now();
      m.ns           = "bbox_ellipsoid";
      m.id           = id++;
      m.type         = visualization_msgs::msg::Marker::SPHERE;
      m.action       = visualization_msgs::msg::Marker::ADD;
      m.lifetime     = rclcpp::Duration::from_seconds(0.2);
      m.pose.position.x    = egg_center.x();
      m.pose.position.y    = egg_center.y();
      m.pose.position.z    = c.z();
      m.pose.orientation.w = q.w();
      m.pose.orientation.x = q.x();
      m.pose.orientation.y = q.y();
      m.pose.orientation.z = q.z();
      m.scale.x = 2.0 * semi_u;
      m.scale.y = 2.0 * semi_lat;
      m.scale.z = 2.0 * hz;
      m.color.r = 1.0f; m.color.g = 0.5f; m.color.b = 0.0f; m.color.a = 0.30f;
      arr.markers.push_back(m);
    }

    // Velocity ARROW (yellow) — shows heading and speed
    if (speed_xy > inflate_params_.v_min) {
      visualization_msgs::msg::Marker arrow;
      arrow.header       = header;
      arrow.header.stamp = this->now();
      arrow.ns           = "bbox_ellipsoid_vel";
      arrow.id           = id++;
      arrow.type         = visualization_msgs::msg::Marker::ARROW;
      arrow.action       = visualization_msgs::msg::Marker::ADD;
      arrow.lifetime     = rclcpp::Duration::from_seconds(0.2);
      const Eigen::Vector3d& vel = bbox.get_velocity();
      geometry_msgs::msg::Point tail, tip;
      tail.x = c.x(); tail.y = c.y(); tail.z = c.z();
      tip.x  = c.x() + vel.x(); tip.y = c.y() + vel.y(); tip.z = c.z() + vel.z();
      arrow.points.push_back(tail);
      arrow.points.push_back(tip);
      arrow.scale.x = 0.08;  // shaft diameter
      arrow.scale.y = 0.15;  // head diameter
      arrow.scale.z = 0.20;  // head length
      arrow.color.r = 1.0f; arrow.color.g = 1.0f; arrow.color.b = 0.0f; arrow.color.a = 0.9f;
      arr.markers.push_back(arrow);
    }
  }

  ellipsoid_markers_pub_->publish(arr);
}

void GlimROS::publish_voxelmap(
    const std_msgs::msg::Header&              header,
    const gtsam_points::DynamicVoxelMapCPU&   voxelmap)
{
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = header.frame_id;
  marker.header.stamp    = this->now();
  marker.ns              = "voxelmap";
  marker.id              = 0;
  marker.type            = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action          = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = marker.scale.y = marker.scale.z = voxelmap.voxel_resolution();

  const int nvox = static_cast<int>(
      voxelmap.gtsam_points::IncrementalVoxelMap<
          gtsam_points::DynamicGaussianVoxel>::num_voxels());

  marker.points.reserve(nvox);
  marker.colors.reserve(nvox);

  for (int i = 0; i < nvox; ++i) {
    const auto& v = voxelmap.lookup_voxel(i);

    geometry_msgs::msg::Point p;
    p.x = v.mean.x(); p.y = v.mean.y(); p.z = v.mean.z();
    marker.points.push_back(p);

    std_msgs::msg::ColorRGBA c;
    c.a = 0.6f;
    if (v.is_wall) {
      // Walls: yellow
      c.r = 1.0f; c.g = 1.0f; c.b = 0.0f;
    } else if (v.is_ground) {
      // Ground: green
      c.r = 0.0f; c.g = 1.0f; c.b = 0.0f;
    } else if (v.is_dynamic) {
      // Dynamic: red
      c.r = 1.0f; c.g = 0.0f; c.b = 0.0f;
    } else if (v.is_outlier) {
      // Outlier: gray
      c.r = 0.5f; c.g = 0.5f; c.b = 0.5f;
    } else {
      // Static: blue
      c.r = 0.0f; c.g = 0.0f; c.b = 1.0f;
    }
    marker.colors.push_back(c);
  }

  visualization_msgs::msg::MarkerArray arr;
  arr.markers.push_back(std::move(marker));
  voxelmap_pub->publish(arr);
}

// =============================================================================
// publish_wall_voxelmap()
// =============================================================================

void GlimROS::publish_wall_voxelmap(
    const std_msgs::msg::Header& header,
    const WallFilterResult&      wf_result)
{
  if (!wf_result.voxelmap) return;

  // Collect the raw points that belong to wall voxels (is_wall == true).
  // Publishing individual points (rather than just centroids) gives a denser,
  // more informative cloud for visualization and debugging.
  std::vector<Eigen::Vector4d> wall_pts;

  const int nvox = static_cast<int>(
      wf_result.voxelmap->gtsam_points::IncrementalVoxelMap<
          gtsam_points::DynamicGaussianVoxel>::num_voxels());

  for (int i = 0; i < nvox; ++i) {
    const auto& v = wf_result.voxelmap->lookup_voxel(i);
    if (!v.is_wall) continue;
    wall_pts.insert(wall_pts.end(),
                    v.voxel_points.begin(), v.voxel_points.end());
  }

  if (wall_pts.empty()) return;

  // Build a minimal PreprocessedFrame to reuse create_pointcloud2_msg()
  auto wall_frame           = std::make_shared<PreprocessedFrame>();
  wall_frame->stamp         = header.stamp.sec + header.stamp.nanosec / 1e9;
  wall_frame->scan_end_time = wall_frame->stamp;
  wall_frame->points        = std::move(wall_pts);
  wall_frame->k_neighbors   = 0;

  auto cloud_msg = glim_ros_utils::create_pointcloud2_msg(header, wall_frame);
  wall_points_pub->publish(std::move(cloud_msg));

  spdlog::debug("[wall] published {} wall points ({} wall voxels / {} total)",
      wall_frame->points.size(), wf_result.num_wall_voxels, wf_result.num_total_voxels);
}
// =============================================================================

void GlimROS::bbox_callback(
    const jo_msgs::msg::ObstacleArray::ConstSharedPtr msg)
{
  if (dynamic_rejection_type != "BBOX") {
    spdlog::debug("received bbox message but dynamic_rejection_type != BBOX");
    return;
  }
  spdlog::info("bbox callback: received {} obstacles", msg->obstacles.size());
  // Look up transform from obstacle frame (odom) → lidar frame
  Eigen::Isometry3d T_odom_base_link = pose_kalman_filter->getPose();
  Eigen::Isometry3d T_base_odom = T_odom_base_link.inverse();
  {
    std::ostringstream oss;
    oss << T_odom_base_link.matrix();
    spdlog::debug("bbox_callback: T_odom_base_link = \n{}", oss.str());
  }
  const std::string& src_frame = msg->header.frame_id;

  // Step 1 — static part: base_link → velodyne (cache on first successful lookup)
  if (!T_velodyne_base_valid) {
    try {
      const auto tf = tf_buffer_->lookupTransform(lidar_frame_id_, "base_link", tf2::TimePointZero);
      T_velodyne_base_ = tf2::transformToEigen(tf);
      T_velodyne_base_valid = true;
    } catch (const tf2::TransformException& e) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
          "bbox_callback: static TF base_link → %s not yet available: %s", lidar_frame_id_.c_str(), e.what());
      return;
    }
  }
  {
    std::ostringstream oss;
    oss << T_velodyne_base_.matrix();
    spdlog::debug("bbox_callback: T_velodyne_base_ = \n{}", oss.str());
  }
  Eigen::Isometry3d T_velodyne_odom = T_velodyne_base_ * T_base_odom;

  const int64_t potentially_dynamic_min_track_age =
      this->get_parameter("bbox_potentially_dynamic_min_track_age").as_int();
  const int64_t static_min_track_age =
      this->get_parameter("bbox_static_min_track_age").as_int();
  const double min_static_vel = this->get_parameter("bbox_min_static_velocity").as_double();

  using Obs = jo_msgs::msg::Obstacle;

  std::vector<BoundingBox> bboxes;
  for (const auto& obs : msg->obstacles) {
    // Condition 1: classified dynamic
    bool should_filter = (obs.status == Obs::STATUS_DYNAMIC);

    // Condition 2: potentially dynamic with enough track history
    if (!should_filter &&
        obs.status == Obs::STATUS_POTENTIALLY_DYNAMIC &&
        static_cast<int64_t>(obs.track_age) > potentially_dynamic_min_track_age) {
      should_filter = true;
    }

    // Condition 3: static label but still moving (age gate + velocity gate)
    if (!should_filter &&
        obs.status == Obs::STATUS_STATIC &&
        static_cast<int64_t>(obs.track_age) > static_min_track_age) {
      const double v = std::sqrt(
          obs.twist.linear.x * obs.twist.linear.x +
          obs.twist.linear.y * obs.twist.linear.y +
          obs.twist.linear.z * obs.twist.linear.z);
      if (v > min_static_vel) {
        should_filter = true;
      }
    }

    if (!should_filter) continue;

    const Eigen::Vector3d p_odom(obs.pose.position.x, obs.pose.position.y, obs.pose.position.z);
    const Eigen::Quaterniond q_odom(obs.pose.orientation.w, obs.pose.orientation.x,
                                    obs.pose.orientation.y, obs.pose.orientation.z);
    const Eigen::Vector3d    p_lidar = T_velodyne_odom * p_odom;
    const Eigen::Matrix3d    R_lidar = T_velodyne_odom.rotation() * q_odom.toRotationMatrix();

    bboxes.emplace_back(
        Eigen::Vector3d(obs.size.x, obs.size.y, obs.size.z),
        p_lidar,
        R_lidar);
    const Eigen::Vector3d vel_odom(obs.twist.linear.x, obs.twist.linear.y, obs.twist.linear.z);
    bboxes.back().set_velocity(T_velodyne_odom.rotation() * vel_odom);
    dynamic_bbox_rejection->insert_bounding_boxes(bboxes.back());
  }

  // Keep raw_bboxes_ in sync — same executor thread as points_callback, no mutex needed.
  raw_bboxes_ = dynamic_bbox_rejection->get_bounding_boxes();

  std_msgs::msg::Header lidar_header;
  lidar_header.stamp    = msg->header.stamp;
  lidar_header.frame_id = lidar_frame_id_;
  publish_bounding_boxes(lidar_header, raw_bboxes_, "bbox_input", false, bbox_markers_pub);
  publish_ellipsoid_markers(lidar_header, raw_bboxes_);
}

// =============================================================================
// timer_callback()
// =============================================================================

void GlimROS::timer_callback() {
  for (const auto& mod : extension_modules) {
    if (!mod->ok()) { rclcpp::shutdown(); }
  }
  spdlog::debug("timer callback: check odometry estimation results and publish");
  std::vector<glim::EstimationFrame::ConstPtr> estimation_frames;
  std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
  odometry_estimation->get_results(estimation_frames, marginalized_frames);

  // Kalman filter update on latest SLAM pose
  if (!estimation_frames.empty() && pose_kalman_filter) {
    {
      std::lock_guard<std::mutex> lock(kf_imu_mutex_);
      for (const auto& imu : kf_imu_queue_) {
        glim::ImuMeasurement m{imu.acc, imu.gyro, imu.dt};
        pose_kalman_filter->predict(m);
      }
      kf_imu_queue_.clear();
    }
    
    const auto& latest = estimation_frames.back();
    if (latest) {
      const Eigen::Isometry3d T_filtered =
          pose_kalman_filter->update(latest->T_world_imu);

      auto pose_msg          = std::make_unique<geometry_msgs::msg::PoseStamped>();
      pose_msg->header.stamp    = rclcpp::Time(static_cast<int64_t>(latest->stamp * 1e9));
      pose_msg->header.frame_id = "map";

      const Eigen::Vector3d    p = T_filtered.translation();
      const Eigen::Quaterniond q(T_filtered.rotation());
      pose_msg->pose.position.x    = p.x();
      pose_msg->pose.position.y    = p.y();
      pose_msg->pose.position.z    = p.z();
      pose_msg->pose.orientation.w = q.w();
      pose_msg->pose.orientation.x = q.x();
      pose_msg->pose.orientation.y = q.y();
      pose_msg->pose.orientation.z = q.z();

      filtered_pose_pub->publish(std::move(pose_msg));
      spdlog::debug("[KF] published filtered pose");
    }
  }

  // Forward marginalized frames to sub / global mapping
  if (sub_mapping) {
    for (const auto& frame : marginalized_frames) {
      sub_mapping->insert_frame(frame);
    }
    auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }
    }
  }
  spdlog::debug("timer callback done");
}

// =============================================================================
// needs_wait()
// =============================================================================

bool GlimROS::needs_wait() {
  for (const auto& mod : extension_modules) {
    if (mod->needs_wait()) return true;
  }
  return false;
}

// =============================================================================
// wait()
// =============================================================================

void GlimROS::wait(bool auto_quit) {
  spdlog::info("waiting for odometry estimation");
  odometry_estimation->join();

  if (sub_mapping) {
    std::vector<glim::EstimationFrame::ConstPtr> est, marg;
    odometry_estimation->get_results(est, marg);
    for (const auto& f : marg) sub_mapping->insert_frame(f);

    spdlog::info("waiting for local mapping");
    sub_mapping->join();

    const auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& s : submaps) global_mapping->insert_submap(s);
      spdlog::info("waiting for global mapping");
      global_mapping->join();
    }
  }

  if (!auto_quit) {
    bool terminate = false;
    while (!terminate && rclcpp::ok()) {
      for (const auto& mod : extension_modules) terminate |= !mod->ok();
    }
  }
}

// =============================================================================
// save()
// =============================================================================

void GlimROS::save(const std::string& path) {
  if (global_mapping) global_mapping->save(path);
  for (auto& mod : extension_modules) mod->at_exit(path);
}

// =============================================================================
// raw_cloud_filter_worker()
// =============================================================================

void GlimROS::raw_cloud_filter_worker()
{
  while (raw_filter_running_) {
    FilterJob job;
    {
      std::unique_lock<std::mutex> lock(raw_queue_mutex_);
      raw_queue_cv_.wait(lock, [this] {
        return !raw_cloud_queue_.empty() || !raw_filter_running_;
      });
      if (!raw_filter_running_ && raw_cloud_queue_.empty()) break;
      job = std::move(raw_cloud_queue_.front());
      raw_cloud_queue_.pop();
    }

    raw_filtered_pub_->publish(apply_ellipsoid_filter(*job.cloud, job.bboxes));
  }
}

// =============================================================================
// apply_ellipsoid_filter()
// Bboxes are already in lidar frame — no TF transform needed.
// =============================================================================

sensor_msgs::msg::PointCloud2 GlimROS::apply_ellipsoid_filter(
    const sensor_msgs::msg::PointCloud2& cloud,
    const std::vector<BoundingBox>& bboxes) const
{
  int x_off = -1, y_off = -1, z_off = -1;
  for (const auto& f : cloud.fields) {
    if      (f.name == "x") x_off = static_cast<int>(f.offset);
    else if (f.name == "y") y_off = static_cast<int>(f.offset);
    else if (f.name == "z") z_off = static_cast<int>(f.offset);
  }
  if (x_off < 0 || y_off < 0 || z_off < 0) return cloud;

  const uint32_t step     = cloud.point_step;
  const uint32_t n_points = cloud.width * cloud.height;
  const uint8_t* raw      = cloud.data.data();

  std::vector<uint8_t> kept;
  kept.reserve(n_points * step);

  for (uint32_t i = 0; i < n_points; ++i) {
    const uint8_t* base = raw + i * step;

    float px, py, pz;
    std::memcpy(&px, base + x_off, sizeof(float));
    std::memcpy(&py, base + y_off, sizeof(float));
    std::memcpy(&pz, base + z_off, sizeof(float));

    const Eigen::Vector4d p4(px, py, pz, 1.0);
    bool inside = false;
    for (const auto& bbox : bboxes) {
      if (bbox.contains_inflated(p4, inflate_params_)) {
        inside = true;
        break;
      }
    }

    if (!inside) kept.insert(kept.end(), base, base + step);
  }

  sensor_msgs::msg::PointCloud2 out;
  out.header       = cloud.header;
  out.height       = 1;
  out.width        = static_cast<uint32_t>(kept.size() / step);
  out.fields       = cloud.fields;
  out.is_bigendian = cloud.is_bigendian;
  out.point_step   = step;
  out.row_step     = static_cast<uint32_t>(kept.size());
  out.data         = std::move(kept);
  out.is_dense     = false;
  return out;
}

}  // namespace glim

RCLCPP_COMPONENTS_REGISTER_NODE(glim::GlimROS)
