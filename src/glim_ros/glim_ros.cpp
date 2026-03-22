#include <glim_ros/glim_ros.hpp>

#define GLIM_ROS2

#include <deque>
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
  if (dynamic_rejection_type == "BBOX") {
    spdlog::info("dynamic rejection: BBOX mode");
    dynamic_bbox_rejection = std::make_shared<glim::DynamicBBoxRejection>();
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
    wall_filter = std::make_shared<glim::WallFilter>(glim::WallFilterConfig{}, wall_bbox_registry_);
    
    cluster_extractor = std::make_shared<glim::DynamicClusterExtractor>();
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
  points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic, qos, std::bind(&GlimROS::points_callback, this, _1));

  if (dynamic_rejection_type == "BBOX") {
    auto bbox_qos = get_qos_settings(config_ros, "glim_ros", "bbox_qos");
    bbox_sub = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        bbox_topic, bbox_qos, std::bind(&GlimROS::bbox_callback, this, _1));

    spdlog::info("advertise ~/filtered_points_bbox and ~/dynamic_points_bbox");
    filtered_points_bbox_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/filtered_points_bbox", 10);
    dynamic_points_bbox_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/dynamic_points_bbox", 10);
  }

  if (dynamic_rejection_type == "VOXEL") {
    spdlog::info("advertise ~/filtered_points_voxel, ~/dynamic_points_voxel, ~/voxelmap, ~/wall_points");
    filtered_points_voxel_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/filtered_points_voxel", 10);
    dynamic_points_voxel_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/dynamic_points_voxel", 10);
    voxelmap_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/voxelmap", 10);
    wall_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "~/wall_points", 10);
    wall_bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/wall_bboxes", 10);
    dynamic_cluster_bboxes_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/cluster_bboxes", 10);

  }

  filtered_pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/filtered_pose", 10);

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

  // ---------------------------------------------------------------------------
  // BBOX mode
  // ---------------------------------------------------------------------------
  if (dynamic_rejection_type == "BBOX") {
    auto filtered = dynamic_bbox_rejection->reject(preprocessed);
    spdlog::debug("BBOX filtered: {} → {} points",
        preprocessed->points.size(), filtered->points.size());

    auto filtered_msg = glim_ros_utils::create_pointcloud2_msg(msg->header, filtered);
    filtered_points_bbox_pub->publish(std::move(filtered_msg));

    odometry_estimation->insert_frame(filtered);

    auto dyn = dynamic_bbox_rejection->get_last_dynamic_frame();
    if (dyn && !dyn->points.empty()) {
      auto dyn_msg = glim_ros_utils::create_pointcloud2_msg(msg->header, dyn);
      dynamic_points_bbox_pub->publish(std::move(dyn_msg));
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
      auto filtered_msg = glim_ros_utils::create_pointcloud2_msg(msg->header, filtered);
      filtered_points_voxel_pub->publish(std::move(filtered_msg));
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
      dynamic_points_voxel_pub->publish(std::move(dyn_msg));
    }

    // Drain wall results and publish wall point cloud
    for (const auto& wf : dynamic_object_rejection->get_wall_results()) {
      if (wf.num_wall_voxels > 0) {
        publish_wall_voxelmap(msg->header, wf);
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

    // publish bounding box of dynamic clusters
    
     // --- Bounding box dei cluster dinamici ---
    auto cluster_bbox_sets = dynamic_object_rejection->get_cluster_bbox_results();
    
    for (const auto& bboxes : cluster_bbox_sets) {
      visualization_msgs::msg::MarkerArray bbox_array;

      int marker_id = 0;
      for (const auto& bbox : bboxes) {
        // --- Cubo wireframe (LINE_LIST sui 12 spigoli) ---
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "velodyne";
        marker.header.stamp    = this->now();
        marker.ns              = "dynamic_clusters";
        marker.id              = marker_id++;
        marker.type            = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action          = visualization_msgs::msg::Marker::ADD;
        marker.lifetime        = rclcpp::Duration::from_seconds(0.2);

        // Spessore linea [m]
        marker.scale.x = 0.05;

        // Colore: rosso semitrasparente
        marker.color.r = 1.0f;
        marker.color.g = 0.2f;
        marker.color.b = 0.0f;
        marker.color.a = 0.9f;

        // Orientamento (quaternione dalla matrice di rotazione OBB)
        const Eigen::Quaterniond q(bbox.get_rotation());
        marker.pose.position.x    = bbox.get_center().x();
        marker.pose.position.y    = bbox.get_center().y();
        marker.pose.position.z    = bbox.get_center().z();
        marker.pose.orientation.w = q.w();
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();

        // I 12 spigoli del cubo in coordinate locali (half-extents)
        const Eigen::Vector3d h = bbox.get_size() * 0.5;

        // 8 vertici in coordinate locali
        const std::array<Eigen::Vector3d, 8> v = {{
          { -h.x(), -h.y(), -h.z() },
          {  h.x(), -h.y(), -h.z() },
          {  h.x(),  h.y(), -h.z() },
          { -h.x(),  h.y(), -h.z() },
          { -h.x(), -h.y(),  h.z() },
          {  h.x(), -h.y(),  h.z() },
          {  h.x(),  h.y(),  h.z() },
          { -h.x(),  h.y(),  h.z() },
        }};

        // 12 spigoli (coppie di indici)
        const std::array<std::pair<int,int>, 12> edges = {{
          {0,1},{1,2},{2,3},{3,0},  // faccia inferiore
          {4,5},{5,6},{6,7},{7,4},  // faccia superiore
          {0,4},{1,5},{2,6},{3,7},  // spigoli verticali
        }};

        auto to_point = [](const Eigen::Vector3d& p) {
          geometry_msgs::msg::Point pt;
          pt.x = p.x(); pt.y = p.y(); pt.z = p.z();
          return pt;
        };

        for (const auto& [a, b] : edges) {
          marker.points.push_back(to_point(v[a]));
          marker.points.push_back(to_point(v[b]));
        }

        bbox_array.markers.push_back(marker);
      }

      dynamic_cluster_bboxes_pub->publish(bbox_array);
      spdlog::info("[glim_ros] published {} dynamic cluster bboxes", bboxes.size());
    }
    
    
  // ---------------------------------------------------------------------------
  // No dynamic rejection
  // ---------------------------------------------------------------------------
  } else {
    odometry_estimation->insert_frame(preprocessed);
  }

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
    } else if (v.is_dynamic) {
      // Dynamic: red
      c.r = 1.0f; c.g = 0.0f; c.b = 0.0f;
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
    const visualization_msgs::msg::MarkerArray::ConstSharedPtr msg)
{
  if (dynamic_rejection_type != "BBOX") {
    spdlog::warn("received bbox message but dynamic_rejection_type != BBOX");
    return;
  }

  for (const auto& marker : msg->markers) {
    if (marker.type != visualization_msgs::msg::Marker::CUBE) continue;
    BoundingBox bbox(
        Eigen::Vector3d(marker.scale.x, marker.scale.y, marker.scale.z),
        Eigen::Vector3d(marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z),
        Eigen::Quaterniond(marker.pose.orientation.w,
                           marker.pose.orientation.x,
                           marker.pose.orientation.y,
                           marker.pose.orientation.z).toRotationMatrix());
    dynamic_bbox_rejection->insert_bounding_boxes(bbox);
  }
}

// =============================================================================
// timer_callback()
// =============================================================================

void GlimROS::timer_callback() {
  for (const auto& mod : extension_modules) {
    if (!mod->ok()) { rclcpp::shutdown(); }
  }

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

}  // namespace glim

RCLCPP_COMPONENTS_REGISTER_NODE(glim::GlimROS)