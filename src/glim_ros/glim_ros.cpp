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
#ifdef GLIM_USE_DYNAMIC_REJECTION
#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>
#endif
#include <glim_ros/ros_compatibility.hpp>
#include <glim_ros/ros_qos.hpp>

namespace glim {

GlimROS::GlimROS(const rclcpp::NodeOptions& options) : Node("glim_ros", options) {
  // Setup logger
  auto logger = spdlog::stdout_color_mt("glim");
  logger->sinks().push_back(get_ringbuffer_sink());
  spdlog::set_default_logger(logger);

  bool debug = false;
  this->declare_parameter<bool>("debug", false);
  this->get_parameter<bool>("debug", debug);

  if (debug) {
    spdlog::info("enable debug printing");
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("/tmp/glim_log.log", true);
    logger->sinks().push_back(file_sink);
    logger->set_level(spdlog::level::trace);

    print_system_info(logger);
  }

  dump_on_unload = false;
  this->declare_parameter<bool>("dump_on_unload", false);
  this->get_parameter<bool>("dump_on_unload", dump_on_unload);

  if (dump_on_unload) {
    spdlog::info("dump_on_unload={}", dump_on_unload);
  }

  std::string config_path;
  this->declare_parameter<std::string>("config_path", "config");
  this->get_parameter<std::string>("config_path", config_path);

  if (config_path.empty() || config_path[0] != '/') {
    // config_path is relative to the glim directory
    config_path = ament_index_cpp::get_package_share_directory("glim") + "/" + config_path;
  }

  logger->info("config_path: {}", config_path);
  glim::GlobalConfig::instance(config_path);
  glim::Config config_ros(glim::GlobalConfig::get_config_path("config_ros"));

  keep_raw_points = config_ros.param<bool>("glim_ros", "keep_raw_points", false);
  imu_time_offset = config_ros.param<double>("glim_ros", "imu_time_offset", 0.0);
  points_time_offset = config_ros.param<double>("glim_ros", "points_time_offset", 0.0);
  acc_scale = config_ros.param<double>("glim_ros", "acc_scale", 0.0);

  glim::Config config_sensors(glim::GlobalConfig::get_config_path("config_sensors"));
  intensity_field = config_sensors.param<std::string>("sensors", "intensity_field", "intensity");
  ring_field = config_sensors.param<std::string>("sensors", "ring_field", "");

  // Setup GPU-based linearization
#ifdef BUILD_GTSAM_POINTS_GPU
  gtsam_points::LinearizationHook::register_hook([]() { return gtsam_points::create_nonlinear_factor_set_gpu(); });
#endif

  // Preprocessing
  time_keeper.reset(new glim::TimeKeeper);
  preprocessor.reset(new glim::CloudPreprocessor);
  // Pose Kalman filter (shared with dynamic object rejection)
  pose_kalman_filter = std::make_shared<glim::PoseKalmanFilter>();
  last_imu_stamp_ = -1.0;

#ifdef GLIM_USE_DYNAMIC_REJECTION
  // Dynamic object rejection (async)
  spdlog::info("enable dynamic object rejection");
  auto dyn_rejection = std::make_shared<glim::DynamicObjectRejectionCPU>(glim::DynamicObjectRejectionParamsCPU(), pose_kalman_filter);
  dynamic_object_rejection = std::make_shared<glim::AsyncDynamicObjectRejection>(dyn_rejection);
#endif

  // Odometry estimation
  glim::Config config_odometry(glim::GlobalConfig::get_config_path("config_odometry"));
  const std::string odometry_estimation_so_name = config_odometry.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_cpu.so");
  spdlog::info("load {}", odometry_estimation_so_name);

  std::shared_ptr<glim::OdometryEstimationBase> odom = OdometryEstimationBase::load_module(odometry_estimation_so_name);
  if (!odom) {
    spdlog::critical("failed to load odometry estimation module");
    abort();
  }
  odometry_estimation.reset(new glim::AsyncOdometryEstimation(odom, odom->requires_imu()));

  // Sub mapping
  if (config_ros.param<bool>("glim_ros", "enable_local_mapping", true)) {
    const std::string sub_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_sub_mapping")).param<std::string>("sub_mapping", "so_name", "libsub_mapping.so");
    if (!sub_mapping_so_name.empty()) {
      spdlog::info("load {}", sub_mapping_so_name);
      auto sub = SubMappingBase::load_module(sub_mapping_so_name);
      if (sub) {
        sub_mapping.reset(new AsyncSubMapping(sub));
      }
    }
  }

  // Global mapping
  if (config_ros.param<bool>("glim_ros", "enable_global_mapping", true)) {
    const std::string global_mapping_so_name =
      glim::Config(glim::GlobalConfig::get_config_path("config_global_mapping")).param<std::string>("global_mapping", "so_name", "libglobal_mapping.so");
    if (!global_mapping_so_name.empty()) {
      spdlog::info("load {}", global_mapping_so_name);
      auto global = GlobalMappingBase::load_module(global_mapping_so_name);
      if (global) {
        global_mapping.reset(new AsyncGlobalMapping(global));
      }
    }
  }

  // Extention modules
  const auto extensions = config_ros.param<std::vector<std::string>>("glim_ros", "extension_modules");
  if (extensions && !extensions->empty()) {
    for (const auto& extension : *extensions) {
      if (extension.find("viewer") == std::string::npos && extension.find("monitor") == std::string::npos) {
        spdlog::warn("Extension modules are enabled!!");
        spdlog::warn("You must carefully check and follow the licenses of ext modules");

        try {
          const std::string config_ext_path = ament_index_cpp::get_package_share_directory("glim_ext") + "/config";
          spdlog::info("config_ext_path: {}", config_ext_path);
          glim::GlobalConfig::instance()->override_param<std::string>("global", "config_ext", config_ext_path);
        } catch (ament_index_cpp::PackageNotFoundError& e) {
          spdlog::warn("glim_ext package path was not found!!");
        }

        break;
      }
    }

    for (const auto& extension : *extensions) {
      spdlog::info("load {}", extension);
      auto ext_module = ExtensionModule::load_module(extension);
      if (ext_module == nullptr) {
        spdlog::error("failed to load {}", extension);
        continue;
      } else {
        extension_modules.push_back(ext_module);

        auto ext_module_ros = std::dynamic_pointer_cast<ExtensionModuleROS2>(ext_module);
        if (ext_module_ros) {
          const auto subs = ext_module_ros->create_subscriptions(*this);
          extension_subs.insert(extension_subs.end(), subs.begin(), subs.end());
        }
      }
    }
  }

  // ROS-related
  using std::placeholders::_1;
  const std::string imu_topic = config_ros.param<std::string>("glim_ros", "imu_topic", "");
  const std::string points_topic = config_ros.param<std::string>("glim_ros", "points_topic", "");
  const std::string image_topic = config_ros.param<std::string>("glim_ros", "image_topic", "");

  // Subscribers
  rclcpp::SensorDataQoS default_imu_qos;
  default_imu_qos.get_rmw_qos_profile().depth = 1000;
  auto qos = get_qos_settings(config_ros, "glim_ros", "imu_qos", default_imu_qos);
  imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, qos, std::bind(&GlimROS::imu_callback, this, _1));

  qos = get_qos_settings(config_ros, "glim_ros", "points_qos");
  points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(points_topic, qos, std::bind(&GlimROS::points_callback, this, _1));
  filtered_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("~/filtered_points", 10);
  dynamic_points_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("~/dynamic_points", 10);
  filtered_pose_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("~/filtered_pose", 10);
#ifdef BUILD_WITH_CV_BRIDGE
  qos = get_qos_settings(config_ros, "glim_ros", "image_qos");
  image_sub = image_transport::create_subscription(this, image_topic, std::bind(&GlimROS::image_callback, this, _1), "raw", qos.get_rmw_qos_profile());
#endif

  for (const auto& sub : this->extension_subscriptions()) {
    spdlog::debug("subscribe to {}", sub->topic);
    sub->create_subscriber(*this);
  }

  // Start timer
  timer = this->create_wall_timer(std::chrono::milliseconds(1), [this]() { timer_callback(); });

  spdlog::debug("initialized");
}

GlimROS::~GlimROS() {
  spdlog::debug("quit");
  extension_modules.clear();

  if (dump_on_unload) {
    std::string dump_path = "/tmp/dump";
    wait(true);
    save(dump_path);
  }
}

const std::vector<std::shared_ptr<GenericTopicSubscription>>& GlimROS::extension_subscriptions() {
  return extension_subs;
}

void GlimROS::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  spdlog::debug("imu callback");
  spdlog::trace("IMU: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "imu_frame_id")) {
    spdlog::debug("auto-detecting IMU frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "imu_frame_id", msg->header.frame_id);
  }

  if (std::abs(acc_scale) < 1e-6) {
    const double norm = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z).norm();
    if (norm > 7.0 && norm < 12.0) {
      acc_scale = 1.0;
      spdlog::debug("assuming [m/s^2] for acceleration unit (acc_scale={}, norm={})", acc_scale, norm);
    } else if (norm > 0.8 && norm < 1.2) {
      acc_scale = 9.80665;
      spdlog::debug("assuming [g] for acceleration unit (acc_scale={}, norm={})", acc_scale, norm);
    } else {
      acc_scale = 1.0;
      spdlog::warn("unexpected acceleration norm {}. assuming [m/s^2] for acceleration unit (acc_scale={})", norm, acc_scale);
    }
  }

  const double imu_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9 + imu_time_offset;
  const Eigen::Vector3d linear_acc = acc_scale * Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d angular_vel(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (!time_keeper->validate_imu_stamp(imu_stamp)) {
    spdlog::warn("skip an invalid IMU data (stamp={})", imu_stamp);
    return;
  }

  odometry_estimation->insert_imu(imu_stamp, linear_acc, angular_vel);
  if (pose_kalman_filter) {
    if (last_imu_stamp_ > 0.0) {
      const double dt = imu_stamp - last_imu_stamp_;
      if (dt > 0.0 && dt < 1.0) {
        std::lock_guard<std::mutex> lock(kf_imu_mutex_);
        kf_imu_queue_.push_back({linear_acc, angular_vel, dt});
      }
    }
    last_imu_stamp_ = imu_stamp;
  }
  if (sub_mapping) {
    sub_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  }
  if (global_mapping) {
    global_mapping->insert_imu(imu_stamp, linear_acc, angular_vel);
  }
}

#ifdef BUILD_WITH_CV_BRIDGE
void GlimROS::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  spdlog::trace("image: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "image_frame")) {
    spdlog::debug("auto-detecting image frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "image_frame", msg->header.frame_id);
  }

  auto cv_image = cv_bridge::toCvCopy(msg, "bgr8");

  const double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;
  odometry_estimation->insert_image(stamp, cv_image->image);
  if (sub_mapping) {
    sub_mapping->insert_image(stamp, cv_image->image);
  }
  if (global_mapping) {
    global_mapping->insert_image(stamp, cv_image->image);
  }
}
#endif

size_t GlimROS::points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  spdlog::debug("points callback");
  spdlog::trace("points: {}.{}", msg->header.stamp.sec, msg->header.stamp.nanosec);
  if (!GlobalConfig::instance()->has_param("meta", "lidar_frame_id")) {
    spdlog::debug("auto-detecting LiDAR frame ID: {}", msg->header.frame_id);
    GlobalConfig::instance()->override_param<std::string>("meta", "lidar_frame_id", msg->header.frame_id);
  }

  auto raw_points = glim::extract_raw_points(*msg, intensity_field, ring_field);
  if (raw_points == nullptr) {
    spdlog::warn("failed to extract points from message");
    return 0;
  }

  raw_points->stamp += points_time_offset;
  if (!time_keeper->process(raw_points)) {
    spdlog::warn("skip an invalid point cloud (stamp={})", raw_points->stamp);
    return 0;
  }
  auto preprocessed = preprocessor->preprocess(raw_points);

  if (keep_raw_points) {
    // note: Raw points are used only in extension modules for visualization purposes.
    //       If you need to reduce the memory footprint, you can safely comment out the following line.
    preprocessed->raw_points = raw_points;
  }

#ifdef GLIM_USE_DYNAMIC_REJECTION
  // Apply dynamic object rejection (async)
  if (dynamic_object_rejection) {
    // Enqueue new frame for background processing
    dynamic_object_rejection->insert_frame(preprocessed);

    // Poll completed results and feed each to odometry (no raw fallback to avoid duplicates)
    auto processed_frames = dynamic_object_rejection->get_results();
    for (const auto& filtered_frame : processed_frames) {
      // Publish filtered point cloud
      if (filtered_points_pub->get_subscription_count() > 0) {
        auto filtered_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        filtered_msg->header = msg->header;
        filtered_msg->header.stamp = rclcpp::Time(static_cast<int64_t>(filtered_frame->stamp * 1e9));
        filtered_msg->height = 1;
        filtered_msg->width = filtered_frame->points.size();
        filtered_msg->fields.resize(3);
        for (int i = 0; i < 3; i++) {
          filtered_msg->fields[i].name = std::vector<std::string>{"x", "y", "z"}[i];
          filtered_msg->fields[i].offset = sizeof(float) * i;
          filtered_msg->fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
          filtered_msg->fields[i].count = 1;
        }
        int point_step = sizeof(float) * 3;
        if (!filtered_frame->intensities.empty()) {
          sensor_msgs::msg::PointField ifield;
          ifield.name = "intensity";
          ifield.offset = point_step;
          ifield.datatype = sensor_msgs::msg::PointField::FLOAT32;
          ifield.count = 1;
          filtered_msg->fields.push_back(ifield);
          point_step += sizeof(float);
        }
        filtered_msg->is_bigendian = false;
        filtered_msg->point_step = point_step;
        filtered_msg->row_step = point_step * filtered_msg->width;
        filtered_msg->data.resize(filtered_msg->row_step);
        filtered_msg->is_dense = true;
        for (size_t i = 0; i < filtered_frame->points.size(); i++) {
          float* ptr = reinterpret_cast<float*>(filtered_msg->data.data() + point_step * i);
          ptr[0] = static_cast<float>(filtered_frame->points[i].x());
          ptr[1] = static_cast<float>(filtered_frame->points[i].y());
          ptr[2] = static_cast<float>(filtered_frame->points[i].z());
          if (!filtered_frame->intensities.empty()) {
            ptr[3] = static_cast<float>(filtered_frame->intensities[i]);
          }
        }
        filtered_points_pub->publish(std::move(filtered_msg));
      }
      odometry_estimation->insert_frame(filtered_frame);
    }

    // Publish dynamic-only points
    if (dynamic_points_pub->get_subscription_count() > 0) {
      auto dyn_frames = dynamic_object_rejection->get_dynamic_results();
      for (const auto& dyn_frame : dyn_frames) {
        if (!dyn_frame || dyn_frame->points.empty()) continue;
        auto dyn_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        dyn_msg->header = msg->header;
        dyn_msg->header.stamp = rclcpp::Time(static_cast<int64_t>(dyn_frame->stamp * 1e9));
        dyn_msg->height = 1;
        dyn_msg->width = dyn_frame->points.size();
        dyn_msg->fields.resize(3);
        for (int i = 0; i < 3; i++) {
          dyn_msg->fields[i].name = std::vector<std::string>{"x", "y", "z"}[i];
          dyn_msg->fields[i].offset = sizeof(float) * i;
          dyn_msg->fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
          dyn_msg->fields[i].count = 1;
        }
        int dyn_step = sizeof(float) * 3;
        if (!dyn_frame->intensities.empty()) {
          sensor_msgs::msg::PointField ifield;
          ifield.name = "intensity";
          ifield.offset = dyn_step;
          ifield.datatype = sensor_msgs::msg::PointField::FLOAT32;
          ifield.count = 1;
          dyn_msg->fields.push_back(ifield);
          dyn_step += sizeof(float);
        }
        dyn_msg->is_bigendian = false;
        dyn_msg->point_step = dyn_step;
        dyn_msg->row_step = dyn_step * dyn_msg->width;
        dyn_msg->data.resize(dyn_msg->row_step);
        dyn_msg->is_dense = true;
        for (size_t i = 0; i < dyn_frame->points.size(); i++) {
          float* ptr = reinterpret_cast<float*>(dyn_msg->data.data() + dyn_step * i);
          ptr[0] = static_cast<float>(dyn_frame->points[i].x());
          ptr[1] = static_cast<float>(dyn_frame->points[i].y());
          ptr[2] = static_cast<float>(dyn_frame->points[i].z());
          if (!dyn_frame->intensities.empty()) {
            ptr[3] = static_cast<float>(dyn_frame->intensities[i]);
          }
        }
        dynamic_points_pub->publish(std::move(dyn_msg));
      }
    }
  } else {
    odometry_estimation->insert_frame(preprocessed);
  }
#else
  odometry_estimation->insert_frame(preprocessed);
#endif

  const size_t workload = odometry_estimation->workload();
  spdlog::debug("workload={}", workload);

  return workload;
}

bool GlimROS::needs_wait() {
  for (const auto& ext_module : extension_modules) {
    if (ext_module->needs_wait()) {
      return true;
    }
  }

  return false;
}

void GlimROS::timer_callback() {
  for (const auto& ext_module : extension_modules) {
    if (!ext_module->ok()) {
      rclcpp::shutdown();
    }
  }

  std::vector<glim::EstimationFrame::ConstPtr> estimation_frames;
  std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
  odometry_estimation->get_results(estimation_frames, marginalized_frames);

  // Save the last estimation frame
  if (!estimation_frames.empty()) {
    if (pose_kalman_filter) {
      // Drain IMU queue and run all pending predictions on this thread
      size_t n_imu = 0;
      {
        std::lock_guard<std::mutex> lock(kf_imu_mutex_);
        n_imu = kf_imu_queue_.size();
        for (const auto& imu : kf_imu_queue_) {
          glim::ImuMeasurement m;
          m.acc = imu.acc;
          m.gyro = imu.gyro;
          m.dt = imu.dt;
          pose_kalman_filter->predict(m);
        }
        kf_imu_queue_.clear();
      }

      spdlog::debug("[KF] predicted {} IMU samples, calling update", n_imu);
      const auto& latest = estimation_frames.back();
      if (!latest) {
        spdlog::warn("[KF] latest estimation frame is null, skipping update");
      } else {
        spdlog::debug("[KF] latest frame stamp={:.6f}", latest->stamp);
        const Eigen::Isometry3d T_world_imu_filtered = pose_kalman_filter->update(latest->T_world_imu);
        spdlog::debug("[KF] update done, publishing");

        auto pose_msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
        pose_msg->header.stamp = rclcpp::Time(static_cast<int64_t>(latest->stamp * 1e9));
        pose_msg->header.frame_id = "map";

        const Eigen::Vector3d p = T_world_imu_filtered.translation();
        const Eigen::Quaterniond q(T_world_imu_filtered.rotation());
        pose_msg->pose.position.x = p.x();
        pose_msg->pose.position.y = p.y();
        pose_msg->pose.position.z = p.z();
        pose_msg->pose.orientation.w = q.w();
        pose_msg->pose.orientation.x = q.x();
        pose_msg->pose.orientation.y = q.y();
        pose_msg->pose.orientation.z = q.z();

        filtered_pose_pub->publish(std::move(pose_msg));
        spdlog::info("[KF] published filtered pose");
      }
    }
  }

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

void GlimROS::wait(bool auto_quit) {
  spdlog::info("waiting for odometry estimation");
  odometry_estimation->join();

  if (sub_mapping) {
    std::vector<glim::EstimationFrame::ConstPtr> estimation_results;
    std::vector<glim::EstimationFrame::ConstPtr> marginalized_frames;
    odometry_estimation->get_results(estimation_results, marginalized_frames);
    
    // Save the last estimation frame
    if (!estimation_results.empty()) {
      // estimation_results available for other uses
    }
    
    for (const auto& marginalized_frame : marginalized_frames) {
      sub_mapping->insert_frame(marginalized_frame);
    }

    spdlog::info("waiting for local mapping");
    sub_mapping->join();

    const auto submaps = sub_mapping->get_results();
    if (global_mapping) {
      for (const auto& submap : submaps) {
        global_mapping->insert_submap(submap);
      }
      spdlog::info("waiting for global mapping");
      global_mapping->join();
    }
  }

  if (!auto_quit) {
    bool terminate = false;
    while (!terminate && rclcpp::ok()) {
      for (const auto& ext_module : extension_modules) {
        terminate |= (!ext_module->ok());
      }
    }
  }
}

void GlimROS::save(const std::string& path) {
  if (global_mapping) global_mapping->save(path);
  for (auto& module : extension_modules) {
    module->at_exit(path);
  }
}

}  // namespace glim

RCLCPP_COMPONENTS_REGISTER_NODE(glim::GlimROS);