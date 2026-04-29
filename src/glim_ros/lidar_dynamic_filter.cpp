#include <cmath>
#include <cstring>

#include <glim_ros/lidar_dynamic_filter.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace glim {

LidarDynamicFilter::LidarDynamicFilter()
: Node("lidar_dynamic_filter"),
  tf_buffer_(get_clock()),
  tf_listener_(tf_buffer_)
{
  declare_parameter("tracking_timeout", 0.5);
  declare_parameter("obstacle_padding", 0.10);

  pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/velodyne_points_filtered", 10);

  obs_sub_ = create_subscription<jo_msgs::msg::ObstacleArray>(
    "/onboard_detector/tracked_dynamic_obstacles", 10,
    std::bind(&LidarDynamicFilter::obstaclesCallback, this, std::placeholders::_1));

  lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/velodyne_points", 10,
    std::bind(&LidarDynamicFilter::lidarCallback, this, std::placeholders::_1));
}

void LidarDynamicFilter::obstaclesCallback(const jo_msgs::msg::ObstacleArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  latest_obstacles_.clear();
  latest_frame_ = msg->header.frame_id;
  last_obstacles_time_ = now();

  const double padding = get_parameter("obstacle_padding").as_double();

  for (const auto& obs : msg->obstacles) {
    ObsBox box;
    box.x  = obs.pose.position.x;
    box.y  = obs.pose.position.y;
    box.z  = obs.pose.position.z;
    box.hx = obs.size.x * 0.5 + padding;
    box.hy = obs.size.y * 0.5 + padding;
    box.hz = obs.size.z * 0.5 + padding;
    latest_obstacles_.push_back(box);
  }
}

void LidarDynamicFilter::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  std::vector<ObsBox> obstacles;
  std::string obstacle_frame;
  bool active = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    const double timeout = get_parameter("tracking_timeout").as_double();
    active =
      last_obstacles_time_.has_value() &&
      (now() - *last_obstacles_time_).seconds() < timeout &&
      !latest_obstacles_.empty();

    if (active) {
      obstacles      = latest_obstacles_;
      obstacle_frame = latest_frame_;
    }
  }

  if (!active || obstacle_frame.empty()) {
    pub_->publish(*msg);
    return;
  }

  int x_offset = -1, y_offset = -1, z_offset = -1;
  for (const auto& field : msg->fields) {
    if (field.name == "x")      x_offset = static_cast<int>(field.offset);
    else if (field.name == "y") y_offset = static_cast<int>(field.offset);
    else if (field.name == "z") z_offset = static_cast<int>(field.offset);
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    pub_->publish(*msg);
    return;
  }

  // When glim odometry is not yet available the odom frame does not exist in
  // the TF tree. In that case pass through the raw cloud so that glim can
  // bootstrap its odometry without being starved of points.
  if (!tf_buffer_.canTransform(obstacle_frame, msg->header.frame_id, rclcpp::Time(0))) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
      "Odom TF not yet available (%s→%s), passing raw cloud to /velodyne_points_filtered",
      obstacle_frame.c_str(), msg->header.frame_id.c_str());
    pub_->publish(*msg);
    return;
  }

  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    tf_msg = tf_buffer_.lookupTransform(
      obstacle_frame, msg->header.frame_id, rclcpp::Time(0));
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "TF lookup failed, passing raw cloud: %s", ex.what());
    pub_->publish(*msg);
    return;
  }

  tf2::Transform tf;
  tf2::fromMsg(tf_msg.transform, tf);

  const uint32_t point_step = msg->point_step;
  const uint32_t n_points   = msg->width * msg->height;
  const uint8_t* raw        = msg->data.data();

  std::vector<uint8_t> kept;
  kept.reserve(n_points * point_step);

  for (uint32_t i = 0; i < n_points; ++i) {
    const uint8_t* base = raw + i * point_step;

    float px = 0.0f, py = 0.0f, pz = 0.0f;
    std::memcpy(&px, base + x_offset, sizeof(float));
    std::memcpy(&py, base + y_offset, sizeof(float));
    std::memcpy(&pz, base + z_offset, sizeof(float));

    const tf2::Vector3 p_obs = tf * tf2::Vector3(px, py, pz);

    bool inside = false;
    for (const auto& obs : obstacles) {
      if (std::abs(p_obs.x() - obs.x) <= obs.hx &&
          std::abs(p_obs.y() - obs.y) <= obs.hy &&
          std::abs(p_obs.z() - obs.z) <= obs.hz) {
        inside = true;
        break;
      }
    }

    if (!inside) {
      kept.insert(kept.end(), base, base + point_step);
    }
  }

  sensor_msgs::msg::PointCloud2 out;
  out.header       = msg->header;
  out.height       = 1;
  out.width        = static_cast<uint32_t>(kept.size() / point_step);
  out.fields       = msg->fields;
  out.is_bigendian = msg->is_bigendian;
  out.point_step   = point_step;
  out.row_step     = static_cast<uint32_t>(kept.size());
  out.data         = std::move(kept);
  out.is_dense     = false;
  pub_->publish(out);
}

}  // namespace glim
