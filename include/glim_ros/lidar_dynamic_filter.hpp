#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <jo_msgs/msg/obstacle_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace glim {

class LidarDynamicFilter : public rclcpp::Node {
public:
  LidarDynamicFilter();

private:
  struct ObsBox {
    double x, y, z;
    double hx, hy, hz;
  };

  void obstaclesCallback(const jo_msgs::msg::ObstacleArray::SharedPtr msg);
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<jo_msgs::msg::ObstacleArray>::SharedPtr obs_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex mutex_;
  std::vector<ObsBox> latest_obstacles_;
  std::string latest_frame_;
  std::optional<rclcpp::Time> last_obstacles_time_;
};

}  // namespace glim
