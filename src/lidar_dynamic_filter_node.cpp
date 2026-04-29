#include <rclcpp/rclcpp.hpp>
#include <glim_ros/lidar_dynamic_filter.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<glim::LidarDynamicFilter>());
  rclcpp::shutdown();
  return 0;
}
