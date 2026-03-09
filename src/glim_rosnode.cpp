#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <spdlog/spdlog.h>
#include <rclcpp/rclcpp.hpp>

#include <glim_ros/glim_ros.hpp>
#include <glim/util/config.hpp>
#include <glim/util/extension_module_ros2.hpp>

int main(int argc, char** argv) {
  printf("GlimROS node created\n");
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::NodeOptions options;

  auto glim = std::make_shared<glim::GlimROS>(options);
  
  rclcpp::spin(glim);
  rclcpp::shutdown();

  std::string dump_path = "/tmp/dump/";
  // Add timestamp to dump_path
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t_now), "%Y_%m_%d_%H_%M_%S");
  
  // Ensure path ends with /
  if (dump_path.back() != '/') {
    dump_path += '/';
  }
  dump_path += ss.str() + "/";
  glim->declare_parameter<std::string>("dump_path", dump_path);
  glim->get_parameter<std::string>("dump_path", dump_path);

  

  glim->wait();
  glim->save(dump_path);

  return 0;
}