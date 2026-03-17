#pragma once

#include <memory>
#include <array>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <rclcpp/rclcpp.hpp>

#include <glim/preprocess/preprocessed_frame.hpp>

namespace glim_ros_utils {

/**
 * @brief Crea un messaggio ROS2 PointCloud2 da un PreprocessedFrame
 * 
 * @param header header del messaggio (timestamp, frame_id)
 * @param frame puntatore al frame contenente punti e intensità
 * @return std::unique_ptr<sensor_msgs::msg::PointCloud2> messaggio pronto per il publish
 */
std::unique_ptr<sensor_msgs::msg::PointCloud2> create_pointcloud2_msg(
    const std_msgs::msg::Header& header,
    const glim::PreprocessedFrame::ConstPtr& frame);

} // namespace glim_ros_utils