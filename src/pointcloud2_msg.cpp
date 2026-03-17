#include "pointcloud_utils.hpp"

std::unique_ptr<sensor_msgs::msg::PointCloud2> create_pointcloud2_msg(
    const std_msgs::msg::Header& header,
    const glim::PreprocessedFrame::ConstPtr& frame)
{
  auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();

  msg->header = header;
  msg->header.stamp = rclcpp::Time(static_cast<int64_t>(frame->stamp * 1e9));

  msg->height = 1;
  msg->width = frame->points.size();

  msg->fields.resize(3);
  const std::array<std::string,3> names = {"x","y","z"};

  for (int i = 0; i < 3; i++) {
    msg->fields[i].name = names[i];
    msg->fields[i].offset = sizeof(float) * i;
    msg->fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg->fields[i].count = 1;
  }

  int point_step = sizeof(float) * 3;

  if (!frame->intensities.empty()) {
    sensor_msgs::msg::PointField ifield;
    ifield.name = "intensity";
    ifield.offset = point_step;
    ifield.datatype = sensor_msgs::msg::PointField::FLOAT32;
    ifield.count = 1;

    msg->fields.push_back(ifield);
    point_step += sizeof(float);
  }

  msg->is_bigendian = false;
  msg->point_step = point_step;
  msg->row_step = point_step * msg->width;
  msg->data.resize(msg->row_step);
  msg->is_dense = true;

  for (size_t i = 0; i < frame->points.size(); i++) {
    float* ptr = reinterpret_cast<float*>(msg->data.data() + point_step * i);

    ptr[0] = static_cast<float>(frame->points[i].x());
    ptr[1] = static_cast<float>(frame->points[i].y());
    ptr[2] = static_cast<float>(frame->points[i].z());

    if (!frame->intensities.empty()) {
      ptr[3] = static_cast<float>(frame->intensities[i]);
    }
  }

  return msg;
}