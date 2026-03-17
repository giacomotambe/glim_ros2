#include <chrono>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

using namespace std::chrono_literals;

struct BBox
{
    float x, y, z;
    float dx, dy, dz;
};

class BBoxPublisher : public rclcpp::Node
{
public:
    BBoxPublisher()
    : Node("bbox_publisher")
    {
        publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/bounding_boxes", 10);

        timer_ = this->create_wall_timer(
            100ms, std::bind(&BBoxPublisher::publish_bboxes, this));

        // Esempio: lista di bounding box
        bboxes_ = {
            {1.0, 2.0, 0.5, 1.0, 0.5, 1.0},
            {2.0, 1.0, 0.5, 0.8, 0.8, 0.8},
            {0.0, 0.0, 0.5, 1.5, 0.3, 1.2}
        };
    }

private:
    void publish_bboxes()
    {
        visualization_msgs::msg::MarkerArray marker_array;

        int id = 0;
        for (const auto & box : bboxes_)
        {
            visualization_msgs::msg::Marker marker;

            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();

            marker.ns = "bounding_boxes";
            marker.id = id++;  // IMPORTANTE: ID univoco

            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // Posizione
            marker.pose.position.x = box.x;
            marker.pose.position.y = box.y;
            marker.pose.position.z = box.z;

            // Orientamento (identity)
            marker.pose.orientation.w = 1.0;

            // Dimensioni
            marker.scale.x = box.dx;
            marker.scale.y = box.dy;
            marker.scale.z = box.dz;

            // Colore (rosso variabile)
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.8;

            marker.lifetime = rclcpp::Duration(0, 0);

            marker_array.markers.push_back(marker);
        }

        publisher_->publish(marker_array);
    }

    std::vector<BBox> bboxes_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BBoxPublisher>());
    rclcpp::shutdown();
    return 0;
}