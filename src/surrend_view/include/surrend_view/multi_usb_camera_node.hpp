#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>

namespace surrend_view {

class MultiUSBCameraNode : public rclcpp::Node {
public:
    MultiUSBCameraNode();
    ~MultiUSBCameraNode();

private:
    struct CameraHandler {
        std::shared_ptr<cv::VideoCapture> cap;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub;
        std::string topic_name;
        std::string device_path;
        std::thread worker;
        bool running = false;

        void run(double fps, rclcpp::Node* parent_node);
        void stop();
    };

    std::vector<std::unique_ptr<CameraHandler>> cameras_;
    double publish_fps_ = 30.0;
    std::vector<std::string> camera_devices_;

    void declare_and_get_params();
    void start_cameras();
    void stop_cameras();
};

} // namespace surrend_view
