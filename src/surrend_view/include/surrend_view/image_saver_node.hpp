#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>

namespace image_saver {

class ImageSaverNode : public rclcpp::Node {
public:
    ImageSaverNode();
    ~ImageSaverNode() = default;

private:
    // 订阅回调函数（前视摄像头）
    void front_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    // 订阅回调函数（后视摄像头）
    void back_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    // 订阅者
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr front_img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr back_img_sub_;

    // 保存标志位（确保各保存1张）
    std::atomic<bool> save_front_flag_;
    std::atomic<bool> save_back_flag_;

    // 保存路径（通过参数获取）
    std::string front_save_path_;
    std::string back_save_path_;
};

} // namespace image_saver