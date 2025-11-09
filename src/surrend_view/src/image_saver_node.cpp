#include "surrend_view/image_saver_node.hpp"
#include <rclcpp/logging.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace image_saver {

ImageSaverNode::ImageSaverNode() 
    : Node("image_saver_node")
    , save_front_flag_(true)  // 初始允许保存前视图片
    , save_back_flag_(true)   // 初始允许保存后视图片
{
    // 1. 声明并获取自定义参数（保存路径，支持启动时指定）
    this->declare_parameter<std::string>("front_save_path", "/home/tl/front_camera.jpg");
    this->declare_parameter<std::string>("back_save_path", "/home/tl/back_camera.jpg");
    this->get_parameter("front_save_path", front_save_path_);
    this->get_parameter("back_save_path", back_save_path_);

    // 2. 初始化订阅者（订阅两个摄像头的图像话题）
    front_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera0/image_raw",  // 前视摄像头话题（对应之前的video0）
        10,
        std::bind(&ImageSaverNode::front_image_callback, this, std::placeholders::_1)
    );

    back_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera2/image_raw",  // 后视摄像头话题（对应之前的video2）
        10,
        std::bind(&ImageSaverNode::back_image_callback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Image Saver Node Started!");
    RCLCPP_INFO(this->get_logger(), "Front image will be saved to: %s", front_save_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Back image will be saved to: %s", back_save_path_.c_str());
}

// 前视摄像头图像回调（保存1张后停止）
void ImageSaverNode::front_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (save_front_flag_.load()) {
        try {
            // ROS图像消息 → OpenCV Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image = cv_ptr->image;

            if (image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Front camera image is empty, skip saving!");
                return;
            }

            // 保存图片到本地
            bool save_success = cv::imwrite(front_save_path_, image);
            if (save_success) {
                RCLCPP_INFO(this->get_logger(), "Front image saved successfully! Path: %s", front_save_path_.c_str());
                save_front_flag_.store(false);  // 置为false，不再保存
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save front image! Check path or permissions.");
            }
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed for front image: %s", e.what());
        }
    }
}

// 后视摄像头图像回调（保存1张后停止）
void ImageSaverNode::back_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (save_back_flag_.load()) {
        try {
            // ROS图像消息 → OpenCV Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image = cv_ptr->image;

            if (image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Back camera image is empty, skip saving!");
                return;
            }

            // 保存图片到本地
            bool save_success = cv::imwrite(back_save_path_, image);
            if (save_success) {
                RCLCPP_INFO(this->get_logger(), "Back image saved successfully! Path: %s", back_save_path_.c_str());
                save_back_flag_.store(false);  // 置为false，不再保存

                // 两个图片都保存完成后，自动关闭节点（可选）
                if (!save_front_flag_.load() && !save_back_flag_.load()) {
                    RCLCPP_INFO(this->get_logger(), "All images saved, exiting node...");
                    rclcpp::shutdown();
                }
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save back image! Check path or permissions.");
            }
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed for back image: %s", e.what());
        }
    }
}

} // namespace image_saver

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<image_saver::ImageSaverNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}