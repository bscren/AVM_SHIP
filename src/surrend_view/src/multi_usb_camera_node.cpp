#include "surrend_view/multi_usb_camera_node.hpp"
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>


using std::placeholders::_1;

namespace surrend_view {

MultiUSBCameraNode::MultiUSBCameraNode() : Node("multi_usb_camera_node") {
    declare_and_get_params();
    start_cameras();
}

MultiUSBCameraNode::~MultiUSBCameraNode() {
    stop_cameras();
}

void MultiUSBCameraNode::declare_and_get_params() {
    this->declare_parameter("camera_devices", std::vector<std::string>{"/dev/video0","/dev/video1","/dev/video2","/dev/video3"});
    this->declare_parameter("publish_fps", 30.0);
    this->declare_parameter("frame_width", 640);
    this->declare_parameter("frame_height", 480);
    this->declare_parameter("frame_rate", 30);
    this->get_parameter("camera_devices", camera_devices_);
    this->get_parameter("publish_fps", publish_fps_);
    this->get_parameter("frame_width", frame_width_);
    this->get_parameter("frame_height", frame_height_);
    this->get_parameter("frame_rate", frame_rate_);
}

/**
 * @brief 启动摄像头
 */
void MultiUSBCameraNode::start_cameras() {
    stop_cameras();
    for (size_t i = 0; i < camera_devices_.size(); ++i) {
        auto cam = std::make_unique<CameraHandler>();
        cam->device_path = camera_devices_[i];
        cam->topic_name = "/camera" + std::to_string(i) + "/image_raw";
        
        // 尝试打开摄像头设备
        // 支持两种方式：设备路径（如/dev/video0）或索引（如0）
        try {
            int camera_index = -1;
            
            // 尝试从路径中提取索引（如/dev/videoX -> X，X为数字）
            if (cam->device_path.find("/dev/video") == 0) {
                std::string index_str = cam->device_path.substr(10); // substr(10)意为从第10个字符开始截取
                try {
                    camera_index = std::stoi(index_str); // 将字符串转换为整数
                } catch (...) {
                    // 如果无法解析，继续使用路径方式
                }
            }
            
            // 优先使用索引方式打开（更稳定，与Python脚本一致）
            if (camera_index >= 0) {
                RCLCPP_INFO(this->get_logger(), 
                    "Opening camera by index: %d (device: %s)", 
                    camera_index, cam->device_path.c_str());
                cam->cap = std::make_shared<cv::VideoCapture>(camera_index, cv::CAP_V4L2);
            } else {
                // 如果无法提取索引，使用路径方式
                RCLCPP_INFO(this->get_logger(), 
                    "Opening camera by path: %s", cam->device_path.c_str());
                cam->cap = std::make_shared<cv::VideoCapture>(cam->device_path, cv::CAP_V4L2);
            }
            
            if (!cam->cap->isOpened()) {
                RCLCPP_WARN(this->get_logger(), 
                    "Failed to open camera device: %s, skipping...", 
                    cam->device_path.c_str());
                continue;  // 跳过无法打开的摄像头
            }
            
            // 在读取帧之前先设置格式和分辨率（防止读取到错误的默认格式）
            cam->cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); // fourcc('M', 'J', 'P', 'G')意为MJPG格式
            cam->cap->set(cv::CAP_PROP_FRAME_WIDTH, frame_width_); // 
            cam->cap->set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_); // 
            cam->cap->set(cv::CAP_PROP_FPS, frame_rate_); // 
            
            // 设置缓冲区大小（减少延迟）
            cam->cap->set(cv::CAP_PROP_BUFFERSIZE, 1);
            
            // 读取一帧来验证摄像头是否正常工作
            cv::Mat test_frame;
            if (!cam->cap->read(test_frame) || test_frame.empty()) {
                RCLCPP_WARN(this->get_logger(), 
                    "Camera %s opened but failed to read test frame, skipping...", 
                    cam->device_path.c_str());
                cam->cap->release();
                continue;
            }
            
            // 验证帧尺寸是否合理
            if (test_frame.cols <= 0 || test_frame.rows <= 0 || 
                test_frame.cols > 10000 || test_frame.rows > 10000) {
                RCLCPP_WARN(this->get_logger(), 
                    "Camera %s returned invalid frame size (%dx%d), skipping...", 
                    cam->device_path.c_str(), test_frame.cols, test_frame.rows);
                cam->cap->release();
                continue;
            }
            
            RCLCPP_INFO(this->get_logger(), 
                "Successfully opened camera: %s (resolution: %dx%d), publishing to: %s", 
                cam->device_path.c_str(), test_frame.cols, test_frame.rows, cam->topic_name.c_str());
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), 
                "OpenCV exception while opening camera %s: %s", 
                cam->device_path.c_str(), e.what());
            continue;  // 跳过异常摄像头
        }
        
        cam->pub = this->create_publisher<sensor_msgs::msg::Image>(cam->topic_name, 2);
        cam->running = true;
        cam->worker = std::thread([this, cam_ptr=cam.get()]() {
            cam_ptr->run(this->publish_fps_, this);
        });
        cameras_.push_back(std::move(cam));
    }
    
    if (cameras_.empty()) {
        RCLCPP_ERROR(this->get_logger(), 
            "No cameras were successfully opened! Please check your camera devices.");
    } else {
        RCLCPP_INFO(this->get_logger(), 
            "Successfully initialized %zu camera(s)", cameras_.size());
    }
}

/**
 * @brief 停止摄像头
 */
void MultiUSBCameraNode::stop_cameras() {
    for (auto &cam : cameras_) {
        cam->stop();
    }
    cameras_.clear();
}

/**
 * @brief 摄像头处理线程
 */
void MultiUSBCameraNode::CameraHandler::run(double fps, rclcpp::Node* parent_node) {
    rclcpp::WallRate loop_rate(fps);
    while (running && rclcpp::ok()) {
        if (!cap || !cap->isOpened()) {
            RCLCPP_WARN(parent_node->get_logger(), 
                "Camera %s is not opened, stopping worker thread", device_path.c_str());
            break;
        }
        
        cv::Mat frame;
        if (!cap->read(frame) || frame.empty()) {
            RCLCPP_WARN_THROTTLE(parent_node->get_logger(), *parent_node->get_clock(), 
                5000, "Camera %s: Failed to capture frame", device_path.c_str());
            loop_rate.sleep();
            continue;
        }
        
        // 验证帧尺寸是否合理（防止内存分配错误）
        if (frame.cols <= 0 || frame.rows <= 0 || 
            frame.cols > 10000 || frame.rows > 10000) {
            RCLCPP_ERROR_THROTTLE(parent_node->get_logger(), *parent_node->get_clock(), 
                5000, "Camera %s: Invalid frame size (%dx%d), skipping frame", 
                device_path.c_str(), frame.cols, frame.rows);
            loop_rate.sleep();
            continue;
        }
        
        try {
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            msg->header.stamp = parent_node->now();
            msg->header.frame_id = "camera_" + device_path;
            pub->publish(*msg);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR_THROTTLE(parent_node->get_logger(), *parent_node->get_clock(), 
                5000, "cv_bridge exception for camera %s: %s", device_path.c_str(), e.what());
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR_THROTTLE(parent_node->get_logger(), *parent_node->get_clock(), 
                5000, "OpenCV exception for camera %s: %s", device_path.c_str(), e.what());
        }
        
        loop_rate.sleep();
    }
}

/**
 * @brief 停止摄像头处理线程
 */
void MultiUSBCameraNode::CameraHandler::stop() {
    running = false;
    if (worker.joinable()) worker.join();
    if (cap && cap->isOpened()) cap->release();
}

} // namespace surrend_view
