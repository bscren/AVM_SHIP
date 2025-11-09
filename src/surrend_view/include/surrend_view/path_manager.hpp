#pragma once

#include <string>
#include <filesystem>

namespace surrend_view {

/**
 * @brief 获取工作空间根目录
 * @return 工作空间根目录路径
 */
std::string getWorkspaceRoot();

/**
 * @brief 获取配置基础目录
 * 优先级：环境变量 AVM_CONFIG_DIR > 工作空间根目录/config
 * @return 配置基础目录路径
 */
std::string getConfigBaseDir();

/**
 * @brief 获取配置子目录
 * @param subdir 子目录名称（如 "calibration_results"）
 * @param create_if_not_exists 如果目录不存在是否创建
 * @return 配置目录路径
 */
std::string getConfigPath(const std::string& subdir = "", bool create_if_not_exists = true);

/**
 * @brief 获取标定文件目录
 * @param create_if_not_exists 如果目录不存在是否创建
 * @return 标定文件目录路径
 */
std::string getCalibrationDir(bool create_if_not_exists = true);

/**
 * @brief 获取标定文件路径
 * @param camera_name 相机名称（front, back, left, right）
 * @param create_dir 如果目录不存在是否创建
 * @return 标定文件路径
 */
std::string getCalibrationFile(const std::string& camera_name, bool create_dir = true);

/**
 * @brief 设置配置目录（通过环境变量）
 * @param config_dir 配置目录路径
 */
void setConfigDir(const std::string& config_dir);

} // namespace surrend_view

