#include "surrend_view/path_manager.hpp"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace fs = std::filesystem;

namespace surrend_view {

std::string getWorkspaceRoot() {
    // 获取可执行文件路径
    char exe_path[1024] = {0};
    
#ifdef _WIN32
    GetModuleFileNameA(NULL, exe_path, sizeof(exe_path));
#else
    ssize_t count = readlink("/proc/self/exe", exe_path, sizeof(exe_path));
    if (count == -1) {
        // 如果无法获取，尝试使用当前工作目录
        char* cwd = getcwd(nullptr, 0);
        if (cwd) {
            std::string cwd_str(cwd);
            free(cwd);
            return cwd_str;
        }
        return ".";
    }
#endif
    
    fs::path exe_file(exe_path);
    fs::path current_path = exe_file.parent_path();
    
    // 向上查找，直到找到包含 src/ 和 install/ 的目录
    for (int i = 0; i < 10; ++i) {
        if (fs::exists(current_path / "src") && 
            fs::exists(current_path / "install")) {
            return current_path.string();
        }
        if (current_path.has_parent_path() && current_path != current_path.parent_path()) {
            current_path = current_path.parent_path();
        } else {
            break;
        }
    }
    
    // 如果找不到，尝试从 install 目录向上查找
    if (fs::exists(exe_file.parent_path() / ".." / ".." / "src")) {
        return (exe_file.parent_path() / ".." / "..").string();
    }
    
    // 最后尝试使用当前工作目录
    char* cwd = getcwd(nullptr, 0);
    if (cwd) {
        std::string cwd_str(cwd);
        free(cwd);
        return cwd_str;
    }
    
    return ".";
}

std::string getConfigBaseDir() {
    // 检查环境变量
    const char* env_config_dir = std::getenv("AVM_CONFIG_DIR");
    if (env_config_dir && fs::exists(env_config_dir)) {
        return std::string(env_config_dir);
    }
    
    // 使用工作空间根目录下的 config 目录
    std::string workspace_root = getWorkspaceRoot();
    fs::path config_dir = fs::path(workspace_root) / "config";
    
    // 如果目录不存在，创建它
    if (!fs::exists(config_dir)) {
        fs::create_directories(config_dir);
    }
    
    return config_dir.string();
}

std::string getConfigPath(const std::string& subdir, bool create_if_not_exists) {
    std::string base_dir = getConfigBaseDir();
    fs::path config_path = fs::path(base_dir);
    
    if (!subdir.empty()) {
        config_path = config_path / subdir;
    }
    
    if (create_if_not_exists && !fs::exists(config_path)) {
        fs::create_directories(config_path);
    }
    
    return config_path.string();
}

std::string getCalibrationDir(bool create_if_not_exists) {
    return getConfigPath("calibration_results", create_if_not_exists);
}

std::string getCalibrationFile(const std::string& camera_name, bool create_dir) {
    std::string calib_dir = getCalibrationDir(create_dir);
    fs::path calib_file = fs::path(calib_dir) / (camera_name + "_calibration.yaml");
    return calib_file.string();
}

void setConfigDir(const std::string& config_dir) {
#ifdef _WIN32
    _putenv_s("AVM_CONFIG_DIR", config_dir.c_str());
#else
    setenv("AVM_CONFIG_DIR", config_dir.c_str(), 1);
#endif
}

} // namespace surrend_view

