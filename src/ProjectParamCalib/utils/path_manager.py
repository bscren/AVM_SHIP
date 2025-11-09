#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径管理工具
功能：统一管理共享配置文件的路径，支持环境变量和参数覆盖
"""

import os
from pathlib import Path


def get_workspace_root():
    """
    获取工作空间根目录
    
    通过查找包含 src/ 目录的父目录来确定工作空间根目录
    
    Returns:
        Path: 工作空间根目录路径
    """
    # 获取当前文件的目录
    current_file = Path(__file__).resolve()
    
    # 向上查找，直到找到包含 src/ 目录的目录
    for parent in current_file.parents:
        if (parent / "src").exists() and (parent / "install").exists():
            return parent
    
    # 如果找不到，使用当前目录的父目录（假设在 src/ProjectParamCalib 下）
    return current_file.parent.parent.parent


def get_config_base_dir():
    """
    获取配置文件基础目录
    
    优先级：
    1. 环境变量 AVM_CONFIG_DIR
    2. 工作空间根目录下的 config 目录
    
    Returns:
        Path: 配置文件基础目录
    """
    # 检查环境变量
    env_config_dir = os.getenv("AVM_CONFIG_DIR")
    if env_config_dir and os.path.exists(env_config_dir):
        return Path(env_config_dir)
    
    # 使用工作空间根目录下的 config 目录
    workspace_root = get_workspace_root()
    config_dir = workspace_root / "config"
    
    # 如果目录不存在，创建它
    config_dir.mkdir(exist_ok=True)
    
    return config_dir


def get_config_path(subdir=None, create_if_not_exists=True):
    """
    获取配置子目录路径
    
    Args:
        subdir: 子目录名称（如 "calibration_results"）
        create_if_not_exists: 如果目录不存在是否创建
        
    Returns:
        Path: 配置目录路径
    """
    base_dir = get_config_base_dir()
    
    if subdir:
        config_path = base_dir / subdir
    else:
        config_path = base_dir
    
    if create_if_not_exists:
        config_path.mkdir(parents=True, exist_ok=True)
    
    return config_path


def get_calibration_dir(create_if_not_exists=True):
    """
    获取标定文件目录
    
    Args:
        create_if_not_exists: 如果目录不存在是否创建
        
    Returns:
        Path: 标定文件目录路径
    """
    return get_config_path("calibration_results", create_if_not_exists)


def get_projection_dir(create_if_not_exists=True):
    """
    获取投影映射文件目录
    
    Args:
        create_if_not_exists: 如果目录不存在是否创建
        
    Returns:
        Path: 投影映射文件目录路径
    """
    return get_config_path("projection_maps", create_if_not_exists)


def get_calibration_file(camera_name, create_dir=True):
    """
    获取标定文件路径
    
    Args:
        camera_name: 相机名称（front, back, left, right）
        create_dir: 如果目录不存在是否创建
        
    Returns:
        Path: 标定文件路径
    """
    calib_dir = get_calibration_dir(create_dir)
    return calib_dir / f"{camera_name}_calibration.yaml"


def get_projection_file(camera_name, file_type="maps", create_dir=True):
    """
    获取投影映射文件路径
    
    Args:
        camera_name: 相机名称
        file_type: 文件类型（"maps" 或 "points"）
        create_dir: 如果目录不存在是否创建
        
    Returns:
        Path: 投影映射文件路径
    """
    proj_dir = get_projection_dir(create_dir)
    
    if file_type == "maps":
        return proj_dir / f"projection_maps_{camera_name}.yaml"
    elif file_type == "points":
        return proj_dir / f"birdview_points_{camera_name}.yaml"
    else:
        raise ValueError(f"未知的文件类型: {file_type}")


def set_config_dir(config_dir):
    """
    设置配置目录（通过环境变量）
    
    Args:
        config_dir: 配置目录路径
    """
    os.environ["AVM_CONFIG_DIR"] = str(config_dir)


# 测试函数
if __name__ == "__main__":
    print("工作空间根目录:", get_workspace_root())
    print("配置基础目录:", get_config_base_dir())
    print("标定文件目录:", get_calibration_dir())
    print("投影映射目录:", get_projection_dir())
    print("前视标定文件:", get_calibration_file("front"))
    print("前视投影映射:", get_projection_file("front", "maps"))

