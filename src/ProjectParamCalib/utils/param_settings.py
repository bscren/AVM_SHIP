#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数设置和加载模块
功能：加载相机标定参数、投影参数等配置
"""

import yaml
import numpy as np
import os
from pathlib import Path
from .path_manager import get_config_base_dir, get_calibration_dir, get_projection_dir
import cv2  

class ParamSettings:
    """参数设置类"""
    
    def __init__(self, config_dir=None):
        """
        初始化参数设置
        
        Args:
            config_dir: 配置文件目录，如果为None则使用默认路径（支持环境变量）
        """
        if config_dir is None:
            # 使用路径管理工具获取默认配置目录
            self.config_dir = get_config_base_dir()
        else:
            self.config_dir = Path(config_dir)
            self.config_dir.mkdir(exist_ok=True)
        
        # 相机标定参数
        self.camera_params = {}
        
        # 投影映射参数
        self.projection_maps = {}
        
        # 鸟瞰图参数
        self.birdview_params = {
            'output_width': 1000,
            'output_height': 1000,
            'pixel_per_meter': 50,  # 每米像素数
        }
        
    def load_camera_calibration(self, camera_name, calib_file=None):
        """
        加载相机标定参数
        
        Args:
            camera_name: 相机名称 (front, back, left, right)
            calib_file: 标定文件路径，如果为None则从默认路径加载
            
        Returns:
            dict: 包含相机内参和畸变系数的字典
        """
        if calib_file is None:
            # 使用路径管理工具获取标定文件路径
            from .path_manager import get_calibration_file
            calib_file = get_calibration_file(camera_name)
        
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"标定文件不存在: {calib_file}")
        
        with open(calib_file, 'r') as f:
            fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        
        # 解析相机内参矩阵
        cam_matrix_data = fs.getNode("camera_matrix").mat()
        camera_matrix = np.array(cam_matrix_data).reshape(3, 3)
        
        # 解析畸变系数
        dist_data = fs.getNode("distortion_coefficients").mat()
        dist_coeffs = np.array(dist_data).flatten()
        
        # 获取图像尺寸
        resolution_data = fs.getNode("resolution").mat()
        image_width = int(resolution_data[0][0])
        image_height = int(resolution_data[1][0])
        
        self.camera_params[camera_name] = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'image_width': image_width,
            'image_height': image_height
        }
        
        print(f"成功加载 {camera_name} 相机标定参数")
        return self.camera_params[camera_name]
    
    def get_camera_params(self, camera_name):
        """获取相机参数"""
        if camera_name not in self.camera_params:
            raise ValueError(f"相机 {camera_name} 的参数未加载")
        return self.camera_params[camera_name]
    
    def save_projection_maps(self, camera_name, map_x, map_y, output_file=None):
        """
        保存投影映射矩阵
        
        Args:
            camera_name: 相机名称
            map_x: X方向映射矩阵
            map_y: Y方向映射矩阵
            output_file: 输出文件路径
        """
        if output_file is None:
            # 使用路径管理工具获取投影映射文件路径
            from .path_manager import get_projection_file
            output_file = get_projection_file(camera_name, "maps")
        
        projection_data = {
            'camera_name': camera_name,
            'map_x': {
                'shape': list(map_x.shape),
                'data': map_x.flatten().tolist(),
                'dtype': str(map_x.dtype)
            },
            'map_y': {
                'shape': list(map_y.shape),
                'data': map_y.flatten().tolist(),
                'dtype': str(map_y.dtype)
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(projection_data, f, default_flow_style=False)
        
        self.projection_maps[camera_name] = {
            'map_x': map_x,
            'map_y': map_y
        }
        
        print(f"投影映射已保存到: {output_file}")
    
    def load_projection_maps(self, camera_name, map_file=None):
        """
        加载投影映射矩阵
        
        Args:
            camera_name: 相机名称
            map_file: 映射文件路径
            
        Returns:
            tuple: (map_x, map_y)
        """
        if map_file is None:
            # 使用路径管理工具获取投影映射文件路径
            from .path_manager import get_projection_file
            map_file = get_projection_file(camera_name, "maps")
        
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"投影映射文件不存在: {map_file}")
        
        with open(map_file, 'r') as f:
            map_data = yaml.safe_load(f)
        
        # 恢复map_x
        map_x_shape = tuple(map_data['map_x']['shape'])
        map_x_data = np.array(map_data['map_x']['data'], dtype=np.float32)
        map_x = map_x_data.reshape(map_x_shape)
        
        # 恢复map_y
        map_y_shape = tuple(map_data['map_y']['shape'])
        map_y_data = np.array(map_data['map_y']['data'], dtype=np.float32)
        map_y = map_y_data.reshape(map_y_shape)
        
        self.projection_maps[camera_name] = {
            'map_x': map_x,
            'map_y': map_y
        }
        
        print(f"成功加载 {camera_name} 投影映射")
        return map_x, map_y
    
    def get_projection_maps(self, camera_name):
        """获取投影映射"""
        if camera_name not in self.projection_maps:
            raise ValueError(f"相机 {camera_name} 的投影映射未加载")
        return self.projection_maps[camera_name]['map_x'], self.projection_maps[camera_name]['map_y']
    
    def save_birdview_points(self, camera_name, src_points, dst_points, output_file=None):
        """
        保存鸟瞰图对应点
        
        Args:
            camera_name: 相机名称
            src_points: 源图像中的点 (n, 2)
            dst_points: 目标鸟瞰图中的点 (n, 2)
            output_file: 输出文件路径
        """
        if output_file is None:
            # 使用路径管理工具获取点文件路径
            from .path_manager import get_projection_file
            output_file = get_projection_file(camera_name, "points")
        
        points_data = {
            'camera_name': camera_name,
            'src_points': src_points.tolist(),
            'dst_points': dst_points.tolist()
        }
        # 需要改为opencv格式的yaml保存方法
        with open(output_file, 'w') as f:
            yaml.dump(points_data, f, default_flow_style=False)
        
        print(f"鸟瞰图对应点已保存到: {output_file}")
    
    def load_birdview_points(self, camera_name, points_file=None):
        """
        加载鸟瞰图对应点
        
        Args:
            camera_name: 相机名称
            points_file: 点文件路径
        Returns:
            tuple: (src_points, dst_points)
        """
        if points_file is None:
            # 使用路径管理工具获取点文件路径
            from .path_manager import get_projection_file
            points_file = get_projection_file(camera_name, "points")
        
        if not os.path.exists(points_file):
            return None, None
        
        with open(points_file, 'r') as f:
            points_data = yaml.safe_load(f)
        
        src_points = np.array(points_data['src_points'], dtype=np.float32)
        dst_points = np.array(points_data['dst_points'], dtype=np.float32)
        
        return src_points, dst_points

