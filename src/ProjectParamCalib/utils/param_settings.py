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
        
        # OpenCV FileStorage expects a string filename; ensure we pass str and don't open the file separately
        fs = cv2.FileStorage(str(calib_file), cv2.FILE_STORAGE_READ)
        
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
    
    def save_calibration_maps(self, camera_name, map_x, map_y, output_file=None):
        """
        保存标定映射矩阵
        
        Args:
            camera_name: 相机名称
            map_x: X方向映射矩阵
            map_y: Y方向映射矩阵
            output_file: 输出文件路径
        """
        if output_file is None:
            # 使用路径管理工具获取标定映射文件路径
            from .path_manager import get_projection_file
            output_file = get_projection_file(camera_name, "calib")

        # 确保路径存在并传入字符串给 OpenCV FileStorage
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 使用OpenCV的FileStorage写入YAML格式
        fs = cv2.FileStorage(str(output_file), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_name", camera_name)
        fs.write("map_x", map_x)
        fs.write("map_y", map_y)
        fs.release()

        self.camera_params[camera_name]['calib_map_x'] = map_x
        self.camera_params[camera_name]['calib_map_y'] = map_y
        
        print(f"标定映射已保存到: {output_file}")

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
            output_file = get_projection_file(camera_name, "project")

        # 确保路径存在并传入字符串给 OpenCV FileStorage
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 使用OpenCV的FileStorage写入YAML格式
        fs = cv2.FileStorage(str(output_file), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_name", camera_name)
        fs.write("map_x", map_x)
        fs.write("map_y", map_y)
        fs.release()

        # projection_data = {
        #     'camera_name': camera_name,
        #     'map_x': {
        #         'shape': list(map_x.shape),
        #         'data': map_x.flatten().tolist(),
        #         'dtype': str(map_x.dtype)
        #     },
        #     'map_y': {
        #         'shape': list(map_y.shape),
        #         'data': map_y.flatten().tolist(),
        #         'dtype': str(map_y.dtype)
        #     }
        # }
        
        # with open(output_file, 'w') as f:
        #     yaml.dump(projection_data, f, default_flow_style=False)
        
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
        
        # 使用 OpenCV FileStorage 读取 opencv 格式的 YAML
        fs = cv2.FileStorage(str(map_file), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise IOError(f"无法打开投影映射文件: {map_file}")

        map_x = fs.getNode('map_x').mat()
        map_y = fs.getNode('map_y').mat()
        fs.release()
         
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
        """
        if output_file is None:
            # 使用路径管理工具获取点文件路径
            from .path_manager import get_projection_file
            output_file = get_projection_file(camera_name, "points")
        
        # 确保路径存在并传入字符串给 OpenCV FileStorage
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 使用OpenCV的FileStorage写入YAML格式
        fs = cv2.FileStorage(str(output_file), cv2.FILE_STORAGE_WRITE)
        fs.write("camera_name", camera_name)
        fs.write("src_points", src_points)
        fs.write("dst_points", dst_points)
        fs.release()


        # points_data = {
        #     'camera_name': camera_name,
        #     'src_points': src_points.tolist(),
        #     'dst_points': dst_points.tolist()
        # }
        # # 需要改为opencv格式的yaml保存方法
        # with open(output_file, 'w') as f:
        #     yaml.dump(points_data, f, default_flow_style=False)
        
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
        
        # 使用 OpenCV FileStorage 读取保存的 src_points / dst_points（opencv yaml 格式）
        fs = cv2.FileStorage(str(points_file), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            # 如果无法用 FileStorage 打开，尝试回退到 yaml 解析以兼容旧格式
            with open(points_file, 'r') as f:
                points_data = yaml.safe_load(f)
            src_points = np.array(points_data['src_points'], dtype=np.float32)
            dst_points = np.array(points_data['dst_points'], dtype=np.float32)
            return src_points, dst_points

        src_node = fs.getNode('src_points')
        dst_node = fs.getNode('dst_points')
        src = src_node.mat()
        dst = dst_node.mat()
        fs.release()

        return src, dst

    def compute_mask_from_points(self, image, points):
        """
        在结果图像上使用鼠标点击单个点，绘制不规则多边形，将其区域设为白色掩码，区域外设为黑色掩码
        """
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        return mask
    
    def save_mask(self, camera_name, mask, output_file=None):
        """
        保存掩码图像
        
        Args:
            camera_name: 相机名称
            mask: 掩码图像
            output_file: 输出文件路径
        """
        if output_file is None:
            # 使用路径管理工具获取掩码文件路径
            from .path_manager import get_projection_file
            output_file = get_projection_file(camera_name, "mask")
        else:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file = str(output_file / f"mask_{camera_name}.jpg")

        cv2.imwrite(output_file, mask)
        print(f"掩码图像已保存到: {output_file}")
