#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼眼相机模型处理模块
功能：处理鱼眼相机的畸变校正和投影变换
"""

import cv2
import numpy as np
from .param_settings import ParamSettings


class FisheyeCamera:
    """鱼眼相机处理类"""
    
    def __init__(self, camera_name, param_settings):
        """
        初始化鱼眼相机
        
        Args:
            camera_name: 相机名称
            param_settings: ParamSettings 实例
        """
        self.camera_name = camera_name
        self.param_settings = param_settings
        
        # 加载相机参数
        self.camera_params = param_settings.get_camera_params(camera_name)
        self.camera_matrix = self.camera_params['camera_matrix']
        self.dist_coeffs = self.camera_params['dist_coeffs']
        self.image_width = self.camera_params['image_width']
        self.image_height = self.camera_params['image_height']
        self.scale_x = self.camera_params.get('scale_x', 1.0)
        self.scale_y = self.camera_params.get('scale_y', 1.0)
        self.translate_x = self.camera_params.get('translate_x', 0.0)
        self.translate_y = self.camera_params.get('translate_y', 0.0)
        
        # 初始化映射矩阵
        self.map_x = None
        self.map_y = None
        self.new_camera_matrix = None
        
    def undistort_image(self, image, alpha=0):
        """
        校正鱼眼图像畸变
        
        Args:
            image: 输入图像
            alpha: 控制校正后图像的有效区域 (0-1)
                   0: 只保留有效像素
                   1: 保留所有像素，可能有黑色边界
                   
        Returns:
            校正后的图像
        """

        # 原作者是这样写的，后期可能会用到
        # def update_undistort_maps(self):
        #     new_matrix = self.camera_matrix.copy()
        #     new_matrix[0, 0] *= self.scale_xy[0]
        #     new_matrix[1, 1] *= self.scale_xy[1]
        #     new_matrix[0, 2] += self.shift_xy[0]
        #     new_matrix[1, 2] += self.shift_xy[1]
        #     width, height = self.resolution

        #     self.undistort_maps = cv2.fisheye.initUndistortRectifyMap(
        #         self.camera_matrix,
        #         self.dist_coeffs,
        #         np.eye(3),
        #         new_matrix,
        #         (width, height),
        #         cv2.CV_16SC2
        #     )
        #     return self
        if self.map_x is None or self.map_y is None:
            # 计算新的相机矩阵和映射
            h, w = image.shape[:2]
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (w, h),
                alpha,
                (w, h)
            )
            
            # 计算映射矩阵
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.new_camera_matrix,
                (w, h),
                cv2.CV_32FC1
            )
        
        # 应用映射
        undistorted = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
        
        return undistorted, self.map_x, self.map_y
    
    def undistort_fisheye(self, image, balance=0.0):
        """
        使用鱼眼模型校正（如果标定使用的是鱼眼模型）
        
        Args:
            image: 输入图像
            balance: 平衡参数 (0.0-1.0)
            
        Returns:
            校正后的图像
        """
        h, w = image.shape[:2]
        
        # 鱼眼校正映射
        if self.map_x is None or self.map_y is None:
            K = self.camera_matrix
            D = self.dist_coeffs
            
            # 计算新的相机矩阵
            dim = (w, h)
            Knew = K.copy()
            if balance != 0:
                Knew[(0, 1), (0, 1)] = K[(0, 1), (0, 1)] / balance
            
            # 计算映射
            self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), Knew, dim, cv2.CV_16SC2
            )
        
        # 应用映射
        undistorted = cv2.remap(image, self.map_x, self.map_y, 
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        return undistorted
    
    def get_undistorted_points(self, points):
        """
        校正点坐标
        
        Args:
            points: 输入点坐标 (n, 2) 或 (n, 1, 2)
            
        Returns:
            校正后的点坐标
        """
        points = np.array(points, dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 2:
            points = points.reshape(-1, 1, 2)
        
        # 使用undistortPoints进行点校正
        undistorted_points = cv2.undistortPoints(
            points,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.new_camera_matrix if self.new_camera_matrix is not None else self.camera_matrix
        )
        
        return undistorted_points.reshape(-1, 2)
    
    def get_camera_matrix(self):
        """获取相机内参矩阵"""
        return self.camera_matrix
    
    def get_dist_coeffs(self):
        """获取畸变系数"""
        return self.dist_coeffs
    
    def get_new_camera_matrix(self):
        """获取新的相机矩阵"""
        return self.new_camera_matrix if self.new_camera_matrix is not None else self.camera_matrix

