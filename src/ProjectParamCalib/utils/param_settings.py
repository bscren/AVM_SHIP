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
        

        # 依据先验参数进行初始化,具体算法见config/calibration_results模块中的示意图说明
        prior_parameters_path = self.config_dir / "calibration_results" / "prior_parameters.yaml"
        fs = cv2.FileStorage(str(prior_parameters_path), cv2.FILE_STORAGE_READ)
        if fs.isOpened():       
            self.ship_pix_size = [fs.getNode('ship_pix_size').mat()[0][0], 
                                  fs.getNode('ship_pix_size').mat()[0][1]]
            
            shift_width = fs.getNode('shift_width').mat()[0][0]
            shift_height = fs.getNode('shift_height').mat()[0][0]
            inn_shift_width = fs.getNode('inn_shift_width').mat()[0][0]
            inn_shift_height = fs.getNode('inn_shift_height').mat()[0][0]
            calib_block_SS_size = [fs.getNode('calib_block_SS_size').mat()[0][0],
                                    fs.getNode('calib_block_SS_size').mat()[0][1]]
            calib_block_FA_size = [fs.getNode('calib_block_FA_size').mat()[0][0],
                                    fs.getNode('calib_block_FA_size').mat()[0][1]]
            
            self.birdview_params = {
                'output_width': int(shift_width*2 + calib_block_FA_size[0]),
                'output_height': int(shift_height*2 + calib_block_FA_size[1]*2 + inn_shift_height*2 + self.ship_pix_size[1])
            }

            fs.release()
        
        # 初始化鸟瞰图各相机标定块的目标点位置,顺序为左上、右上、右下、左下，顺时针定义,单位：像素
        self.avm_dst_points = {
            "front": [(shift_width ,                          shift_height),
                      (shift_width + calib_block_FA_size[0] , shift_height),
                      (shift_width + calib_block_FA_size[0] , shift_height + calib_block_FA_size[1]),
                      (shift_width ,                          shift_height + calib_block_FA_size[1])],
            "back":  [(shift_width ,                          self.birdview_params['output_height'] - shift_height - calib_block_FA_size[1]),
                      (shift_width + calib_block_FA_size[0] , self.birdview_params['output_height'] - shift_height - calib_block_FA_size[1]),
                      (shift_width + calib_block_FA_size[0] , self.birdview_params['output_height'] - shift_height),
                      (shift_width ,                          self.birdview_params['output_height'] - shift_height)],
            "left_front":
                    [(shift_width,                          shift_height),
                     (shift_width + calib_block_SS_size[0], shift_height),
                     (shift_width + calib_block_SS_size[0], shift_height + calib_block_SS_size[1]),
                     (shift_width,                          shift_height + calib_block_SS_size[1])],
            "left_back":
                    [(shift_width,                          self.birdview_params['output_height'] - shift_height - calib_block_SS_size[1]),
                     (shift_width + calib_block_SS_size[0], self.birdview_params['output_height'] - shift_height - calib_block_SS_size[1]),
                     (shift_width + calib_block_SS_size[0], self.birdview_params['output_height'] - shift_height),
                     (shift_width,                          self.birdview_params['output_height'] - shift_height)],
            "right_front":
                    [(self.birdview_params['output_width'] - shift_width - calib_block_SS_size[0],      shift_height),
                     (self.birdview_params['output_width'] - shift_width ,                              shift_height),
                     (self.birdview_params['output_width'] - shift_width ,                              shift_height + calib_block_SS_size[1]),
                     (self.birdview_params['output_width'] - shift_width - calib_block_SS_size[0],      shift_height + calib_block_SS_size[1])],
            "right_back":
                    [(self.birdview_params['output_width'] - shift_width - calib_block_SS_size[0],      self.birdview_params['output_height'] - shift_height - calib_block_SS_size[1]),
                     (self.birdview_params['output_width'] - shift_width ,                              self.birdview_params['output_height'] - shift_height - calib_block_SS_size[1]),
                     (self.birdview_params['output_width'] - shift_width ,                              self.birdview_params['output_height'] - shift_height),
                     (self.birdview_params['output_width'] - shift_width - calib_block_SS_size[0],      self.birdview_params['output_height'] - shift_height)]
        }

        # 初始化各个摄像头投影图像的大小，由 标定块大小+向外看的距离+标定块内侧边缘 与船只的距离决定
        self.proj_image_sizes = {
            "front": (self.birdview_params['output_width'],
                      shift_height + calib_block_FA_size[1] + inn_shift_height),
            "back":  (self.birdview_params['output_width'],
                      shift_height + calib_block_FA_size[1] + inn_shift_height),
            "left_front": (shift_width + calib_block_SS_size[0] + inn_shift_width,
                          shift_height + calib_block_SS_size[1] + inn_shift_height),
            "left_back":  (shift_width + calib_block_SS_size[0] + inn_shift_width,
                          shift_height + calib_block_SS_size[1] + inn_shift_height),
            "right_front":  (shift_width + calib_block_SS_size[0] + inn_shift_width,
                          shift_height + calib_block_SS_size[1] + inn_shift_height),
            "right_back":   (shift_width + calib_block_SS_size[0] + inn_shift_width,
                          shift_height + calib_block_SS_size[1] + inn_shift_height),
        }

        # 初始化各个摄像头投影图像的标定块源点位置,顺序为左上、右上、右下、左下，顺时针定义,单位：像素
        self.proj_dst_points = {
            "front": [shift_width , shift_height,
                    shift_width + calib_block_FA_size[0] , shift_height,
                    shift_width + calib_block_FA_size[0] , shift_height + calib_block_FA_size[1],
                    shift_width , shift_height + calib_block_FA_size[1]],
            "back": [shift_width, inn_shift_height,
                    shift_width + calib_block_FA_size[0], inn_shift_height,
                    shift_width + calib_block_FA_size[0], inn_shift_height + calib_block_FA_size[1],
                    shift_width, inn_shift_height + calib_block_FA_size[1]],
            "left_front": 
                    [shift_width , shift_height,
                    shift_width + calib_block_SS_size[0] , shift_height,
                    shift_width + calib_block_SS_size[0] , shift_height + calib_block_SS_size[1],
                    shift_width , shift_height + calib_block_SS_size[1]],
            "left_back": 
                    [shift_width , inn_shift_height,
                    shift_width + calib_block_SS_size[0] , inn_shift_height,
                    shift_width + calib_block_SS_size[0] , inn_shift_height + calib_block_SS_size[1],
                    shift_width , inn_shift_height + calib_block_SS_size[1]
                    ],
            "right_front": 
                    [inn_shift_width , shift_height,
                    inn_shift_width + calib_block_SS_size[0] , shift_height,
                    inn_shift_width + calib_block_SS_size[0] , shift_height + calib_block_SS_size[1],
                    inn_shift_width , shift_height + calib_block_SS_size[1]],
            "right_back": 
                    [inn_shift_width , inn_shift_height,
                    inn_shift_width + calib_block_SS_size[0] , inn_shift_height,
                    inn_shift_width + calib_block_SS_size[0] , inn_shift_height + calib_block_SS_size[1],
                    inn_shift_width , inn_shift_height + calib_block_SS_size[1]
                    ]
        }
    
        # =============================DEBUG========================================
        # 绘制各摄像头投影图像大小示意图
        for camera, size in self.proj_image_sizes.items():
            debug_image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
            pts = np.array(self.proj_dst_points[camera]).reshape(-1, 2).astype(np.int32)
            cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            debug_image_path = self.config_dir / "calibration_results" / f"{camera}_proj_image_size_debug.jpg"
            cv2.imwrite(str(debug_image_path), debug_image)
            print(f"{camera} 投影图像大小示意图已保存到: {debug_image_path}")
        # ================================DEBUG====================================
        # 绘制标定块位置示意图
        debug_image = np.ones((self.birdview_params['output_height'], self.birdview_params['output_width'], 3), dtype=np.uint8)
        debug_image[:] = 255  # 将背景设置为白色
        for camera, points in self.avm_dst_points.items():
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # 在标定块中心写上摄像头名称
            center_x = int(sum([p[0] for p in points]) / 4)
            center_y = int(sum([p[1] for p in points]) / 4)
            cv2.putText(debug_image, camera, (center_x - 30, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        debug_image_path = self.config_dir / "calibration_results" / "avm_calib_block_positions_debug.jpg"
        cv2.imwrite(str(debug_image_path), debug_image)
        print(f"标定块位置示意图已保存到: {debug_image_path}")
        # ================================DEBUG====================================


        
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
