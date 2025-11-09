#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成投影映射矩阵脚本
功能：交互式选点、计算投影映射、显示结果、保存参数
使用：python3 run_get_projection_maps.py --camera_name front --image_path ./test_images/front.jpg
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from utils.param_settings import ParamSettings
from utils.fisheye_camera import FisheyeCamera
from utils.simple_gui import SimpleGUI, select_points_interactive
from utils.path_manager import get_calibration_dir, get_config_base_dir


class ProjectionMapper:
    """投影映射类"""
    
    def __init__(self, camera_name, param_settings, output_width=1000, output_height=1000):
        """
        初始化投影映射器
        
        Args:
            camera_name: 相机名称
            param_settings: ParamSettings 实例
            output_width: 输出图像宽度
            output_height: 输出图像高度
        """
        self.camera_name = camera_name
        self.param_settings = param_settings
        self.output_width = output_width
        self.output_height = output_height
        
        # 初始化鱼眼相机
        self.fisheye_camera = FisheyeCamera(camera_name, param_settings)
        
        # 投影映射矩阵
        self.map_x = None
        self.map_y = None
        self.homography = None
        
    def select_birdview_points(self, image, load_existing=True):
        """
        选择鸟瞰图对应点
        
        Args:
            image: 输入图像
            load_existing: 是否尝试加载已存在的点
            
        Returns:
            tuple: (src_points, dst_points) 源点和目标点
        """
        # 尝试加载已存在的点
        if load_existing:
            src_points, dst_points = self.param_settings.load_birdview_points(self.camera_name)
            if src_points is not None and dst_points is not None:
                print(f"加载已存在的 {self.camera_name} 相机选点")
                return src_points, dst_points
        
        # 交互式选点
        print(f"\n请为 {self.camera_name} 相机选择4个对应点")
        print("顺序：左上、右上、右下、左下（顺时针点击特征点）")
        
        # 选择源图像中的点
        src_points = select_points_interactive(
            image, 
            min_points=4, 
            window_name=f"Select Source Points - {self.camera_name}"
        )
        
        if src_points is None or len(src_points) < 4:
            raise ValueError("至少需要选择4个点")
        
        src_points = np.array(src_points[:4], dtype=np.float32)
        
        # 定义目标鸟瞰图中的点（默认矩形）
        # 可以根据实际车辆尺寸调整
        margin = 100  # 边距
        # 依据相机的类型，预设对应的目标点默认坐标
        if self.camera_name == 'front':
            margin_top = 50
            dst_points = np.array([
                [margin, margin_top],  # 左上
                [self.output_width - margin, margin_top],  # 右上
                [self.output_width - margin, self.output_height - margin],  # 右下
                [margin, self.output_height - margin]  # 左下
            ], dtype=np.float32)
        elif self.camera_name == 'back':
            margin_bottom = self.output_height - 50
            dst_points = np.array([
                [margin, margin],  # 左上
                [self.output_width - margin, margin],  # 右上
                [self.output_width - margin, margin_bottom],  # 右下
                [margin, margin_bottom]  # 左下
            ], dtype=np.float32)

        # 可以选择自定义目标点
        print(f"\n目标点（鸟瞰图）默认设置为:")
        print(f"  左上: ({dst_points[0][0]:.0f}, {dst_points[0][1]:.0f})")
        print(f"  右上: ({dst_points[1][0]:.0f}, {dst_points[1][1]:.0f})")
        print(f"  右下: ({dst_points[2][0]:.0f}, {dst_points[2][1]:.0f})")
        print(f"  左下: ({dst_points[3][0]:.0f}, {dst_points[3][1]:.0f})")
        
        user_input = input("是否自定义目标点？(y/n，默认n): ").strip().lower()
        if user_input == 'y':
            print("\n请为目标鸟瞰图选择4个对应点")
            dst_image = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            # 绘制网格帮助定位
            self._draw_grid(dst_image)
            dst_points_list = select_points_interactive(
                dst_image,
                min_points=4,
                window_name=f"Select Destination Points - {self.camera_name}"
            )
            if dst_points_list is not None and len(dst_points_list) >= 4:
                dst_points = np.array(dst_points_list[:4], dtype=np.float32)
        
        # 保存选点
        self.param_settings.save_birdview_points(self.camera_name, src_points, dst_points)
        
        return src_points, dst_points
    
    def _draw_grid(self, image, grid_size=50):
        """绘制网格"""
        h, w = image.shape[:2]
        color = (50, 50, 50)
        # 垂直线
        for x in range(0, w, grid_size):
            cv2.line(image, (x, 0), (x, h), color, 2)
        # 水平线
        for y in range(0, h, grid_size):
            cv2.line(image, (0, y), (w, y), color, 2)
    
    def compute_homography(self, src_points, dst_points):
        """
        计算单应性矩阵
        
        Args:
            src_points: 源点 (4, 2)
            dst_points: 目标点 (4, 2)
            
        Returns:
            单应性矩阵 (3, 3)
        """
        self.homography = cv2.getPerspectiveTransform(src_points, dst_points)
        return self.homography
    
    def compute_projection_maps(self, image_shape, src_points, dst_points):
        """
        计算投影映射矩阵
        
        Args:
            image_shape: 输入图像形状 (height, width)
            src_points: 源点
            dst_points: 目标点
            
        Returns:
            tuple: (map_x, map_y) 映射矩阵
        """
        h, w = image_shape[:2]
        
        # 计算单应性矩阵
        homography = self.compute_homography(src_points, dst_points)
        self.homography = homography
        
        # 创建输出图像的坐标网格
        dst_y, dst_x = np.mgrid[0:self.output_height, 0:self.output_width].astype(np.float32)
        
        # 将目标坐标转换为齐次坐标
        dst_coords = np.stack([dst_x.flatten(), dst_y.flatten(), np.ones(self.output_width * self.output_height)])
        
        # 应用逆单应性变换，得到源图像坐标
        src_coords = np.linalg.inv(homography) @ dst_coords
        src_coords = src_coords / src_coords[2, :]  # 归一化
        
        # 提取x和y坐标
        src_x = src_coords[0, :].reshape(self.output_height, self.output_width)
        src_y = src_coords[1, :].reshape(self.output_height, self.output_width)
        
        # 限制坐标范围
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)
        
        self.map_x = src_x.astype(np.float32)
        self.map_y = src_y.astype(np.float32)
        
        return self.map_x, self.map_y
    
    def apply_projection(self, image):
        """
        应用投影变换
        
        Args:
            image: 输入图像
            
        Returns:
            投影后的图像
        """
        if self.map_x is None or self.map_y is None:
            raise ValueError("投影映射未计算，请先调用 compute_projection_maps")
        
        # 使用remap进行投影
        projected = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return projected
    
    def save_projection_maps(self, output_file=None):
        """保存投影映射"""
        if self.map_x is None or self.map_y is None:
            raise ValueError("投影映射未计算")
        
        self.param_settings.save_projection_maps(self.camera_name, self.map_x, self.map_y, output_file)

    def save_mask(self, output_file=None):
        """保存掩码"""
        if self.mask is None:
            raise ValueError("掩码未计算")
        self.param_settings.save_mask(self.camera_name, self.mask, output_file)

def main():
    parser = argparse.ArgumentParser(description='生成投影映射矩阵')
    parser.add_argument('--camera_name', type=str, required=False,
                        choices=['front', 'back', 'left', 'right'],
                        help='相机名称')
    parser.add_argument('--image_path', type=str, required=False,
                        help='测试single图像路径')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='测试图像目录（如果提供，则忽略 image_path）')
    parser.add_argument('--calib_dir', type=str, default=None,
                        help='标定文件目录（默认使用统一配置目录）')
    parser.add_argument('--config_dir', type=str, default=None,
                        help='配置文件目录（默认使用统一配置目录）')
    parser.add_argument('--output_width', type=int, default=500,
                        help='输出图像宽度')
    parser.add_argument('--output_height', type=int, default=1000,
                        help='输出图像高度')
    parser.add_argument('--undistort', action='store_true',
                        help='是否先进行畸变校正')
    parser.add_argument('--load_existing', action='store_false',
                        help='不加载已存在的选点，强制重新选点')
    
     # 解析参数
    
    args = parser.parse_args()


    # ================================ DEBUG ================================
    # 调试时可直接设置参数
    args.camera_name = 'front'
    # 使用相对于项目根目录（以脚本文件为基准）的相对路径
    args.image_path = str(Path(__file__).resolve().parents[2] / "config" / "images" /f"cam_{args.camera_name}.jpg")
    args.images_dir = str(Path(__file__).resolve().parents[2] / "config" / "images")
    args.calib_dir = str(Path(__file__).resolve().parents[2] / "config" / "calibration_results")
    args.config_dir = str(Path(__file__).resolve().parents[2] / "config")
    args.output_width = 500
    args.output_height = 1000
    args.undistort = True
    args.load_existing = True
    # ===================================================================== 

    # 检查图像文件
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在: {args.image_path}")
        return
    
    # 初始化参数设置（使用统一路径管理）
    param_settings = ParamSettings(config_dir=args.config_dir)
    
    # 加载相机标定参数
    if args.calib_dir:
        # 使用指定的标定目录
        calib_file = os.path.join(args.calib_dir, f"cam_{args.camera_name}.yaml")
    else:
        # 使用统一路径管理工具
        from utils.path_manager import get_calibration_file
        calib_file = str(get_calibration_file(args.camera_name))
    
    if not os.path.exists(calib_file):
        print(f"警告: 标定文件不存在: {calib_file}")
        print("将使用默认参数（可能不准确）")
    else:
        try:
            param_settings.load_camera_calibration(args.camera_name, calib_file)
        except Exception as e:
            print(f"加载标定参数失败: {e}")
            return
    
    # 读取图像
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"错误: 无法读取图像: {args.image_path}")
        return
    
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 畸变校正（可选）
    if args.undistort:
        try:
            fisheye_camera = FisheyeCamera(args.camera_name, param_settings)
            image = fisheye_camera.undistort_image(image)
            print("已进行畸变校正")
        except Exception as e:
            print(f"畸变校正失败: {e}")
    
    # 创建投影映射器
    mapper = ProjectionMapper(
        args.camera_name,
        param_settings,
        output_width=args.output_width,
        output_height=args.output_height
    )
    
    # 选择对应点
    try:
        src_points, dst_points = mapper.select_birdview_points(image, args.load_existing)
        print(f"\n源点:")
        for i, pt in enumerate(src_points):
            print(f"  点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
        print(f"\n目标点:")
        for i, pt in enumerate(dst_points):
            print(f"  点{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    except Exception as e:
        print(f"选点失败: {e}")
        return
    
    # 计算投影映射
    print("\n计算投影映射矩阵...")
    map_x, map_y = mapper.compute_projection_maps(image.shape, src_points, dst_points)
    print("投影映射计算完成")
    
    # 应用投影
    print("应用投影变换...")
    projected_image = mapper.apply_projection(image)
    
    # 显示结果
    gui = SimpleGUI("Projection Result")
    gui.set_image(projected_image)
    gui.show_result(image, "Original Image")
    
    print("\n按任意键查看结果，按 'q' 退出")
    
    while True:
        key = gui.wait_key(30) # 30ms表示刷新间隔
        if key == ord('q'):
            break
    
    # 保存投影映射
    save = input("\n是否保存投影映射？(y/n，默认y): ").strip().lower()
    if save != 'n':
        mapper.save_projection_maps()
        print("投影映射已保存")
    
    # 保存结果图像
    save_image = input("是否保存投影结果图像？(y/n，默认n): ").strip().lower()
    if save_image == 'y':
        output_image_path = os.path.join(args.images_dir, f"{args.camera_name}_projected.jpg")
        cv2.imwrite(output_image_path, projected_image)
        print(f"结果图像已保存到: {output_image_path}")
    

    # 在结果图像上使用鼠标点击单个点，绘制不规则多边形，将其区域设为白色掩码，区域外设为黑色掩码
    mask_choice = input("是否为投影结果图像创建不规则多边形掩码？(y/n，默认n): ").strip().lower()
    if mask_choice == 'y':
        mask_points = select_points_interactive(
            projected_image, 
            min_points=4, 
            window_name="按顺序人工选取多边形掩码区域"
            )
        if mask_points is not None and len(mask_points) >= 4:
            print("生成掩码...")
            mapper.mask = param_settings.compute_mask_from_points( projected_image, mask_points)
            mapper.save_mask(args.images_dir)
            gui.show_result(mapper.mask, "Hand-drawn Mask")
            print("\n按任意键查看结果，按 'q' 退出")
            while True:
                key = gui.wait_key(30) # 30ms表示刷新间隔
                if key == ord('q'):
                    break
            
        
    gui.destroy_all_windows()
    print("完成！")


if __name__ == '__main__':
    main()

