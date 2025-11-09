#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的GUI界面模块
功能：交互式选点、显示投影结果
"""

import cv2
import numpy as np
from typing import List, Tuple, Callable, Optional


class SimpleGUI:
    """简单的GUI界面类"""
    
    def __init__(self, window_name="Surround View Calibration"):
        """
        初始化GUI
        
        Args:
            window_name: 窗口名称
        """
        self.window_name = window_name
        self.points = []  # 选中的点列表
        self.current_image = None
        self.display_image = None
        self.callback = None
        
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击，添加点
            self.points.append((x, y))
            self._update_display()
            print(f"添加点: ({x}, {y}), 共 {len(self.points)} 个点")
            
            if self.callback:
                self.callback(self.points)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击，删除最后一个点
            if len(self.points) > 0:
                removed = self.points.pop()
                self._update_display()
                print(f"删除点: {removed}, 剩余 {len(self.points)} 个点")
                
                if self.callback:
                    self.callback(self.points)
    
    def _update_display(self):
        """更新显示图像"""
        if self.current_image is None:
            return
        
        self.display_image = self.current_image.copy()
        
        # 绘制选中的点
        for i, (x, y) in enumerate(self.points):
            # 绘制点
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            # 绘制序号
            cv2.putText(self.display_image, str(i + 1), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 如果点数>=4，绘制连线（用于透视变换）
        if len(self.points) >= 4:
            pts = np.array(self.points[:4], dtype=np.int32)
            cv2.polylines(self.display_image, [pts], True, (255, 0, 0), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def set_image(self, image):
        """
        设置显示的图像
        
        Args:
            image: 输入图像
        """
        self.current_image = image.copy()
        self._update_display()
    
    def set_callback(self, callback: Callable[[List[Tuple[int, int]]], None]):
        """
        设置点选择回调函数
        
        Args:
            callback: 回调函数，参数为点列表
        """
        self.callback = callback
    
    def get_points(self) -> List[Tuple[int, int]]:
        """获取选中的点"""
        return self.points.copy()
    
    def clear_points(self):
        """清空所有点"""
        self.points = []
        self._update_display()
    
    def set_points(self, points: List[Tuple[int, int]]):
        """
        设置点列表
        
        Args:
            points: 点列表
        """
        self.points = points.copy()
        self._update_display()
    
    def show_result(self, result_image, window_name=None):
        """
        显示结果图像
        
        Args:
            result_image: 结果图像
            window_name: 窗口名称，如果为None则使用默认名称
        """
        if window_name is None:
            window_name = self.window_name + "_Result"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result_image)
        return window_name
    
    def wait_key(self, delay=0) -> int:
        """
        等待按键
        
        Args:
            delay: 等待时间（毫秒），0表示无限等待
            
        Returns:
            按键的ASCII码
        """
        return cv2.waitKey(delay) & 0xFF
    
    def destroy_all_windows(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()
    
    def show_instructions(self):
        """显示使用说明"""
        print("=" * 50)
        print("使用说明:")
        print("  左键点击: 添加点")
        print("  右键点击: 删除最后一个点")
        print("  按 'r' 键: 重新开始选点")
        print("  按 's' 键: 保存当前选点")
        print("  按 'q' 键: 退出")
        print("  至少需要选择4个点用于透视变换")
        print("=" * 50)


def select_points_interactive(image, min_points=4, window_name="Select Points"):
    """
    交互式选点函数
    
    Args:
        image: 输入图像
        min_points: 最少需要的点数
        window_name: 窗口名称
        
    Returns:
        选中的点列表，如果用户取消则返回None
    """
    gui = SimpleGUI(window_name)
    gui.set_image(image)
    gui.show_instructions()
    
    while True:
        key = gui.wait_key(30)
        
        if key == ord('q'):
            # 退出
            points = gui.get_points()
            if len(points) >= min_points:
                return points
            else:
                print(f"点数不足 {min_points} 个，退出")
                return None
        
        elif key == ord('r'):
            # 重新开始
            gui.clear_points()
            print("已清空所有点，重新开始选点")
        
        elif key == ord('s'):
            # 保存当前选点
            points = gui.get_points()
            if len(points) >= min_points:
                print(f"已选择 {len(points)} 个点")
                return points
            else:
                print(f"点数不足 {min_points} 个，至少需要 {min_points} 个点")
        
        elif key == ord('c'):
            # 继续（不返回，继续选点）
            points = gui.get_points()
            print(f"当前已选择 {len(points)} 个点")
    
    gui.destroy_all_windows()
    return None

