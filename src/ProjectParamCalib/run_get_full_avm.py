import cv2
import numpy as np
import yaml
import os
from pathlib import Path
import argparse

class AVMImageStitcher:
    """环视图像拼接器，用于处理相机投影和图像拼接"""
    
    def __init__(self, base_dir=None):
        """初始化拼接器
        
        Args:
            base_dir: 项目根目录，默认为当前文件所在目录的父目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parents[2]  # 假设脚本位于src目录下
        else:
            self.base_dir = Path(base_dir)
            
        # 定义路径
        self.images_dir = self.base_dir / "config" / "images"
        self.projection_dir = self.base_dir / "config" / "projection_maps"
        self.calibration_dir = self.base_dir / "config" / "projection_maps"  # 标定map也存放在此目录下
        
        # 确保目录存在
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.projection_dir, exist_ok=True)
        os.makedirs(self.calibration_dir, exist_ok=True)

        # 存储加载的投影映射
        self.calibration_maps = {}  # 存储加载的标定映射
        self.projection_maps = {}  # 存储加载的投影映射

    def load_calibration_map(self, camera_name):
        """加载指定相机的标定参数
        
        Args:
            camera_name: 相机名称 (front, back)
        """
        calib_file = self.calibration_dir / f"calibration_{camera_name}.yaml"

        if not calib_file.exists():
            raise FileNotFoundError(f"标定参数文件不存在: {calib_file}")
            
        # 读取opencv形式的yaml文件中的mapx和mapy
        fs = cv2.FileStorage(str(calib_file), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise IOError(f"无法打开标定文件: {calib_file}")

        map_x_node = fs.getNode('map_x')
        map_y_node = fs.getNode('map_y')
        map_x = map_x_node.mat()
        map_y = map_y_node.mat()
        fs.release()
 
        self.calibration_maps[camera_name] = (map_x, map_y)

        return map_x, map_y

    def load_projection_map(self, camera_name):
        """加载指定相机的投影映射矩阵
        
        Args:
            camera_name: 相机名称 (front, back)
        """
        projection_file = self.projection_dir / f"projection_maps_{camera_name}.yaml"
        
        if not projection_file.exists():
            raise FileNotFoundError(f"投影矩阵文件不存在: {projection_file}")
            
        fs = cv2.FileStorage(str(projection_file), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise IOError(f"无法打开投影矩阵文件: {projection_file}")

        map_x = fs.getNode('map_x').mat()
        map_y = fs.getNode('map_y').mat()
        fs.release()
         
        self.projection_maps[camera_name] = (map_x, map_y)
        return map_x, map_y
    
    def load_image(self, camera_name):
        """加载指定相机的原始图像
        
        Args:
            camera_name: 相机名称 (front, back)
        """
        image_path = self.images_dir / f"cam_{camera_name}.jpg"
        
        if not image_path.exists():
            raise FileNotFoundError(f"原始图像文件不存在: {image_path}")
            
        image = cv2.imread(str(image_path))
        return image
    
    def load_mask(self, camera_name):
        """加载指定相机的掩码图像
        
        Args:
            camera_name: 相机名称 (front, back)
        """
        mask_path = self.images_dir / f"mask_{camera_name}.jpg"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"掩码图像文件不存在: {mask_path}")
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask

    def calibrate_image(self, camera_name, image = None):
        """对图像进行畸变校正（假设已知内参和畸变参数）
        
        Args:
            image: 输入图像
        """
        # 加载图像（如果未提供）
        if image is None:
            image = self.load_image(camera_name)

        if camera_name not in self.calibration_maps:
            map_x, map_y = self.load_calibration_map(camera_name)

        # 应用畸变校正
        undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # 边界填充黑色
        )
        return undistorted

    def project_image(self, camera_name, image):
        """对图像应用投影变换
        
        Args:
            camera_name: 相机名称 (front, back)
            image: 可选，矫正图像，如果为None则自动加载
        """
        
            
        # 加载投影映射（如果未加载）
        if camera_name not in self.projection_maps:
            self.load_projection_map(camera_name)
            
        map_x, map_y = self.projection_maps[camera_name]
        
        # 应用投影变换
        projected = cv2.remap(
            image, 
            map_x, 
            map_y, 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # 边界填充黑色
        )
        
        return projected
    
    def apply_mask(self, image, mask):
        """将图像与掩码取交集
        
        Args:
            image: 输入图像
            mask: 掩码图像（单通道）
        """
        # 确保掩码与图像尺寸一致
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
        # 应用掩码
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked
    
    def stitch_images(self, camera_images):
        """拼接多个相机的投影图像
        
        Args:
            camera_images: 字典，键为相机名称，值为处理后的图像
        """
        if not camera_images:
            raise ValueError("没有图像可拼接")
            
        # 获取第一个图像的尺寸作为输出尺寸
        first_image = next(iter(camera_images.values()))
        height, width = first_image.shape[:2]
        
        # 创建输出图像（黑色背景）
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 依次将每个相机的图像拼接到结果上
        for camera_name, image in camera_images.items():
            # 确保图像尺寸与输出尺寸一致
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
                
            # 将非黑色区域叠加到结果上
            # 创建图像的掩码（非黑色区域）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # 反转掩码，用于清除结果图像中对应区域
            inverse_mask = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(result, result, mask=inverse_mask)
            
            # 将当前图像叠加到结果上
            result = cv2.bitwise_or(result, image)
            
        return result

    def process_and_stitch(self, camera_names=['front', 'back'], save_result=False):
        """处理并拼接指定相机的图像
        
        Args:
            camera_names: 相机名称列表，默认为['front', 'back']
        """
        processed_images = {}
        
        for camera in camera_names:
            print(f"处理 {camera} 相机...")
            
            # 加载并投影图像
            image = self.load_image(camera)
            calib_image = self.calibrate_image(camera, image)
            projected = self.project_image(camera, calib_image)
            cv2.imshow(f"Projected {camera}", projected)
            cv2.waitKey()

            # 加载并应用掩码
            mask = self.load_mask(camera)
            masked_image = self.apply_mask(projected, mask)
            
            processed_images[camera] = masked_image
            
            # 保存中间结果
            cv2.imwrite(str(self.images_dir / f"projected_{camera}.jpg"), projected)
            cv2.imwrite(str(self.images_dir / f"masked_{camera}.jpg"), masked_image)
        
        # 拼接图像
        print("拼接图像...")
        stitched = self.stitch_images(processed_images)
        
        # 保存拼接结果
        if save_result:
            output_path = self.images_dir / "stitched_result.jpg"
            cv2.imwrite(str(output_path), stitched)
            print(f"拼接结果已保存到: {output_path}")
        
        return stitched

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='生成投影映射矩阵')
    parser.add_argument('--camera_name', type=str, required=False,
                        choices=['front', 'back'],
                        help='相机名称')
    parser.add_argument('--save_result', action='store_false',
                        help='不加载已存在的选点，强制重新选点')
    parser.add_argument('--avm_shapes', type=str, default='1500,1500',
                        help='环视图像尺寸，格式为width,height，例如500,1000')
    # 解析参数
    args = parser.parse_args()

    # 创建拼接器实例
    stitcher = AVMImageStitcher()
    
    # 处理并拼接前视和后视相机图像
    result = stitcher.process_and_stitch(['front', 'back'], args.save_result)

    # 显示结果
    cv2.imshow("Stitched Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()