import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import sys
from utils.param_settings import ParamSettings

class WeightPair:
    """存储每个相机重叠区域的权重矩阵weights，共有cam_num 个,每个里面有一张图片各自两边的重叠区域权重矩阵:
    """
    def __init__(self):
        self.G_left_top = None
        self.G_right_down = None


class AVMImageStitcher:
    def __init__(self, prior_parameters_path, images_dir, yaml_dir):
        self.param_settings = ParamSettings(images_dir, yaml_dir)
        # 自动加载先验参数，获得 avm_size、proj_dst_points、avm_dst_points
        self.param_settings.load_prior_projection_parameters(prior_parameters_path)
        self.avm_width = self.param_settings.birdview_params['output_width']
        self.avm_height = self.param_settings.birdview_params['output_height']
        self.proj_dst_points = self.param_settings.proj_dst_points
        self.avm_dst_points = self.param_settings.avm_dst_points
        self.overlap_pairs = []
        self.projected_image = {}
        # 存储每个相机重叠区域的权重矩阵weights，共有cam_num 个,每个里面有一张图片各自两边的重叠区域权重矩阵:
        # 用自定义的类:WeightPair 表示，便于不同时刻的读写
        self.weights = {}  #

        self.blend_masks = {} # 存储每个相机重叠区域的掩码，由0，1组成


    def initialize_WeightPairs(self, camera_names):
        """初始化每个相机的WeightPair对象"""
        for cam in camera_names:
            self.weights[cam] = WeightPair()

    def initialize_projected_image(self, camera_names):
        """初始化已投影图片字典"""
        for cam in camera_names:
            self.projected_image[cam]= self.load_image_or_mask(cam, type="calib_projected")

    def initialize_overlap_pairs(self, camera_names):
        """初始化重叠区域对
        returns: List of tuples, 每个元组包含两个相机名称，表示它们之间有重叠区域
        """
        # 这里假设相机按顺时针顺序排列
        num_cams = len(camera_names)
        self.overlap_pairs = []
        if num_cams < 2:
            raise ValueError("至少需要两个相机进行拼接")
        elif num_cams == 2:
            self.overlap_pairs.append((camera_names[0], camera_names[1]))
            return
        elif num_cams > 6:
            raise ValueError("最多支持6个相机进行拼接")
        
        for i in range(num_cams):
            cam1 = camera_names[i]
            cam2 = camera_names[(i + 1) % num_cams]  # 下一个相机，环绕回第一个
            if i == num_cams :
                break
            self.overlap_pairs.append((cam1, cam2))

    def get_projection_offset(self, camera_name):
        """计算每个相机投影图片在AVM图上的左上角坐标（根据目标点位置）"""
        # 取proj_dst_points和avm_dst_points的左上角点，形式是(x,y)，而当前代码中普遍的顺序是(y,x)
        proj_topleft = np.array(self.proj_dst_points[camera_name][0])
        avm_topleft = np.array(self.avm_dst_points[camera_name][0])
        offset = avm_topleft - proj_topleft
        # 返回 (y, x) 顺序，方便 NumPy 切片
        return int(offset[1]), int(offset[0])
    

    def load_image_or_mask(self, camera_name, type="calib_projected"):
        if type == "calib_projected":
            """type = "calib_projected" 加载已投影的图片（已矫正+投影）"""
            from utils.path_manager import  get_images_path
            img_projected = get_images_path(camera_name, "calib_projected")
            if not os.path.exists(img_projected):
                raise FileNotFoundError(f"未找到投影图片: {img_projected}")
            img = cv2.imread(str(img_projected))

        elif type == "mask":
            """type="mask",加载相机的掩码图片"""
            from utils.path_manager import  get_images_path
            img_mask_path = get_images_path(camera_name, type)
            if not os.path.exists(img_mask_path):
                raise FileNotFoundError(f"未找到掩码图片: {img_mask_path}")
            img = cv2.imread(str(img_mask_path), cv2.IMREAD_GRAYSCALE)
        return img
    
    def stitch_back_edge_region(self, cam_name, img, edged_img1, edged_img2):
        """将加权后的边缘区域放回原图"""
        # 根据相机名称获取裁剪逻辑: 先左后右，先上后下，具体先左后右还是先上后下，由键值在后续引用时决定
        (sacle_1 , scale_2) = self.param_settings.crop_edge_logics.get(cam_name, {})
        # img.shape: (height, width, channels)
        # if cam_name == "front":
        #     edged_img1 = img[:, 0:sacle_1]  # 左侧
        #     edged_img2 = img[:, scale_2: img.shape[1]]  # 右侧
        # if cam_name == "right_front":
        #     edged_img1 = img[0:sacle_1, :]  # 顶部
        #     edged_img2 = img[scale_2: img.shape[0], :]  # 底部
        # if cam_name == "right_back":
        #     edged_img1 = img[0:sacle_1, :]  # 顶部
        #     edged_img2 = img[scale_2: img.shape[0], :]  # 底部
        # if cam_name == "back":
        #     edged_img1 = img[:, 0:sacle_1]  # 左侧
        #     edged_img2 = img[:, scale_2: img.shape[1]]  # 右侧
        # if cam_name == "left_back":
        #     edged_img1 = img[0:sacle_1, :]  # 顶部
        #     edged_img2 = img[scale_2: img.shape[0], :]  # 底部
        # if cam_name == "left_front":
        #     edged_img1 = img[0:sacle_1, :]  # 左侧
        #     edged_img2 = img[scale_2: img.shape[0],:]  # 右侧
        if cam_name == "front":
            img[:, 0:sacle_1] = edged_img1  # 左侧
            img[:, scale_2: img.shape[1]] = edged_img2  # 右侧
        if cam_name == "right_front":
            img[0:sacle_1, :] = edged_img1  # 顶部
            img[scale_2: img.shape[0], :] = edged_img2  # 底部
        if cam_name == "right_back":
            img[0:sacle_1, :] = edged_img1  # 顶部
            img[scale_2: img.shape[0], :] = edged_img2  # 底部
        if cam_name == "back":
            img[:, 0:sacle_1] = edged_img1  # 左侧
            img[:, scale_2: img.shape[1]] = edged_img2  # 右侧
        if cam_name == "left_back":
            img[0:sacle_1, :] = edged_img1  # 顶部
            img[scale_2: img.shape[0], :] = edged_img2  # 底部
        if cam_name == "left_front":
            img[0:sacle_1, :] = edged_img1  # 左侧
            img[scale_2: img.shape[0],:] = edged_img2  # 右侧
        return img
    

    def crop_edge_region(self, cam_name, img):
        """裁剪单张图片的边缘区域，用于后续与重叠区域掩码相乘"""
        # 根据相机名称获取裁剪逻辑: 先左后右，先上后下，具体先左后右还是先上后下，由键值在后续引用时决定
        (sacle_1 , scale_2) = self.param_settings.crop_edge_logics.get(cam_name, {})
        # img.shape: (height, width, channels)
        if cam_name == "front":
            edged_img1 = img[:, 0:sacle_1]  # 左侧
            edged_img2 = img[:, scale_2: img.shape[1]]  # 右侧
        if cam_name == "right_front":
            edged_img1 = img[0:sacle_1, :]  # 顶部
            edged_img2 = img[scale_2: img.shape[0], :]  # 底部
        if cam_name == "right_back":
            edged_img1 = img[0:sacle_1, :]  # 顶部
            edged_img2 = img[scale_2: img.shape[0], :]  # 底部
        if cam_name == "back":
            edged_img1 = img[:, 0:sacle_1]  # 左侧
            edged_img2 = img[:, scale_2: img.shape[1]]  # 右侧
        if cam_name == "left_back":
            edged_img1 = img[0:sacle_1, :]  # 顶部
            edged_img2 = img[scale_2: img.shape[0], :]  # 底部
        if cam_name == "left_front":
            edged_img1 = img[0:sacle_1, :]  # 左侧
            edged_img2 = img[scale_2: img.shape[0],:]  # 右侧
        return edged_img1, edged_img2
    

    def crop_overlap_region(self, cam1_name, cam2_name, img1, img2):
        """裁剪两张图片的重叠区域，并将其返回"""
        # 重叠区域是预定义的，根据输入图片的名称获得预先制定的，对应的分割策略，返回的图像有顺序，前后为顺时针方向
        (cam1_scale , cam2_scale) = self.param_settings.crop_overlap_logics.get((cam1_name, cam2_name), {})

        if (cam1_name, cam2_name) == ("front", "right_front"):
            # 例如，裁剪 img1 的右侧 cam1_scale 像素，裁剪 img2 的顶部 cam2_scale 像素
            cropfrom_img1 = img1[:, img1.shape[1] - cam1_scale: img1.shape[1]]
            cropfrom_img2 = img2[0:cam2_scale, :]
        elif (cam1_name, cam2_name) == ("right_front", "right_back"):
            cropfrom_img1 = img1[cam1_scale: img1.shape[0], :]
            cropfrom_img2 = img2[0:cam2_scale, :]
        elif (cam1_name, cam2_name) == ("right_back", "back"):
            cropfrom_img1 = img1[img1.shape[0] - cam1_scale: img1.shape[0], :]
            cropfrom_img2 = img2[:, 0:cam2_scale]
        elif (cam1_name, cam2_name) == ("back", "left_back"):
            cropfrom_img1 = img1[:, 0:cam1_scale]
            cropfrom_img2 = img2[img2.shape[0] - cam2_scale: img2.shape[0], :]
        elif (cam1_name, cam2_name) == ("left_back", "left_front"):
            cropfrom_img1 = img1[0:cam1_scale, :]
            cropfrom_img2 = img2[:, 0:cam2_scale]
        elif (cam1_name, cam2_name) == ("left_front", "front"):
            cropfrom_img1 = img1[:, img1.shape[1] - cam1_scale: img1.shape[1]]
            cropfrom_img2 = img2[0:cam2_scale, :]
        else:
            print(f"未定义的图像对: ({cam1_name}, {cam2_name}) 的裁剪逻辑")
            # 终止整个程序运行
            sys.exit(1)

        # =========================== DEBUG 显示裁剪结果用于调试 =========================
        # cv2.imshow(f" {img1}", img1)
        # cv2.imshow(f" {img2}", img2)
        # cv2.imshow(f"Crop from {cam1_name}", cropfrom_img1)
        # cv2.imshow(f"Crop from {cam2_name}", cropfrom_img2)
        # waitKey = cv2.waitKey(0)
        # ==========================================
        return cropfrom_img1, cropfrom_img2
        
        
        # cropfrom_img1 = img1[:, w1 - overlap_width1:w1]
        # cropfrom_img2 = img2[:, 0:overlap_width2]

        # return cropfrom_img1, cropfrom_img`2
    
    def get_overlap_region_mask(self, cropfrom_img1, cropfrom_img2):
        """计算两张裁剪图片的重叠区域掩码"""
        # 假设重叠区域掩码是两张裁剪图片的按位与操作
        overlapMask = cv2.bitwise_and(cropfrom_img1, cropfrom_img2)
        #     Convert an image to a mask array.
        # 如果已经是单通道，不需要再转灰度
        if overlapMask.ndim == 3 and overlapMask.shape[2] == 3:
            gray = cv2.cvtColor(overlapMask, cv2.COLOR_BGR2GRAY)
        else:
            gray = overlapMask
        _, overlapMask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        overlapMask = cv2.dilate(overlapMask, np.ones((5, 5), np.uint8), iterations=5)

        return overlapMask
    
    def compute_blend_weight(self, overlapMask, cam1_name, cam2_name, dist_threshold=1e-6):
        """
        对 overlapMask 的每个像素计算权重矩阵 G
        """
        h, w = overlapMask.shape
        G = np.zeros((h, w), dtype=np.float32)

        # 获取所有像素坐标
        indices = np.argwhere(overlapMask == 255)

        # 根据相机组合决定边界方向
        if (cam1_name, cam2_name) in [("front", "right_front")]:
            # 上右融合：A为左边界，B为下边界
            edgeA = 0  # x=0
            edgeB = w - 1  # x=w-1
            for y, x in indices:
                distToA = abs(x - edgeA)
                distToB = abs(y - edgeB)
                distToA = distToA ** 2
                distToB = distToB ** 2
                if distToA + distToB > dist_threshold:
                    G[y, x] = distToB / (distToA + distToB)
        elif (cam1_name, cam2_name) in [("right_front", "right_back"), ("left_front", "left_back")]:
            # 上下融合：A为顶端，B为底端
            edgeA = 0  # y=0
            edgeB = h - 1  # y=h-1
            for y, x in indices:
                distToA = abs(y - edgeA)
                distToB = abs(y - edgeB)
                distToA = distToA ** 2
                distToB = distToB ** 2
                if distToA + distToB > dist_threshold:
                    G[y, x] = distToB / (distToA + distToB)
        elif (cam1_name, cam2_name) in [("right_back", "back")]:
            # 右下融合：A为上边界，B为左边界
            edgeA = 0  # y=0
            edgeB = 0  # x=0
            for y, x in indices:
                distToA = abs(y - edgeA)
                distToB = abs(x - edgeB)
                distToA = distToA ** 2
                distToB = distToB ** 2
                if distToA + distToB > dist_threshold:
                    G[y, x] = distToB / (distToA + distToB)
        elif (cam1_name, cam2_name) in [("back", "left_back")]:
            # 左下融合：A为右边界，B为上边界
            edgeA = w - 1  # x=w-1
            edgeB = 0      # y=0
            for y, x in indices:
                distToA = abs(x - edgeA)
                distToB = abs(y - edgeB)
                distToA = distToA ** 2
                distToB = distToB ** 2
                if distToA + distToB > dist_threshold:
                    G[y, x] = distToB / (distToA + distToB)
        elif (cam1_name, cam2_name) in [("left_front", "front")]:
            # 左上融合：A为下边界，B为右边界
            edgeA = h - 1  # y=h-1
            edgeB = w - 1  # x=w-1
            for y, x in indices:
                distToA = abs(y - edgeA)
                distToB = abs(x - edgeB)
                distToA = distToA ** 2
                distToB = distToB ** 2
                if distToA + distToB > dist_threshold:
                    G[y, x] = distToB / (distToA + distToB)
        else:
            # 默认左右融合
            edgeA = 0
            edgeB = w - 1
            for y, x in indices:
                distToA = abs(x - edgeA)
                distToB = abs(x - edgeB)
                distToA = distToA ** 2
                distToB = distToB ** 2
                if distToA + distToB > dist_threshold:
                    G[y, x] = distToB / (distToA + distToB)

        return G

    def get_weight_mask_matrix(self, cam1_name, cam2_name):
        """计算两张图片的权重掩码矩阵"""
        img1 = self.projected_image[cam1_name]
        img2 = self.projected_image[cam2_name]
        
        # 获得两张图片之间的重叠区域剪切逻辑
        cropfrom_img1, cropfrom_img2 = self.crop_overlap_region(cam1_name, cam2_name, img1, img2)
        cv2.imshow(f"Crop from {cam1_name}", cropfrom_img1)
        cv2.imshow(f"Crop from {cam2_name}", cropfrom_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        overlapMask = self.get_overlap_region_mask(cropfrom_img1, cropfrom_img2)
        G = self.compute_blend_weight(overlapMask, cam1_name, cam2_name)
        return G, overlapMask


    def get_weights_and_blendmasks(self, camera_names):
        """计算每个相机的渐变掩码（预留接口，后续可根据像素距离生成权重图）
            掩码序号由左上角从零开始，顺时针
        """
        for cam1_name, cam2_name in self.overlap_pairs:
            # 计算重叠区域的掩码 G_right_down
            G1, M = self.get_weight_mask_matrix(cam1_name, cam2_name)
            self.weights[cam1_name].G_right_down = G1 # 权重矩阵 in cam1's overlap region
            self.weights[cam2_name].G_left_top = 1 - G1 # 权重矩阵 in cam2's overlap region
            self.blend_masks[(cam1_name, cam2_name)] = M
            # 可视化M（掩码矩阵）
            # M0是二值图像（0或255），直接作为灰度图显示
            cv2.imshow("M (Overlap Mask)", M)
            # 可视化G (权重矩阵)    
            # G是0~1的浮点数，需转换为0~255的uint8类型
            G_vis = (G1 * 255).astype(np.uint8)  # 映射到0-255范围
            cv2.imshow("G1 (Weight Matrix)", G_vis)
            # 等待按键关闭窗口
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    def make_luminance_balance(self):
        """对所有投影图片进行亮度均衡处理（预留接口，后续可根据重叠区域均值进行调整）"""
        # 这里简单实现为不做任何处理
        print("亮度均衡处理（预留接口，当前未实现）")
        return
    
    def make_white_balance(self):
        """对所有投影图片进行白平衡处理（预留接口，后续可根据重叠区域均值进行调整）"""
        # 这里简单实现为不做任何处理
        print("白平衡处理（预留接口，当前未实现）")
        return
    
    def copy_ship_image_to_avm(self):
        """将船只图像复制到AVM大图上（预留接口，后续可根据实际需求实现）"""
        # 这里简单实现为不做任何处理
        print("将船只图像复制到AVM大图上（预留接口，当前未实现）")
        return


    def blend_overlap_region(self, cam1_name, cam2_name):
        """
        对两张图片的重叠区域进行融合，并返回融合后的区域
        """
        imgA = self.projected_image[cam1_name]
        imgB = self.projected_image[cam2_name]
        # 裁剪重叠区域
        cropA, cropB = self.crop_overlap_region(cam1_name, cam2_name, imgA, imgB)
        # 获得重叠掩码
        overlapMask = self.get_overlap_region_mask(cropA, cropB)
        # 获得权重矩阵
        G1 = self.weights[cam1_name].G_right_down[..., np.newaxis]
        G2 = self.weights[cam2_name].G_left_top[..., np.newaxis]
        # 融合公式
        blend = (cropA.astype(np.float32) * G1 + cropB.astype(np.float32) * G2).astype(np.uint8)
        cv2.imshow("Blended Overlap Region", blend)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return blend, overlapMask
    
    def stitch_all_parts(self):
        """将所有投影图片拼接到AVM大图上"""
        avm_img = np.zeros((self.avm_height, self.avm_width, 3), dtype=np.uint8)
        blend_masks = {}
        # 先将所有非重叠区域直接粘贴到AVM图上
        for cam_name, img in self.projected_image.items():
            offset_y, offset_x = self.get_projection_offset(cam_name)
            h, w = img.shape[0:2]
            avm_img[offset_y:offset_y + h, offset_x:offset_x + w] = img
        cv2.imwrite("avm_initial.jpg", avm_img)

        # 然后再用融合后的重叠区域（blend）覆盖原图的重叠部分
        # 所有相机的重叠区域，统一拆分、加权、粘贴，不能单独处理某个相机，否则会导致像素权重无法叠加

        for cam1_name, cam2_name in self.overlap_pairs:
            blend, overlapMask = self.blend_overlap_region(cam1_name, cam2_name)
            # 计算重叠区域在AVM图上的位置
            cam1_scale, cam2_scale = self.param_settings.crop_overlap_logics.get((cam1_name, cam2_name), {})
            offset1_y, offset1_x = self.get_projection_offset(cam1_name)

            if (cam1_name, cam2_name) == ("front", "right_front"):
                avm_img[offset1_y:offset1_y + blend.shape[0], offset1_x + cam1_scale:offset1_x + cam1_scale + blend.shape[1]] = blend
            elif (cam1_name, cam2_name) == ("right_front", "right_back"):
                avm_img[offset1_y + cam1_scale:offset1_y + cam1_scale + blend.shape[0], offset1_x:offset1_x + blend.shape[1]] = blend
            elif (cam1_name, cam2_name) == ("right_back", "back"):
                avm_img[offset1_y + cam1_scale:offset1_y + cam1_scale + blend.shape[0], offset1_x:offset1_x + blend.shape[1]] = blend
            elif (cam1_name, cam2_name) == ("back", "left_back"):   
                avm_img[offset1_y:offset1_y + blend.shape[0], offset1_x:offset1_x + blend.shape[1]] = blend
            elif (cam1_name, cam2_name) == ("left_back", "left_front"):
                avm_img[offset1_y:offset1_y + blend.shape[0], offset1_x:offset1_x + blend.shape[1]] = blend
            elif (cam1_name, cam2_name) == ("left_front", "front"):
                avm_img[offset1_y:offset1_y + blend.shape[0], offset1_x:offset1_x + blend.shape[1]] = blend

        # 预留渐变掩码接口
        # blend_masks[cam] = self.get_weights_and_blendmasks(cam, (h, w), offset)
        return avm_img, blend_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AVM拼接')
    parser.add_argument('--prior_parameters_path',type = str,
                        default = str(Path(__file__).resolve().parents[2] / "config" / "yaml" / "prior_parameters.yaml"),
                        help='先验参数文件路径')
    parser.add_argument('--yaml_dir',type = str,
                        default = str(Path(__file__).resolve().parents[2] / "config" / "yaml" ),
                        help='yaml目录')
    parser.add_argument('--images_dir', type=str,
                        default=str(Path(__file__).resolve().parents[2] / "config" / "images"),
                        help='图像目录')
    parser.add_argument('--save_result', default=True, help='保存拼接结果')
    parser.add_argument('--camera_names', type=str, default='right_front,right_back', 
                        help='参与拼接的相机列表.最多支持6个相机(顺时针排布)：front,right_front,right_back,back,left_back,left_front')
    args = parser.parse_args()

    camera_list = args.camera_names.split(',')

    stitcher = AVMImageStitcher(args.prior_parameters_path, args.images_dir, args.yaml_dir)
    stitcher.initialize_WeightPairs(camera_list)
    stitcher.initialize_projected_image(camera_list)
    stitcher.initialize_overlap_pairs(camera_list)
    stitcher.get_weights_and_blendmasks(camera_list)
    stitcher.make_luminance_balance()
    avm_img, blend_masks = stitcher.stitch_all_parts()
    stitcher.make_white_balance()
    stitcher.copy_ship_image_to_avm()

    # # 等比例缩小至像素宽为500
    # scale_factor = 500 / avm_img.shape[1]
    # avm_img = cv2.resize(avm_img, (500, int(avm_img.shape[0] * scale_factor)))

    cv2.imshow("AVM Stitched Result", avm_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save_result:
        out_path = os.path.join(args.images_dir, "avm_stitch.jpg")
        cv2.imwrite(str(out_path), avm_img)
        print(f"拼接结果已保存到: {out_path}")