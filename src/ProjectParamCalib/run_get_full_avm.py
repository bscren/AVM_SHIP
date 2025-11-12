import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from utils.param_settings import ParamSettings

class AVMImageStitcher:
    def __init__(self, prior_parameters_path, images_dir, yaml_dir):
        self.param_settings = ParamSettings(images_dir, yaml_dir)
        # 自动加载先验参数，获得 avm_size、proj_dst_points、avm_dst_points
        self.param_settings.load_prior_projection_parameters(prior_parameters_path)
        self.avm_width = self.param_settings.birdview_params['output_width']
        self.avm_height = self.param_settings.birdview_params['output_height']
        self.proj_dst_points = self.param_settings.proj_dst_points
        self.avm_dst_points = self.param_settings.avm_dst_points

    def get_projection_offset(self, camera_name):
        """计算每个相机投影图片在AVM图上的左上角坐标（根据目标点位置）"""
        # 取proj_dst_points和avm_dst_points的左上角点
        proj_topleft = np.array(self.proj_dst_points[camera_name][0])
        avm_topleft = np.array(self.avm_dst_points[camera_name][0])
        offset = avm_topleft - proj_topleft
        return tuple(offset.astype(int))
    
    def load_mask(self, camera_name):
        """加载相机的掩码图片"""
        from utils.path_manager import  get_images_path
        img_mask_path = get_images_path(camera_name, "mask")
        if not os.path.exists(img_mask_path):
            raise FileNotFoundError(f"未找到掩码图片: {img_mask_path}")
        img = cv2.imread(str(img_mask_path), cv2.IMREAD_GRAYSCALE)
        return img
    
    def get_blend_mask(self, camera_names):
        """计算每个相机的渐变掩码（预留接口，后续可根据像素距离生成权重图）"""
        self.blend_masks = {}
        for cam in camera_names:
            avm_mask = np.zeros((self.avm_height, self.avm_width, 3), dtype=np.uint8)
            mask = self.load_mask(cam)
            offset = self.get_projection_offset(cam)
            h, w = mask.shape[:2]
            # 粘贴到大图
            avm_mask[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = mask
            self.blend_masks[cam] = avm_mask
        


    def load_projected_image(self, camera_name):
        """加载已投影的图片（已矫正+投影）"""
        from utils.path_manager import  get_images_path
        img_projected = get_images_path(camera_name, "calib_projected")
        if not os.path.exists(img_projected):
            raise FileNotFoundError(f"未找到投影图片: {img_projected}")
        img = cv2.imread(str(img_projected))
        return img

    def stitch_avm(self, camera_names):
        """将所有投影图片拼接到AVM大图上"""
        avm_img = np.zeros((self.avm_height, self.avm_width, 3), dtype=np.uint8)
        blend_masks = {}
        for cam in camera_names:
            proj_img = self.load_projected_image(cam)
            offset = self.get_projection_offset(cam)
            h, w = proj_img.shape[:2]
            # 粘贴到大图
            avm_img[offset[1]:offset[1]+h, offset[0]:offset[0]+w] = proj_img
            # 预留渐变掩码接口
            blend_masks[cam] = self.get_blend_mask(cam, (h, w), offset)
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
    parser.add_argument('--save_result', action='store_true', default=False, help='保存拼接结果')
    parser.add_argument('--camera_names', type=str, default='right_front,right_back', help='参与拼接的相机列表')
    args = parser.parse_args()

    camera_list = args.camera_names.split(',')

    stitcher = AVMImageStitcher(args.prior_parameters_path, args.images_dir, args.yaml_dir)
    stitcher.get_blend_mask(camera_list)
    avm_img, blend_masks = stitcher.stitch_avm(camera_list)

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