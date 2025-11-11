import cv2
import numpy as np
import argparse

def generate_image(width, height, color, output_path):
    """
    生成指定分辨率和颜色的图片
    
    参数:
        width: 图片宽度（像素）
        height: 图片高度（像素）
        color: 背景颜色（BGR格式，如(255,0,0)表示蓝色）
        output_path: 输出图片路径（如"output.png"）
    """
    # 验证输入参数
    if width <= 0 or height <= 0:
        raise ValueError("宽度和高度必须为正整数")
    if not all(0 <= c <= 255 for c in color):
        raise ValueError("颜色值必须在0-255范围内（BGR格式）")
    
    # 创建指定尺寸和颜色的图片（OpenCV图像本质是numpy数组）
    # 格式为 (高度, 宽度, 3)，3表示BGR三通道
    image = np.full((height, width, 3), color, dtype=np.uint8)
    
    # 保存图片
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"无法保存图片到路径: {output_path}")
    
    print(f"成功生成图片: {output_path}")
    print(f"分辨率: {width}×{height}")
    print(f"背景颜色 (BGR): {color}")
    
    return image

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成指定分辨率的图片")
    parser.add_argument("--width", type=int, default=1080, help="图片宽度（默认1500像素）")
    parser.add_argument("--height", type=int, default=1920, help="图片高度（默认1080像素）")
    parser.add_argument("--color", type=int, nargs=3, default=[0, 0, 0], 
                       help="背景颜色（BGR格式，默认黑色(0 0 0)，例如红色为0 0 255）")
    parser.add_argument("--output", type=str, default="output.png", 
                       help="输出图片路径（默认output.png）")
    parser.add_argument("--show", default=True, help="是否显示生成的图片")

    args = parser.parse_args()
    
    try:
        # 生成图片
        img = generate_image(
            width=args.width,
            height=args.height,
            color=tuple(args.color),
            output_path=args.output
        )
        
        # 显示图片（如果需要）
        if args.show:
            cv2.imshow("Generated Image", img)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)  # 等待用户按键
            cv2.destroyAllWindows()  # 关闭窗口
            
    except Exception as e:
        print(f"错误: {e}")