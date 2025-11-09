import cv2
import time

def get_available_cameras(max_try=10):
    """检测所有可用的USB摄像头索引"""
    available_indices = []
    for index in range(max_try):
        # 尝试打开摄像头（根据系统自动适配后端）
        try:
            # Linux常用V4L2后端，Windows/macOS可自动适配
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        except:
            cap = cv2.VideoCapture(index)
            
        # 检查是否成功打开
        if cap.isOpened():
            # 尝试读取一帧验证是否有效
            ret, _ = cap.read()
            if ret:
                available_indices.append(index)
            cap.release()  # 临时释放，后续统一打开
        # 避免频繁尝试导致系统占用
        time.sleep(0.1)
    return available_indices

def show_all_cameras():
    """显示所有检测到的USB摄像头画面"""
    # 检测可用摄像头
    available_indices = get_available_cameras()
    if not available_indices:
        print("未检测到任何可用的USB摄像头，请检查连接")
        return
    
    print(f"检测到 {len(available_indices)} 个可用摄像头，索引：{available_indices}")
    print("按 'q' 键关闭所有摄像头窗口")
    
    # 打开所有摄像头并存储（键：索引，值：VideoCapture对象）
    cameras = {}
    for index in available_indices:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        except:
            cap = cv2.VideoCapture(index)
        # 设置合适的分辨率（可选，根据摄像头支持情况调整）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cameras[index] = cap
    
    try:
        while True:
            # 逐个读取并显示每个摄像头的画面
            for index, cap in cameras.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"摄像头 {index} 已断开连接，停止显示")
                    cap.release()
                    cv2.destroyWindow(f'Camera {index}')
                    del cameras[index]
                    continue
                
                # 显示画面（窗口标题包含索引，方便区分）
                cv2.imshow(f'Camera {index}', frame)
            
            # 若所有摄像头都断开，则退出
            if not cameras:
                print("所有摄像头已断开连接")
                break
            
            # 检测按键，按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 释放所有摄像头资源
        for cap in cameras.values():
            cap.release()
        # 关闭所有显示窗口
        cv2.destroyAllWindows()
        print("已关闭所有摄像头")

if __name__ == "__main__":
    show_all_cameras()
