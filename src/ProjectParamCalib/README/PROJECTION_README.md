# 投影映射参数标定工具

用于环视系统的投影映射矩阵生成工具，参考 [surround-view-system-introduction](https://github.com/neozhaoliang/surround-view-system-introduction) 项目实现。

## 功能特点

- ✅ 交互式选点界面
- ✅ 鱼眼相机畸变校正
- ✅ 透视变换投影计算
- ✅ 实时显示投影结果
- ✅ 保存投影映射矩阵
- ✅ 支持多个摄像头

## 文件说明

### 核心模块

1. **param_settings.py** - 参数设置和加载模块
   - 加载相机标定参数
   - 保存/加载投影映射矩阵
   - 管理鸟瞰图对应点

2. **fisheye_camera.py** - 鱼眼相机处理模块
   - 鱼眼图像畸变校正
   - 点坐标校正
   - 相机参数管理

3. **simple_gui.py** - 简单的GUI界面模块
   - 交互式选点
   - 实时显示结果
   - 鼠标操作支持

4. **run_get_projection_maps.py** - 主程序脚本
   - 完整的投影映射生成流程
   - 选点-投影-显示-保存一体化

## 使用流程

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 准备标定参数

首先需要完成相机标定，生成标定参数文件（YAML格式），保存在 `calibration_results` 目录：

```
calibration_results/
├── front_calibration.yaml
├── back_calibration.yaml
├── left_calibration.yaml
└── right_calibration.yaml
```

### 3. 准备测试图像

准备每个摄像头的测试图像，用于选点和预览投影结果。

### 4. 运行投影映射生成

```bash
# 生成前视摄像头投影映射
python3 run_get_projection_maps.py \
    --camera_name front \
    --image_path ./test_images/front.jpg \
    --calib_dir ./calibration_results \
    --config_dir ./config \
    --output_width 1000 \
    --output_height 1000 \
    --undistort

# 生成后视摄像头投影映射
python3 run_get_projection_maps.py \
    --camera_name back \
    --image_path ./test_images/back.jpg \
    --undistort

# 生成左视摄像头投影映射
python3 run_get_projection_maps.py \
    --camera_name left \
    --image_path ./test_images/left.jpg \
    --undistort

# 生成右视摄像头投影映射
python3 run_get_projection_maps.py \
    --camera_name right \
    --image_path ./test_images/right.jpg \
    --undistort
```

## 参数说明

### run_get_projection_maps.py 参数

- `--camera_name`: 相机名称（必需），可选值：front, back, left, right
- `--image_path`: 测试图像路径（必需）
- `--calib_dir`: 标定文件目录（默认：`./calibration_results`）
- `--config_dir`: 配置文件输出目录（默认：`./config`）
- `--output_width`: 输出图像宽度（默认：1000）
- `--output_height`: 输出图像高度（默认：1000）
- `--undistort`: 是否先进行畸变校正（可选）

## 使用步骤

### 步骤1: 选择源图像中的点

1. 运行脚本后，会打开一个窗口显示测试图像
2. **左键点击**：在图像中选择4个点
   - 建议选择车辆周围的参考点（如地面标记、停车线等）
   - 顺序：左上、右上、右下、左下
3. **右键点击**：删除最后一个点
4. **按 's' 键**：保存当前选点并继续
5. **按 'r' 键**：重新开始选点
6. **按 'q' 键**：退出

### 步骤2: 选择目标鸟瞰图中的点

1. 系统会提示是否自定义目标点
2. 如果选择 'y'，会打开鸟瞰图窗口
3. 在鸟瞰图中选择对应的4个点
4. 如果选择 'n'，将使用默认矩形区域

### 步骤3: 查看投影结果

1. 投影计算完成后，会显示投影结果
2. 可以对比原图和投影后的鸟瞰图
3. 按 'q' 键退出

### 步骤4: 保存结果

1. 系统会提示是否保存投影映射
2. 选择 'y' 保存映射矩阵到配置文件
3. 可选择保存投影结果图像

## 输出文件

运行完成后，会在 `config` 目录生成以下文件：

```
config/
├── front_projection_maps.yaml      # 前视投影映射
├── front_birdview_points.yaml      # 前视选点数据
├── front_projected.jpg             # 前视投影结果（可选）
├── back_projection_maps.yaml
├── back_birdview_points.yaml
├── left_projection_maps.yaml
├── left_birdview_points.yaml
├── right_projection_maps.yaml
└── right_birdview_points.yaml
```

### 投影映射文件格式

```yaml
camera_name: front
map_x:
  shape: [1000, 1000]
  data: [...]
  dtype: float32
map_y:
  shape: [1000, 1000]
  data: [...]
  dtype: float32
```

## 在ROS节点中使用

投影映射生成后，可以在ROS2节点中加载并使用：

```python
from param_settings import ParamSettings

# 加载投影映射
param_settings = ParamSettings(config_dir="./config")
map_x, map_y = param_settings.load_projection_maps("front")

# 应用投影
projected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
```

## 注意事项

1. **选点顺序**：确保源点和目标点的顺序一致
2. **点位置**：选择的点应该在图像中清晰可见
3. **投影区域**：目标点定义了鸟瞰图的有效区域
4. **畸变校正**：如果使用鱼眼相机，建议使用 `--undistort` 参数
5. **保存选点**：选点数据会自动保存，下次运行可以加载使用

## 故障排除

- **无法加载标定参数**：检查标定文件路径是否正确
- **投影结果异常**：检查选点是否正确，尝试重新选点
- **窗口无法显示**：确保系统支持GUI显示（X11或Wayland）

## 参考

- [surround-view-system-introduction](https://github.com/neozhaoliang/surround-view-system-introduction) - 原始参考项目

