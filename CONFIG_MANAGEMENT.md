# 共享配置文件管理方案

## 概述

本项目实现了统一的配置文件管理方案，使得 `ProjectParamCalib` 和 `surrend_view` 两个功能包可以共享相机标定参数等配置文件。

## 目录结构

```
AVM_Ship/                          # 工作空间根目录
├── config/                         # 共享配置目录（统一管理）
│   ├── calibration_results/        # 相机标定参数
│   │   ├── front_calibration.yaml
│   │   ├── back_calibration.yaml
│   │   ├── left_calibration.yaml
│   │   └── right_calibration.yaml
│   ├── projection_maps/            # 投影映射参数
│   │   ├── front_projection_maps.yaml
│   │   ├── front_birdview_points.yaml
│   │   └── ...
│   └── README.md
├── src/
│   ├── ProjectParamCalib/          # Python标定工具
│   │   ├── path_manager.py         # Python路径管理工具
│   │   └── ...
│   └── surrend_view/               # ROS2 C++节点
│       ├── include/surrend_view/
│       │   └── path_manager.hpp    # C++路径管理工具
│       └── src/
│           └── path_manager.cpp
└── ...
```

## 设计原则

### 1. 统一配置目录

**位置**：工作空间根目录下的 `config/` 目录

**优点**：
- ✅ 所有功能包都能访问
- ✅ 便于版本控制和部署
- ✅ 符合ROS2工作空间规范
- ✅ 易于备份和迁移

### 2. 路径优先级

1. **命令行参数**（最高优先级）
   - 支持通过参数覆盖默认路径
   - 适用于临时测试

2. **环境变量 `AVM_CONFIG_DIR`**
   - 可以通过环境变量指定自定义配置目录
   - 适用于多环境部署（开发/测试/生产）

3. **默认路径**（工作空间根目录/config）
   - 如果以上都不指定，使用默认路径
   - 最常用的情况

### 3. 路径管理工具

#### Python (`path_manager.py`)

```python
from path_manager import (
    get_config_base_dir,      # 获取配置基础目录
    get_calibration_dir,      # 获取标定文件目录
    get_calibration_file,     # 获取标定文件路径
    get_projection_dir,       # 获取投影映射目录
    get_projection_file       # 获取投影映射文件路径
)

# 自动使用统一配置目录
calib_file = get_calibration_file("front")
```

#### C++ (`path_manager.hpp`)

```cpp
#include "surrend_view/path_manager.hpp"

// 自动使用统一配置目录
std::string calib_file = surrend_view::getCalibrationFile("front");
```

## 使用示例

### Python脚本

```python
# 方式1：使用默认路径（推荐）
from param_settings import ParamSettings

param_settings = ParamSettings()  # 自动使用统一配置目录
param_settings.load_camera_calibration("front")

# 方式2：通过环境变量指定
import os
os.environ["AVM_CONFIG_DIR"] = "/custom/path"
param_settings = ParamSettings()

# 方式3：通过参数指定
param_settings = ParamSettings(config_dir="/custom/path")
```

### C++ ROS节点

```cpp
#include "surrend_view/path_manager.hpp"

// 自动使用统一配置目录
std::string calib_file = surrend_view::getCalibrationFile("front");

// 读取YAML文件
// ... 使用calib_file读取标定参数 ...
```

### 命令行使用

```bash
# Python脚本（使用默认路径）
python3 run_get_projection_maps.py \
    --camera_name front \
    --image_path ./test.jpg

# Python脚本（指定自定义路径）
python3 run_get_projection_maps.py \
    --camera_name front \
    --image_path ./test.jpg \
    --calib_dir /custom/path

# 使用环境变量
export AVM_CONFIG_DIR=/custom/config/path
python3 run_get_projection_maps.py --camera_name front --image_path ./test.jpg
```

## 配置文件位置

### 标定参数文件

**默认位置**：`config/calibration_results/{camera_name}_calibration.yaml`

**示例**：
- `config/calibration_results/front_calibration.yaml`
- `config/calibration_results/back_calibration.yaml`

### 投影映射文件

**默认位置**：`config/projection_maps/{camera_name}_projection_maps.yaml`

**示例**：
- `config/projection_maps/front_projection_maps.yaml`
- `config/projection_maps/front_birdview_points.yaml`

## 环境变量设置

### 临时设置

```bash
export AVM_CONFIG_DIR=/path/to/custom/config
```

### 永久设置

```bash
# 添加到 ~/.bashrc
echo 'export AVM_CONFIG_DIR=$HOME/avm_config' >> ~/.bashrc
source ~/.bashrc
```

## 优势

1. **统一管理**：所有配置文件集中在一个目录
2. **易于部署**：只需要复制 `config/` 目录
3. **灵活配置**：支持环境变量和参数覆盖
4. **跨语言**：Python和C++都支持
5. **版本控制**：可以单独管理配置目录
6. **自动创建**：目录不存在时自动创建

## 注意事项

1. **首次使用**：`config/` 目录会自动创建
2. **权限问题**：确保有读写权限
3. **路径解析**：路径管理工具会自动查找工作空间根目录
4. **相对路径**：优先使用相对路径，提高可移植性
5. **C++标准**：需要C++17支持（用于文件系统操作）

## 迁移指南

如果你之前将配置文件放在其他位置，可以：

1. **移动文件到统一目录**：
   ```bash
   mkdir -p config/calibration_results
   mv /old/path/*_calibration.yaml config/calibration_results/
   ```

2. **更新代码**：使用路径管理工具，无需硬编码路径

3. **测试**：确保所有功能包都能正确访问配置文件

## 参考文档

- `config/README.md` - 配置目录说明
- `src/ProjectParamCalib/README_PATH.md` - Python路径管理说明
- `src/surrend_view/include/surrend_view/path_manager.hpp` - C++ API文档

