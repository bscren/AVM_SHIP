# 路径管理说明

## 共享配置文件目录结构

```
AVM_Ship/                          # 工作空间根目录
├── config/                         # 共享配置目录（新建）
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
│   │   ├── path_manager.py         # 路径管理工具
│   │   ├── camera_calibration.py
│   │   └── ...
│   └── surrend_view/               # ROS2 C++节点
│       ├── include/surrend_view/
│       │   └── path_manager.hpp    # C++路径管理工具
│       └── src/
│           └── path_manager.cpp
└── ...
```

## 路径管理原则

### 1. 统一配置目录

所有共享配置文件放在工作空间根目录下的 `config/` 目录：
- **优点**：所有功能包都能访问
- **优点**：便于版本控制和部署
- **优点**：符合ROS2工作空间规范

### 2. 路径优先级

1. **命令行参数**（最高优先级）
   - 支持通过参数覆盖默认路径
   
2. **环境变量 `AVM_CONFIG_DIR`**
   - 可以通过环境变量指定自定义配置目录
   - 适用于多环境部署

3. **默认路径**（工作空间根目录/config）
   - 如果以上都不指定，使用默认路径

### 3. 路径管理工具

#### Python (`path_manager.py`)

```python
from path_manager import (
    get_config_base_dir,
    get_calibration_dir,
    get_calibration_file,
    get_projection_dir,
    get_projection_file
)

# 获取标定文件目录
calib_dir = get_calibration_dir()

# 获取标定文件路径
calib_file = get_calibration_file("front")

# 获取投影映射文件路径
proj_file = get_projection_file("front", "maps")
```

#### C++ (`path_manager.hpp`)

```cpp
#include "surrend_view/path_manager.hpp"

// 获取标定文件路径
std::string calib_file = surrend_view::getCalibrationFile("front");

// 获取配置目录
std::string config_dir = surrend_view::getConfigBaseDir();
```

## 使用示例

### Python脚本

```python
# 自动使用统一配置目录
from param_settings import ParamSettings

param_settings = ParamSettings()  # 自动使用默认路径
param_settings.load_camera_calibration("front")  # 从统一目录加载
```

### C++ ROS节点

```cpp
#include "surrend_view/path_manager.hpp"

// 获取标定文件路径
std::string calib_file = surrend_view::getCalibrationFile("front");

// 读取YAML文件
// ... 使用calib_file读取标定参数 ...
```

### 命令行参数覆盖

```bash
# Python脚本
python3 run_get_projection_maps.py \
    --camera_name front \
    --image_path ./test.jpg \
    --calib_dir /custom/path  # 覆盖默认路径

# 环境变量
export AVM_CONFIG_DIR=/custom/config/path
python3 run_get_projection_maps.py --camera_name front --image_path ./test.jpg
```

## 环境变量设置

```bash
# 临时设置
export AVM_CONFIG_DIR=/path/to/custom/config

# 永久设置（添加到 ~/.bashrc）
echo 'export AVM_CONFIG_DIR=$HOME/avm_config' >> ~/.bashrc
source ~/.bashrc
```

## 优势

1. **统一管理**：所有配置文件集中管理
2. **易于部署**：只需要复制 `config/` 目录
3. **灵活配置**：支持环境变量和参数覆盖
4. **跨平台**：Python和C++都支持
5. **版本控制**：可以单独管理配置目录

## 注意事项

1. **首次使用**：`config/` 目录会自动创建
2. **权限问题**：确保有读写权限
3. **路径解析**：路径管理工具会自动查找工作空间根目录
4. **相对路径**：优先使用相对路径，提高可移植性

