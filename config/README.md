# 共享配置文件目录

此目录用于存放所有功能包共享的配置文件，包括相机标定参数、投影映射参数等。

## 目录结构

```
config/
├── calibration_results/          # 相机标定参数（YAML格式）
│   ├── front_calibration.yaml
│   ├── back_calibration.yaml
│   ├── left_calibration.yaml
│   └── right_calibration.yaml
├── projection_maps/              # 投影映射参数（YAML格式）
│   ├── front_projection_maps.yaml
│   ├── front_birdview_points.yaml
│   ├── back_projection_maps.yaml
│   └── ...
└── README.md                     # 本文件
```

## 使用原则

1. **统一路径**：所有功能包使用统一的路径管理工具访问配置
2. **环境变量支持**：支持通过环境变量 `AVM_CONFIG_DIR` 覆盖默认路径
3. **相对路径优先**：优先使用相对路径，提高可移植性
4. **参数覆盖**：支持通过命令行参数覆盖默认路径

## 路径优先级

1. 命令行参数（最高优先级）
2. 环境变量 `AVM_CONFIG_DIR`
3. 工作空间根目录下的 `config` 目录（默认）

## 访问方式

### Python脚本

```python
from path_manager import get_config_path

# 获取标定文件路径
calib_dir = get_config_path("calibration_results")
calib_file = os.path.join(calib_dir, "front_calibration.yaml")
```

### C++ ROS节点

```cpp
#include "surrend_view/path_manager.hpp"

// 获取标定文件路径
std::string calib_dir = getConfigPath("calibration_results");
std::string calib_file = calib_dir + "/front_calibration.yaml";
```

## 环境变量设置

```bash
# 设置自定义配置目录
export AVM_CONFIG_DIR=/path/to/custom/config

# 或者在 ~/.bashrc 中永久设置
echo 'export AVM_CONFIG_DIR=$HOME/avm_config' >> ~/.bashrc
```

