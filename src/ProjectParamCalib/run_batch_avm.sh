#!/bin/bash
# 批量处理多个相机的投影映射，并拼接整体环视图
# 用法示例：
# bash run_batch_avm.sh --cameras front,back --load_existing_points false

set -e

# 默认参数
CAMERAS="right_front,right_back"
LOAD_EXISTING_POINTS="true"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cameras)
        CAMERAS="$2"
        shift; shift
        ;;
        --load_existing_points)
        LOAD_EXISTING_POINTS="$2"
        shift; shift
        ;;
        *)
        echo "未知参数: $1"; exit 1
        ;;
    esac
done

IFS=',' read -ra CAMERA_LIST <<< "$CAMERAS"

for CAMERA in "${CAMERA_LIST[@]}"; do
    echo "处理相机: $CAMERA"
    if [[ "$LOAD_EXISTING_POINTS" == "true" ]]; then
        python3 run_get_projection_maps.py --camera_name "$CAMERA" --load_existing_points || {
            echo "run_get_projection_maps.py 处理 $CAMERA 失败，终止批处理。"; exit 1;
        }
    else
        python3 run_get_projection_maps.py --camera_name "$CAMERA" || {
            echo "run_get_projection_maps.py 处理 $CAMERA 失败，终止批处理。"; exit 1;
        }
    fi
done

# 所有相机处理完后，拼接整体环视图
python3 run_get_full_avm.py || {
    echo "run_get_full_avm.py 失败。"; exit 1;
}
