cmake_minimum_required(VERSION 3.8)
project(surrend_view)

# 设置C++标准（需要C++17支持文件系统）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 查找依赖（先查依赖，再编译代码）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# 2. 编译路径管理库
add_library(path_manager_lib SHARED
  src/path_manager.cpp
)
target_include_directories(path_manager_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 3. 编译库文件（multi_usb_camera_node 库）
add_library(multi_usb_camera_node SHARED
  src/multi_usb_camera_node.cpp
)
target_link_libraries(multi_usb_camera_node
  path_manager_lib
)
target_include_directories(multi_usb_camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# 链接依赖（合并写法，更规范）
ament_target_dependencies(multi_usb_camera_node 
  rclcpp 
  sensor_msgs 
  cv_bridge 
  image_transport 
  std_msgs 
  OpenCV
)
target_link_libraries(multi_usb_camera_node
  ${OpenCV_LIBS}
)

# 4. 编译可执行文件（multi_usb_camera_node_exec）
add_executable(multi_usb_camera_node_exec 
  src/main.cpp
)
add_executable(image_saver_node
  src/image_saver_node.cpp
)

# 头文件路径
target_include_directories(multi_usb_camera_node_exec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(image_saver_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 链接库和依赖
target_link_libraries(multi_usb_camera_node_exec
  multi_usb_camera_node  # 链接上面编译的库
  path_manager_lib       # 链接路径管理库
  ${OpenCV_LIBS}
)
ament_target_dependencies(multi_usb_camera_node_exec
  rclcpp 
  sensor_msgs 
  cv_bridge 
  image_transport 
  std_msgs 
  OpenCV
)

# image_saver_node 自身的依赖（不要把它链接进其他可执行）
target_link_libraries(image_saver_node
  ${OpenCV_LIBS}
)
ament_target_dependencies(image_saver_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  std_msgs
  OpenCV
)

# 5. 安装指令（必须在 add_library/add_executable 之后）
# 安装库文件
install(TARGETS path_manager_lib multi_usb_camera_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
# 安装可执行文件（关键：路径改为 lib/${PROJECT_NAME}）
install(TARGETS multi_usb_camera_node_exec
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}  # 符合 ROS 2 规范
)
install(TARGETS image_saver_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
# 安装头文件
install(DIRECTORY include/
  DESTINATION include/
)

# 6. 测试配置（可选，保留原逻辑）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 7. 包配置终结（必须在最后！）
ament_package()