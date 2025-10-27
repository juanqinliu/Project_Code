cmake_minimum_required(VERSION 3.10)

# 设置CUDA路径（确保使用CUDA 11.6）
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)

# 设置编译器为GCC-8以兼容CUDA 11.6
set(CMAKE_C_COMPILER /usr/bin/gcc-8)
set(CMAKE_CXX_COMPILER /usr/bin/g++-8)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-8)

project(export_engine CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 检测系统架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(SYSTEM_ARCH "aarch64")
    message(STATUS "检测到ARM64架构")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(SYSTEM_ARCH "x86_64")
    message(STATUS "检测到x86_64架构")
else()
    message(WARNING "未知架构: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 查找CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

# 查找OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

# 查找TensorRT
if(NOT DEFINED TENSORRT_ROOT)
    # 尝试在常见位置查找TensorRT
    if(SYSTEM_ARCH STREQUAL "aarch64")
        # ARM64架构的TensorRT路径
        foreach(path "/usr" "/usr/local/TensorRT" "/usr/local/TensorRT-8.6.0" "/usr/local/TensorRT-8.4.2.4" "/opt/TensorRT" "/opt/TensorRT-8.6.0" "/opt/TensorRT-8.4.2.4")
            if(EXISTS "${path}/include/NvInfer.h" OR EXISTS "${path}/include/aarch64-linux-gnu/NvInfer.h")
                set(TENSORRT_ROOT ${path})
                break()
            endif()
        endforeach()
    else()
        # x86_64架构的TensorRT路径
        foreach(path "/usr" "/usr/local/TensorRT" "/usr/local/TensorRT-8.6.0" "/usr/local/TensorRT-8.4.2.4" "/opt/TensorRT" "/opt/TensorRT-8.6.0" "/opt/TensorRT-8.4.2.4")
            if(EXISTS "${path}/include/NvInfer.h" OR EXISTS "${path}/include/x86_64-linux-gnu/NvInfer.h")
                set(TENSORRT_ROOT ${path})
                break()
            endif()
        endforeach()
    endif()
    
    if(NOT DEFINED TENSORRT_ROOT)
        message(FATAL_ERROR "未找到TensorRT，请设置TENSORRT_ROOT")
    endif()
endif()

# 设置TensorRT包含路径
if(EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
elseif(SYSTEM_ARCH STREQUAL "aarch64" AND EXISTS "${TENSORRT_ROOT}/include/aarch64-linux-gnu/NvInfer.h")
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include/aarch64-linux-gnu)
elseif(SYSTEM_ARCH STREQUAL "x86_64" AND EXISTS "${TENSORRT_ROOT}/include/x86_64-linux-gnu/NvInfer.h")
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include/x86_64-linux-gnu)
endif()

message(STATUS "Using TensorRT from: ${TENSORRT_ROOT}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")

# 设置TensorRT库路径
if(EXISTS "${TENSORRT_ROOT}/lib/libnvinfer.so")
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
elseif(EXISTS "${TENSORRT_ROOT}/lib64/libnvinfer.so")
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib64)
elseif(SYSTEM_ARCH STREQUAL "aarch64" AND EXISTS "${TENSORRT_ROOT}/lib/aarch64-linux-gnu/libnvinfer.so")
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib/aarch64-linux-gnu)
elseif(SYSTEM_ARCH STREQUAL "x86_64" AND EXISTS "${TENSORRT_ROOT}/lib/x86_64-linux-gnu/libnvinfer.so")
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib/x86_64-linux-gnu)
endif()

message(STATUS "TensorRT libraries: ${TENSORRT_LIB_DIR}")

# 包含目录
include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# 添加链接目录
link_directories(
    ${TENSORRT_LIB_DIR}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 根据架构设置CUDA架构
if(SYSTEM_ARCH STREQUAL "aarch64")
    # ARM64架构的CUDA架构（Jetson系列）
    set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87")
    message(STATUS "设置ARM64 CUDA架构: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # x86_64架构的CUDA架构
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
    message(STATUS "设置x86_64 CUDA架构: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# 注意：logger是header-only实现，不需要单独编译

# 编译INT8校准和预处理库
add_library(int8_utils STATIC
    src/int8_calibrator.cpp
    src/preprocess.cu
)

# 创建可执行文件
add_executable(export_engine src/main.cpp)
target_link_libraries(export_engine 
    int8_utils
    nvinfer
    nvonnxparser
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# 为了支持C++14
target_compile_features(export_engine PRIVATE cxx_std_14)

# 设置RPATH以便在运行时能找到TensorRT库
set_target_properties(export_engine PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${TENSORRT_LIB_DIR}"
)

# 添加内存优化编译选项
target_compile_options(export_engine PRIVATE 
    -O2                    # 优化级别2
    -DNDEBUG              # 禁用调试
    -DTENSORRT_MAX_WORKSPACE_SIZE=2147483648  # 2GB工作空间
    -DTENSORRT_MAX_GPU_MEMORY=4294967296      # 4GB GPU内存限制
)

# 显示编译信息
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "System Architecture: ${SYSTEM_ARCH}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}") 

# 打印配置信息
message(STATUS "TensorRT Motion Detector Converter 配置完成")
message(STATUS "这个工具可以将带有双帧输入的ONNX模型转换为TensorRT引擎")
message(STATUS "支持FP32和FP16精度") 