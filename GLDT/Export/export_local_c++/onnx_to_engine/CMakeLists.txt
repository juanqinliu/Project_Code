cmake_minimum_required(VERSION 3.10)
project(trt_converter)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
add_compile_options(-Wno-deprecated-declarations)

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找TensorRT
if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
else()
    set(TENSORRT_ROOT /usr/lib/x86_64-linux-gnu)
endif()

include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# ONNX解析器
add_executable(onnx_to_engine src/onnx_to_engine.cpp src/int8_calibrator.cpp)
target_link_libraries(onnx_to_engine 
    nvinfer 
    nvonnxparser
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# 批处理验证工具 (暂时注释掉，因为源文件不存在)
# add_executable(verify_batch src/verify_batch.cpp)
# target_link_libraries(verify_batch 
#     nvinfer
#     ${CUDA_LIBRARIES}
# )

