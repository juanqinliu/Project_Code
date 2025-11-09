cmake_minimum_required(VERSION 3.10)

# Automatically detect CUDA path, prioritize using the actual version installed in the system
if(EXISTS "/usr/local/cuda-11.8/bin/nvcc")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
    message(STATUS "Using CUDA 11.8: ${CMAKE_CUDA_COMPILER}")
elseif(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    message(STATUS "Using system default CUDA: ${CMAKE_CUDA_COMPILER}")
else()
    message(FATAL_ERROR "CUDA compiler not found, please check CUDA installation")
endif()

# Set CUDA host compiler to GCC-8 in the project declaration to resolve compatibility issues
# Change hardcoded to existence detection: if gcc-8/g++-8 exists, use it, otherwise fall back to system default
if(EXISTS "/usr/bin/gcc-8" AND EXISTS "/usr/bin/g++-8")
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-8)
    set(CMAKE_C_COMPILER /usr/bin/gcc-8)
    set(CMAKE_CXX_COMPILER /usr/bin/g++-8)
    message(STATUS "Detected gcc-8/g++-8, use it as host compiler")
else()
    message(WARNING "gcc-8/g++-8 not found, using system default compiler")
endif()

project(gldt CUDA CXX)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Detect platform type
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLATFORM_TYPE "JETSON_PLATFORM")
    message(STATUS "Detected Jetson platform")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(PLATFORM_TYPE "X86_PLATFORM")
    message(STATUS "Detected x86 platform")
else()
    set(PLATFORM_TYPE "UNKNOWN_PLATFORM")
    message(STATUS "Unknown platform: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Add platform-related compile definitions
if(PLATFORM_TYPE STREQUAL "JETSON_PLATFORM")
    add_definitions(-DJETSON_PLATFORM)
    message(STATUS "Enable Jetson platform support")
elseif(PLATFORM_TYPE STREQUAL "X86_PLATFORM")
    add_definitions(-DX86_PLATFORM)
    message(STATUS "Enable x86 platform support")
endif()

# find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)   
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Dependencies message package
find_package(gldt_msgs REQUIRED)

# CUDA settings
find_package(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-Xcompiler -fPIC)

# Set CUDA architecture
if(PLATFORM_TYPE STREQUAL "JETSON_PLATFORM")
    set(CMAKE_CUDA_ARCHITECTURES 72)  # Jetson Xavier NX
    message(STATUS "Set CUDA architecture to 72 (Jetson)")
elseif(PLATFORM_TYPE STREQUAL "X86_PLATFORM")
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # RTX 20xx, 30xx, 40xx系列
    message(STATUS "Set CUDA architecture to 75,80,86 (x86)")
else()
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Default architecture
    message(STATUS "Set default CUDA architecture to 75")
endif()

# Find other dependencies
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Find FFmpeg libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)
pkg_check_modules(AVDEVICE REQUIRED libavdevice)

message(STATUS "Found FFmpeg libraries:")
message(STATUS "  libavcodec: ${AVCODEC_VERSION}")
message(STATUS "  libavformat: ${AVFORMAT_VERSION}")
message(STATUS "  libavutil: ${AVUTIL_VERSION}")
message(STATUS "  libswscale: ${SWSCALE_VERSION}")
message(STATUS "  libavdevice: ${AVDEVICE_VERSION}")

# Check GStreamer support
find_package(PkgConfig)
pkg_check_modules(GSTREAMER gstreamer-1.0)
pkg_check_modules(GSTREAMER_RTSP gstreamer-rtsp-server-1.0)
if(GSTREAMER_FOUND AND GSTREAMER_RTSP_FOUND)
    message(STATUS "GStreamer found: ${GSTREAMER_VERSION}")
    message(STATUS "GStreamer RTSP server found: ${GSTREAMER_RTSP_VERSION}")
    add_definitions(-DWITH_GSTREAMER)
    include_directories(${GSTREAMER_INCLUDE_DIRS})
    include_directories(${GSTREAMER_RTSP_INCLUDE_DIRS})
else()
    message(STATUS "GStreamer or RTSP server not found, GStreamer video reader will not be available")
    message(STATUS "GStreamer found: ${GSTREAMER_FOUND}")
    message(STATUS "GStreamer RTSP found: ${GSTREAMER_RTSP_FOUND}")
endif()

# Find gflags library
find_package(gflags REQUIRED)
if(gflags_FOUND)
    message(STATUS "Found gflags: ${gflags_VERSION}")
else()
    message(FATAL_ERROR "gflags library not found. Please install it using: sudo apt-get install libgflags-dev")
endif()

# Find fmt library
pkg_check_modules(FMT fmt)
if(FMT_FOUND)
    message(STATUS "Found fmt: ${FMT_VERSION}")
    message(STATUS "FMT_INCLUDE_DIRS: ${FMT_INCLUDE_DIRS}")
    message(STATUS "FMT_LIBRARY_DIRS: ${FMT_LIBRARY_DIRS}")
    message(STATUS "FMT_LIBRARIES: ${FMT_LIBRARIES}")
    add_definitions(-DWITH_FMT)
    include_directories(${FMT_INCLUDE_DIRS})
    link_directories(${FMT_LIBRARY_DIRS})
else()
    message(STATUS "fmt not found, trying to find with find_package")
    find_package(fmt QUIET)
    if(fmt_FOUND)
        message(STATUS "Found fmt with find_package")
        add_definitions(-DWITH_FMT)
    else()
        message(WARNING "fmt library not found, some camera features may not work")
    endif()
endif()

# Find TensorRT libraries
find_library(NVINFER_LIBRARY nvinfer)
find_library(NVONNXPARSER_LIBRARY nvonnxparser)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin)
# NVML set to optional, try to find in multiple paths
find_library(NVML_LIBRARY nvidia-ml)

# If library not found, try to find in CUDA directory with stubs_ prefix
if(NOT NVML_LIBRARY)
    find_library(NVML_LIBRARY nvidia-ml PATHS /usr/local/cuda-11.8/targets/aarch64-linux/lib/stubs_)
endif()

if(NOT NVINFER_LIBRARY OR NOT NVONNXPARSER_LIBRARY OR NOT NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR "TensorRT libraries not found")
endif()

if(NOT NVML_LIBRARY)
    message(WARNING "NVML library not found, GPU telemetry features will be disabled")
endif()

message(STATUS "Found TensorRT libraries:")
message(STATUS "  nvinfer: ${NVINFER_LIBRARY}")
message(STATUS "  nvonnxparser: ${NVONNXPARSER_LIBRARY}")
message(STATUS "  nvinfer_plugin: ${NVINFER_PLUGIN_LIBRARY}")
message(STATUS "  nvml: ${NVML_LIBRARY}")

# Find Google Logging library
find_library(GLOG_LIBRARY glog)
if(NOT GLOG_LIBRARY)
    message(FATAL_ERROR "Google Logging library not found")
endif()

# Find FFI library
find_library(FFI_LIBRARY ffi)
if(NOT FFI_LIBRARY)
    message(FATAL_ERROR "FFI library not found")
endif()

# Find TIFF library
find_library(TIFF_LIBRARY tiff)
if(NOT TIFF_LIBRARY)
    message(FATAL_ERROR "TIFF library not found")
endif()

# Compile CUDA files
set(CUDA_SOURCE_FILES
    src/inference/preprocess.cu
    src/inference/global_postprocess.cu  # CUDA post-processing file for global model
    src/inference/local_postprocess.cu   # CUDA post-processing file for local model
)
cuda_add_library(cuda_lib STATIC ${CUDA_SOURCE_FILES})

# Video reader source files
set(VIDEO_READER_SOURCES
    src/video/VideoReaderFactory.cpp
    src/video/OpenCVVideoReader.cpp
    src/video/GStreamerVideoReader.cpp
    src/video/FFmpegVideoReader.cpp
)

# camera source files
set(CAMERA_SOURCES
    src/camera/sv_video_base.cpp
    src/camera/sv_video_input.cpp
    src/camera/sv_video_output.cpp
    src/camera/sv_util.cpp
    src/camera/sv_crclib.cpp
    src/camera/ellipse_detector.cpp
    # GStreamer related source files
    src/camera/gstreamer/streamer_gstreamer_impl.cpp
    src/camera/gstreamer/writer_gstreamer_impl.cpp
    # FFmpeg related source files
    src/camera/ffmpeg/bs_push_streamer.cpp
    src/camera/ffmpeg/bs_video_saver.cpp
)

# Flags system source files
set(FLAGS_SOURCES
    src/common/Flags.cpp
)

# Copy config file
configure_file(config/config.flag ${CMAKE_BINARY_DIR}/config.flag COPYONLY)

# Include header files directory
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/include/roi
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tracking
    ${CMAKE_CURRENT_SOURCE_DIR}/include/video
    ${CMAKE_CURRENT_SOURCE_DIR}/include/camera
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ros2
    ${AVCODEC_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
    ${AVDEVICE_INCLUDE_DIRS}
    ${gflags_INCLUDE_DIRS}
)

if(GSTREAMER_FOUND)
    include_directories(${GSTREAMER_INCLUDE_DIRS})
endif()

# Force add fmt library support
include_directories(/usr/local/include)

# Main source files
file(GLOB_RECURSE SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Build executable file
add_executable(gldt_node ${SOURCE_FILES})

# Add ROS2 message dependencies
ament_target_dependencies(gldt_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  builtin_interfaces
  gldt_msgs
)

# Link libraries
set(LINK_LIBS
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
    ${GLOG_LIBRARY}
    ${FFI_LIBRARY}
    ${TIFF_LIBRARY}
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${AVDEVICE_LIBRARIES}
    gflags
    cuda_lib
    stdc++fs  # Add support for std::filesystem in GCC-8
)

# Add fmt library if found
if(FMT_FOUND)
    list(APPEND LINK_LIBS ${FMT_LIBRARIES})
elseif(fmt_FOUND)
    list(APPEND LINK_LIBS fmt::fmt)
endif()

# Only append link if NVML library is found
if(NVML_LIBRARY)
    list(APPEND LINK_LIBS ${NVML_LIBRARY})
endif()

if(GSTREAMER_FOUND AND GSTREAMER_RTSP_FOUND)
    set(LINK_LIBS ${LINK_LIBS} ${GSTREAMER_LIBRARIES} ${GSTREAMER_RTSP_LIBRARIES})
endif()

target_link_libraries(gldt_node ${LINK_LIBS})

# Set output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Add install rules
install(TARGETS gldt_node DESTINATION lib/${PROJECT_NAME})
install(FILES config/config.flag DESTINATION share/${PROJECT_NAME}/config)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter that checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repository)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()