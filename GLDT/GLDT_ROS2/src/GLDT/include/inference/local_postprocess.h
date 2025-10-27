#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "common/Detection.h"
#include "inference/PostprocessCommon.h"

namespace tracking {
namespace gpu {

// 局部检测用的GPU后处理函数
// -------------------------

// GPU版本的NMS实现
void localNmsGPU(float* boxes, float* scores, int* indices, int& nboxes, int boxes_num, float threshold);

// 解析YOLO输出并在GPU上执行NMS (针对局部检测)
std::vector<Detection> parseLocalYOLOOutputGPU(
    float* output,
    int output_size,
    const cv::Mat& original_image,
    float conf_threshold,
    float nms_threshold = 0.7f);

// 批量处理局部YOLO输出并在GPU上执行NMS
std::vector<std::vector<Detection>> batchDecodeLocalYOLOOutputGPU(
    float* batch_output,
    int batch_size,
    const std::vector<cv::Mat>& original_images,
    float conf_threshold,
    float nms_threshold = 0.7f);

// 帮助函数：将局部检测框从模型输出格式转换为物理坐标
void transformLocalBoxesGPU(
    float* d_boxes,
    int num_boxes,
    const cv::Mat& original_image,
    int input_width,
    int input_height);

} // namespace gpu
} // namespace tracking 