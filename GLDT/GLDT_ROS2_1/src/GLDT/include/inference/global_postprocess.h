#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "common/Detection.h"
#include "inference/PostprocessCommon.h"

namespace tracking {

// GPU全局后处理函数声明
namespace gpu {

// GPU版本的NMS实现
void nmsGPU(float* boxes, float* scores, int* indices, int& nboxes, int boxes_num, float threshold);

// 解析YOLO输出并在GPU上执行NMS (只用于全局检测)
std::vector<Detection> parseYOLOOutputGPU(
    float* output,
    int output_size,
    const cv::Mat& original_image,
    float conf_threshold,
    float nms_threshold = 0.7f);

// 帮助函数：将检测框从模型输出格式转换为物理坐标
void transformBoxesGPU(
    float* d_boxes,
    int num_boxes,
    const cv::Mat& original_image,
    int input_width,
    int input_height);

} // namespace gpu

} // namespace tracking 