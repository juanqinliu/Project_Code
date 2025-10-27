#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 初始化和销毁CUDA预处理环境
void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();

// 主要预处理函数
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height);

// 不同预处理模式的接口
void process_input_gpu(cv::Mat& input_img, float* input_device_buffer);
void process_input_cv_affine(cv::Mat& src, float* input_device_buffer);
void process_input_cpu(cv::Mat& src, float* input_device_buffer);

// ROI预处理相关函数
void cuda_preprocess_roi_safe(uint8_t* src, int src_width, int src_height,
                              float* dst, int dst_width, int dst_height,
                              void* temp_buffer, cudaStream_t stream);
void process_roi_gpu(const cv::Mat& roi, float* output, void* temp_buffer, cudaStream_t stream);

// 线程清理函数
void cleanup_current_thread();

#ifdef __cplusplus
}
#endif

#endif // PREPROCESS_H 