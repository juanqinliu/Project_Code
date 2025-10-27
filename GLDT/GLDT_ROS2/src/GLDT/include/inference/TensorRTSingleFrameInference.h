#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <limits>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "common/Detection.h"
#include "inference/PostprocessCommon.h"
#include "inference/global_postprocess.h"
#include "inference/InferenceInterface.h"
#include "common/Flags.h"
#include "inference/preprocess.h"
#include "common/Logger.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstring>

namespace tracking {

// TensorRT日志记录器（单帧）
class SingleFrameLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT SingleFrame] " << msg << std::endl;
        }
    }
};

// 单帧全局推理类
class TensorRTSingleFrameInference : public InferenceInterface {
public:
    TensorRTSingleFrameInference(const std::string& engine_path);
    ~TensorRTSingleFrameInference() override;

    // 单张图像检测 - 重写基类方法
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.5f) override;
    
    // 批量图像检测 - 重写基类方法
    std::vector<std::vector<Detection>> detectBatch(
        const std::vector<cv::Mat>& images, float conf_threshold = 0.5f) override;
    
    // 设置后处理模式
    void setPostprocessMode(PostprocessMode mode) { postprocess_mode_ = mode; }
    
    // 获取当前后处理模式
    PostprocessMode getPostprocessMode() const { return postprocess_mode_; }
    
    // 检查是否支持批量检测 - 重写基类方法
    bool supportsBatchDetection() const override;
    
    // 获取最大批次大小 - 重写基类方法
    int getMaxBatchSize() const override;
    
    // 从Flags获取全局预处理模式
    static int getGlobalPreprocessMode();
    
    // 从Flags获取全局后处理模式
    static PostprocessMode getGlobalPostprocessMode();
    
    // 🔥 单帧推理时间统计方法
    void updateSingleFramePreprocessTime(double preprocess_time);
    void updateSingleFrameInferenceTime(double inference_time);
    void updateSingleFramePostprocessTime(double postprocess_time);
    void updateSingleFrameTotalTime(double total_time);
    void printSingleFrameTimeStatistics();

private:
    // 初始化绑定
    bool initializeBindings();
    
    // 获取绑定尺寸
    size_t getBindingSize(const nvinfer1::Dims& dims);
    
    // 调试绑定
    void debugBindings();
    
    // 执行批量推理
    std::vector<std::vector<Detection>> executeBatchInference(
        const std::vector<cv::Mat>& batch_images, float conf_threshold);
    
    // 批量预处理
    void preprocessBatch(const std::vector<cv::Mat>& images, float* batch_input_device);
    
    // 批量后处理（CPU版本）
    std::vector<std::vector<Detection>> postprocessBatch(
        float* batch_output, 
        const std::vector<cv::Mat>& original_images, 
        float conf_threshold);
    
    // 批量YOLO解码（CPU版本）
    void batchDecodeYOLOOutput(
        float* batch_output, 
        int batch_size,
        const std::vector<cv::Mat>& original_images,
        std::vector<std::vector<Detection>>& batch_detections,
        float conf_threshold);

    // 预处理相关方法
    void preprocessImageCPU(const cv::Mat& image, float* input_device_buffer);
    void preprocessImageCVAffine(const cv::Mat& image, float* input_device_buffer);
    void preprocessImageGPU(const cv::Mat& image, float* input_device_buffer);
    
    // 辅助函数
    cv::Mat letterbox(const cv::Mat& src);
    void preprocessImage(const cv::Mat& image, float* input_data);
    float clamp(const float val, const float minVal, const float maxVal);
    cv::Rect get_rect(const cv::Mat& img, float bbox[4]);
    
    // CPU后处理函数
    std::vector<Detection> parseYOLOOutput(float* output, int output_size, 
                                     const cv::Mat& original_image, 
                                     float conf_threshold);
    float iou(float lbox[4], float rbox[4]);
    void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh);
    
    // 上下文初始化
    bool initPreprocessContext();
    bool ensurePreprocessBuffer(size_t required_size);
    bool initBatchContext();
    bool ensureBatchBuffers(int batch_size);
    
    // TensorRT相关
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> bindings_;
    cudaStream_t stream_;
    
    // 预处理和批处理配置
    int preprocess_mode_;
    bool cuda_initialized_;
    bool bindings_initialized_;
    bool needs_dynamic_setting_ = false;
    
    // 后处理配置
    PostprocessMode postprocess_mode_ = getGlobalPostprocessMode();
    
    // 模型配置
    cv::Size input_dims_;
    int input_size_;
    int output_size_;
    int num_boxes_;
    int input_index_;
    int detection_output_index_;
    
    // 预处理上下文
    struct PreprocessContext {
        cudaStream_t stream = nullptr;
        void* device_buffer = nullptr;
        size_t buffer_capacity = 0;
        std::mutex mutex;
        
        ~PreprocessContext() {
            if (stream) cudaStreamDestroy(stream);
            if (device_buffer) cudaFree(device_buffer);
        }
    };
    std::unique_ptr<PreprocessContext> preprocess_ctx_;
    
    // 批处理内存上下文
    struct BatchMemoryContext {
        void* input_buffer = nullptr;
        void* output_buffer = nullptr;
        size_t input_capacity = 0;
        size_t output_capacity = 0;
        int max_batch_allocated = 0;
        std::mutex mutex;
        
        ~BatchMemoryContext() {
            if (input_buffer) cudaFree(input_buffer);
            if (output_buffer) cudaFree(output_buffer);
        }
    };
    std::unique_ptr<BatchMemoryContext> batch_ctx_;
    
    // 模型常量
    static constexpr int kInputW = 640;
    static constexpr int kInputH = 640;
    static constexpr int kBoxInfoSize = 5;  // x, y, w, h, conf
    
    // 🔥 单帧推理时间统计成员变量
    double total_preprocess_time_ = 0.0;
    double avg_preprocess_time_ = 0.0;
    double max_preprocess_time_ = 0.0;
    double min_preprocess_time_ = std::numeric_limits<double>::max();
    int preprocess_count_ = 0;
    
    double total_inference_time_ = 0.0;
    double avg_inference_time_ = 0.0;
    double max_inference_time_ = 0.0;
    double min_inference_time_ = std::numeric_limits<double>::max();
    int inference_count_ = 0;
    
    double total_postprocess_time_ = 0.0;
    double avg_postprocess_time_ = 0.0;
    double max_postprocess_time_ = 0.0;
    double min_postprocess_time_ = std::numeric_limits<double>::max();
    int postprocess_count_ = 0;
    
    double total_processing_time_ = 0.0;
    double avg_total_time_ = 0.0;
    double max_total_time_ = 0.0;
    double min_total_time_ = std::numeric_limits<double>::max();
};

} // namespace tracking 


