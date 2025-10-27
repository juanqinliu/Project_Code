#pragma once

#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "common/Detection.h"
#include "inference/global_postprocess.h"  // 添加全局后处理头文件引用
#include "inference/InferenceInterface.h"  // 添加接口头文件
#include "common/Flags.h"  // 添加Flags头文件

namespace tracking {

class GlobalLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// 添加继承自InferenceInterface
class TensorRTGlobalInference : public InferenceInterface {
public:
    TensorRTGlobalInference(const std::string& engine_path);
    TensorRTGlobalInference(const std::string& engine_path, int preprocess_mode);
    ~TensorRTGlobalInference() override;
    
    // 重写基类方法
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.5f) override;
    
    // 实现双帧检测方法（原本叫detectWithPreviousFrame，现在映射到接口方法）
    std::vector<Detection> detectWithMotion(const cv::Mat& prev_frame, const cv::Mat& current_frame, float conf_threshold = 0.5f) override {
        // 转发调用到原来的方法
        return detectWithPreviousFrame(current_frame, prev_frame, conf_threshold);
    }
    
    // 检查是否支持动态检测
    bool supportsMotionDetection() const override {
        return true; // 这个类支持双帧检测
    }
    
    // 保留原来的方法名，但实际功能通过接口暴露  
    std::vector<Detection> detectWithPreviousFrame(const cv::Mat& current_frame, const cv::Mat& previous_frame, float conf_threshold = 0.5f);
    
    // 新增：带帧号的优化版本，支持预处理缓存复用
    std::vector<Detection> detectWithPreviousFrame(const cv::Mat& current_frame, const cv::Mat& previous_frame, float conf_threshold, int frame_id);

    // 从Flags获取全局预处理模式
    static int getGlobalPreprocessMode();
    
    // 从Flags获取全局后处理模式
    static PostprocessMode getGlobalPostprocessMode();
    
    // 设置后处理模式（CPU或GPU）
    void setPostprocessMode(PostprocessMode mode) { postprocess_mode_ = mode; }
    
    // 获取当前后处理模式
    PostprocessMode getPostprocessMode() const { return postprocess_mode_; }
    
    // 🔥 全局推理时间统计方法
    void updateGlobalPreprocessTime(double preprocess_time);
    void updateGlobalInferenceTime(double inference_time);
    void updateGlobalPostprocessTime(double postprocess_time);
    void updateGlobalTotalTime(double total_time);
    void printGlobalTimeStatistics();

private:
    // 初始化绑定
    bool initializeBindings();
    
    // 获取绑定尺寸
    size_t getBindingSize(const nvinfer1::Dims& dims);

    // 调试绑定
    void debugBindings();
    
    // 页锁定内存管理
    bool allocatePinnedMemory();
    void freePinnedMemory();

    // 预处理相关方法
    void preprocessImageCPU(const cv::Mat& image, float* input_device_buffer);
    void preprocessImageCVAffine(const cv::Mat& image, float* input_device_buffer);
    void preprocessImageGPU(const cv::Mat& image, float* input_device_buffer);
    cv::Mat letterbox(const cv::Mat& src);
    void preprocessImage(const cv::Mat& image, float* input_data);
    float clamp(const float val, const float minVal, const float maxVal);
    cv::Rect get_rect(const cv::Mat& img, float bbox[4]);
    
    // CPU后处理函数
    std::vector<Detection> parseYOLOOutput(float* output, int output_size, 
                                          const cv::Mat& original_image, float conf_threshold);
    float iou(float lbox[4], float rbox[4]);
    void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh);
    
    // TensorRT相关
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> bindings_;
    cudaStream_t stream_;
    
    // 预处理相关
    int preprocess_mode_;
    bool cuda_initialized_;
    bool bindings_initialized_;
    
    // 后处理相关
    PostprocessMode postprocess_mode_ = getGlobalPostprocessMode();

    // 缓存相关
    cv::Mat cached_previous_frame_;
    bool has_cached_frame_;
    
    // 🔥 帧连续性预处理优化缓存
    int last_processed_frame_id_;          // 上次处理的帧号
    void* cached_previous_preprocessed_;   // 缓存的前一帧预处理结果
    bool has_cached_preprocessed_;         // 是否有预处理缓存
    size_t cached_preprocess_size_;        // 缓存大小
    
    // 模型尺寸
    cv::Size input_dims_;
    int input_size_;
    int output_size_;
    int num_boxes_;

    // 绑定索引
    int current_frame_index_;
    int previous_frame_index_;
    int detection_output_index_;
    
    // 页锁定内存
    float* host_pinned_output_buffer_;
    size_t pinned_output_size_;
    bool use_pinned_memory_;
    
    // 预处理上下文
    struct PreprocessContext {
        void* device_buffer = nullptr;
        size_t buffer_capacity = 0;
        cudaStream_t stream = nullptr;
        std::mutex mutex;
        
        ~PreprocessContext() {
            if (device_buffer) cudaFree(device_buffer);
            if (stream) cudaStreamDestroy(stream);
        }
    };
    std::unique_ptr<PreprocessContext> preprocess_ctx_;
    
    // 🔥 全局推理时间统计成员变量
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
    
    // 🔥 预处理优化统计
    int cache_hit_count_ = 0;           // 缓存命中次数
    int cache_miss_count_ = 0;          // 缓存未命中次数
    
    static constexpr int kInputW = 640;
    static constexpr int kInputH = 640;
    static constexpr int kBoxInfoSize = 5;  // x, y, w, h, conf
};

} // namespace tracking