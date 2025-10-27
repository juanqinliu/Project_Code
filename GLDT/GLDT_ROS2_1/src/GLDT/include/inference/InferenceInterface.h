#ifndef INFERENCE_INTERFACE_H
#define INFERENCE_INTERFACE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "common/Detection.h"  // 确保包含Detection.h

namespace tracking {

class InferenceInterface {
public:
    virtual ~InferenceInterface() = default;
    
    // 单张图像检测
    virtual std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.5f) = 0;
    
    // 🔥 新增：双帧检测方法（使用前后两帧进行运动检测）
    virtual std::vector<Detection> detectWithMotion(const cv::Mat& prev_frame, const cv::Mat& current_frame, float conf_threshold = 0.5f) {
        // 默认实现：忽略前一帧，只对当前帧进行检测
        return detect(current_frame, conf_threshold);
    }
    
    // 🔥 新增：批量检测方法
    virtual std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& images, float conf_threshold = 0.5f) {
        // 默认实现：逐个调用单张检测
        std::vector<std::vector<Detection>> results;
        results.reserve(images.size());
        for (const auto& image : images) {
            results.push_back(detect(image, conf_threshold));
        }
        return results;
    }
    
    // 🔥 新增：检查是否支持真正的批量检测
    virtual bool supportsBatchDetection() const {
        return false; // 默认不支持，子类可以重写
    }
    
    // 🔥 新增：获取支持的最大批次大小
    virtual int getMaxBatchSize() const {
        return 1; // 默认批次大小为1
    }
    
    // 🔥 新增：检查是否支持双帧检测
    virtual bool supportsMotionDetection() const {
        return false; // 默认不支持，子类可以重写
    }
};

} // namespace tracking

#endif // INFERENCE_INTERFACE_H 