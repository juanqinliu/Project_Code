#ifndef INFERENCE_FACTORY_H
#define INFERENCE_FACTORY_H

#include "InferenceInterface.h"
#include <memory>
#include <string>

namespace tracking {

// 创建推理引擎（通用方法，根据文件类型自动选择）
std::unique_ptr<InferenceInterface> createInferenceEngine(const std::string& model_path);

// 创建全局推理引擎（支持双帧输入）
std::unique_ptr<InferenceInterface> createGlobalInferenceEngine(const std::string& model_path);

// 创建单帧推理引擎（优化版全局推理）
std::unique_ptr<InferenceInterface> createSingleFrameInferenceEngine(const std::string& model_path);

// 创建局部推理引擎（支持单帧和批量输入）
std::unique_ptr<InferenceInterface> createLocalInferenceEngine(const std::string& model_path);

} // namespace tracking

#endif // INFERENCE_FACTORY_H