#include "inference/InferenceInterface.h"
#include "common/Detection.h"  // Replace TrackingDetection.h with Detection.h
#include "inference/TensorRTLocalInference.h"
#include "inference/TensorRTGlobalInference.h"
#include "inference/TensorRTSingleFrameInference.h"
#ifdef USE_ONNXRUNTIME
#include "ONNXInference.h"
#endif
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace tracking {

std::unique_ptr<InferenceInterface> createInferenceEngine(const std::string& model_path) {
    // Get file extension
    std::string extension = model_path.substr(model_path.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == "engine") {
        return std::make_unique<TensorRTLocalInference>(model_path);
    } else if (extension == "onnx") {
#ifdef USE_ONNXRUNTIME
        return std::make_unique<ONNXInference>(model_path);
#else
        throw std::runtime_error("ONNX Runtime support not enabled. Cannot load .onnx file: " + model_path);
#endif
    } else {
        throw std::runtime_error("Unsupported model format: " + extension + " for file: " + model_path);
    }
    
    // Never reach here, but to avoid compiler warning
    return nullptr;
}

std::unique_ptr<InferenceInterface> createGlobalInferenceEngine(const std::string& model_path) {
    // Get file extension
    std::string extension = model_path.substr(model_path.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == "engine") {
        // Try to create global inference engine with double frame input; if failed, fallback to single frame inference engine
        try {
            return std::make_unique<TensorRTGlobalInference>(model_path);
        } catch (const std::exception& e) {
            // // Maybe the model only contains single input (single frame), use single frame inference engine
            // std::cerr << "[InferenceFactory] 创建TensorRTGlobalInference失败，回退到TensorRTSingleFrameInference: "
            //           << e.what() << std::endl;
            return std::make_unique<TensorRTSingleFrameInference>(model_path);
        }
    } else if (extension == "onnx") {
#ifdef USE_ONNXRUNTIME
        return std::make_unique<ONNXInference>(model_path);
#else
        throw std::runtime_error("ONNX Runtime support not enabled. Cannot load .onnx file: " + model_path);
#endif
    } else {
        throw std::runtime_error("Unsupported model format: " + extension + " for file: " + model_path);
    }
    
    // Never reach here, but to avoid compiler warning
    return nullptr;
}

std::unique_ptr<InferenceInterface> createSingleFrameInferenceEngine(const std::string& model_path) {
    // Get file extension
    std::string extension = model_path.substr(model_path.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == "engine") {
        return std::make_unique<TensorRTSingleFrameInference>(model_path);
    } else if (extension == "onnx") {
#ifdef USE_ONNXRUNTIME
        return std::make_unique<ONNXInference>(model_path);
#else
        throw std::runtime_error("ONNX Runtime support not enabled. Cannot load .onnx file: " + model_path);
#endif
    } else {
        throw std::runtime_error("Unsupported model format: " + extension + " for file: " + model_path);
    }
    
    // Never reach here, but to avoid compiler warning
    return nullptr;
}

std::unique_ptr<InferenceInterface> createLocalInferenceEngine(const std::string& model_path) {
    // Get file extension
    std::string extension = model_path.substr(model_path.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == "engine") {
        return std::make_unique<TensorRTLocalInference>(model_path);
    } else if (extension == "onnx") {
#ifdef USE_ONNXRUNTIME
        return std::make_unique<ONNXInference>(model_path);
#else
        throw std::runtime_error("ONNX Runtime support not enabled. Cannot load .onnx file: " + model_path);
#endif
    } else {
        throw std::runtime_error("Unsupported model format: " + extension + " for file: " + model_path);
    }
    
    // Never reach here, but to avoid compiler warning
    return nullptr;
}

} // namespace tracking 