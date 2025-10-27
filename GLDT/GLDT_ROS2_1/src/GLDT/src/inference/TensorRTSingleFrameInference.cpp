#include "inference/TensorRTSingleFrameInference.h"
#include "common/Detection.h"
#include "inference/preprocess.h"
#include "inference/global_postprocess.h"
#include "common/Logger.h"
#include "common/Flags.h"
#include <chrono>
#include <iomanip>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstring>
#include <cmath>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            LOG_ERROR("CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error_code));\
            throw std::runtime_error("CUDA error");\
        }\
    }
#endif

namespace tracking {

// Get the global preprocess mode from Flags
int TensorRTSingleFrameInference::getGlobalPreprocessMode() {
    return FLAGS_global_preprocess_mode;
}

// Get the global postprocess mode from Flags
PostprocessMode TensorRTSingleFrameInference::getGlobalPostprocessMode() {
    return static_cast<PostprocessMode>(FLAGS_global_postprocess_mode);
}

TensorRTSingleFrameInference::TensorRTSingleFrameInference(const std::string& engine_path)
    : stream_(nullptr), cuda_initialized_(false), bindings_initialized_(false),
      input_index_(-1), detection_output_index_(-1),
      inference_count_(0), preprocess_count_(0), postprocess_count_(0) {
    
    // Read the engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR("Failed to open engine file: " + engine_path);
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create the TensorRT runtime and engine
    static SingleFrameLogger logger;
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) {
        LOG_ERROR("Failed to create TensorRT runtime");
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        LOG_ERROR("Failed to deserialize CUDA engine");
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        LOG_ERROR("Failed to create execution context");
        throw std::runtime_error("Failed to create execution context");
    }

    LOG_INFO("Single frame inference engine loaded: " << engine_path);

    // Create the CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    LOG_INFO("✅ CUDA stream created successfully");
    
    // Initialize the preprocess context
    preprocess_ctx_ = std::make_unique<PreprocessContext>();
    LOG_INFO("✅ Preprocess context initialized successfully");
    
    // Initialize the batch context
    batch_ctx_ = std::make_unique<BatchMemoryContext>();
    LOG_INFO("✅ Batch context initialized successfully");
    
    // Initialize the bindings
    LOG_INFO("🔍 [Single frame inference] Starting to initialize bindings...");
    if (!initializeBindings()) {
        LOG_ERROR("❌ Bindings initialization failed");
        throw std::runtime_error("Failed to initialize bindings");
    }
    LOG_INFO("✅ Bindings initialized successfully");

    // Set the preprocess and postprocess mode
    preprocess_mode_ = getGlobalPreprocessMode();
    postprocess_mode_ = getGlobalPostprocessMode();
    LOG_INFO("Single frame inference engine loaded, preprocess mode: " << preprocess_mode_ 
             << ", postprocess mode: " << (postprocess_mode_ == PostprocessMode::CPU ? "CPU" : "GPU"));
}

TensorRTSingleFrameInference::~TensorRTSingleFrameInference() {
    // Clean up the CUDA resources - add error checking
    if (stream_) {
        cudaError_t status = cudaStreamDestroy(stream_);
        if (status != cudaSuccess) {
            LOG_WARNING("CUDA stream destruction failed: " << cudaGetErrorString(status));
        }
        stream_ = nullptr;
    }
    
    // Clean up the binding memory - add error checking
    for (auto& binding : bindings_) {
        if (binding) {
            cudaError_t status = cudaFree(binding);
            if (status != cudaSuccess) {
                LOG_WARNING("CUDA binding memory release failed: " << cudaGetErrorString(status));
            }
            binding = nullptr;
        }
    }
    bindings_.clear();
    
    // Clean up the preprocess context - add error checking
    if (preprocess_ctx_) {
        if (preprocess_ctx_->stream) {
            cudaError_t status = cudaStreamDestroy(preprocess_ctx_->stream);
            if (status != cudaSuccess) {
                LOG_WARNING("Preprocess CUDA stream destruction failed: " << cudaGetErrorString(status));
            }
            preprocess_ctx_->stream = nullptr;
        }
        if (preprocess_ctx_->device_buffer) {
            cudaError_t status = cudaFree(preprocess_ctx_->device_buffer);
            if (status != cudaSuccess) {
                LOG_WARNING("Preprocess device buffer release failed: " << cudaGetErrorString(status));
            }
            preprocess_ctx_->device_buffer = nullptr;
        }
        preprocess_ctx_.reset();
    }
    
    // Clean up the batch context - add error checking
    if (batch_ctx_) {
        if (batch_ctx_->input_buffer) {
            cudaError_t status = cudaFree(batch_ctx_->input_buffer);
            if (status != cudaSuccess) {
                LOG_WARNING("Batch input buffer release failed: " << cudaGetErrorString(status));
            }
            batch_ctx_->input_buffer = nullptr;
        }
        if (batch_ctx_->output_buffer) {
            cudaError_t status = cudaFree(batch_ctx_->output_buffer);
            if (status != cudaSuccess) {
                LOG_WARNING("Batch output buffer release failed: " << cudaGetErrorString(status));
            }
            batch_ctx_->output_buffer = nullptr;
        }
        batch_ctx_.reset();
    }
    
    // Ensure the CUDA device synchronization
    cudaDeviceSynchronize();
}

bool TensorRTSingleFrameInference::initializeBindings() {
    // Suppress the TensorRT API deprecation warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    
    int nb_bindings = engine_->getNbBindings();
    bindings_.resize(nb_bindings, nullptr);
    
    // Check if dynamic setting is needed
    for (int i = 0; i < nb_bindings; ++i) {
        if (engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    needs_dynamic_setting_ = true;
                    break;
                }
            }
        }
    }
    
    // First find the input and output binding indices
    for (int i = 0; i < nb_bindings; ++i) {
        std::string name = engine_->getBindingName(i);
        if (engine_->bindingIsInput(i)) {
            // Find the input binding
            if (name == "current_frame" || name == "images" || name == "input") {
                input_index_ = i;
                LOG_INFO("🔍 [Bindings finding] Found input binding[" << i << "]: " << name);
            }
        } else {
            // Find the output binding
            if (name == "output0" || name == "output" || name == "detections") {
                detection_output_index_ = i;
                LOG_INFO("🔍 [Bindings finding] Found output binding[" << i << "]: " << name);
            }
        }
    }
    
    // If no specific binding name is found, use the first input and the first output
    if (input_index_ == -1) {
        for (int i = 0; i < nb_bindings; ++i) {
            if (engine_->bindingIsInput(i)) {
                input_index_ = i;
                LOG_INFO("🔍 [Bindings finding] Using the first input binding[" << i << "]: " << engine_->getBindingName(i));
                break;
            }
        }
    }
    
    if (detection_output_index_ == -1) {
        for (int i = 0; i < nb_bindings; ++i) {
            if (!engine_->bindingIsInput(i)) {
                detection_output_index_ = i;
                LOG_INFO("🔍 [Bindings finding] Using the first output binding[" << i << "]: " << engine_->getBindingName(i));
                break;
            }
        }
    }
    
    // Check if the necessary bindings are found
    if (input_index_ == -1 || detection_output_index_ == -1) {
        LOG_ERROR("❌ Necessary bindings not found: input_index_=" << input_index_ 
                  << ", detection_output_index_=" << detection_output_index_);
        return false;
    }
    
    LOG_INFO("✅ [Bindings finding] Input binding index: " << input_index_ << ", output binding index: " << detection_output_index_);
    
    // Allocate the binding memory
    for (int i = 0; i < nb_bindings; ++i) {
        auto dims = engine_->getBindingDimensions(i);
        size_t size = getBindingSize(dims);
        
        LOG_INFO("🔍 [Bindings initialization] Binding[" << i << "] - " 
                 << (engine_->bindingIsInput(i) ? "input" : "output") 
                 << " dimensions: [" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ", " << dims.d[3] << "]"
                 << " size: " << size << " bytes");
        
        void* device_mem;
        CUDA_CHECK(cudaMalloc(&device_mem, size));
        bindings_[i] = device_mem;
        
        if (i == input_index_) {
            input_size_ = size;
            input_dims_ = cv::Size(dims.d[3], dims.d[2]); // W, H
            LOG_INFO("  - ✅ Input binding set completed, size: " << input_dims_.width << "x" << input_dims_.height);
        } else if (i == detection_output_index_) {
            output_size_ = size;
            LOG_INFO("  - ✅ Primary output binding set completed, index: " << detection_output_index_);
        } else {
            LOG_INFO("  - ✅ Other binding[" << i << "] size: " << size << " bytes");
        }
    }
    
    #pragma GCC diagnostic pop
    
    bindings_initialized_ = true;
    debugBindings();
    return true;
}

size_t TensorRTSingleFrameInference::getBindingSize(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == -1) {
            // Dynamic dimension, use the default value
            if (i == 0) {
                size *= 1; // The batch size is default to 1
            } else if (i == 1) {
                size *= 3; // The channel number is default to 3
            } else if (i == 2) {
                size *= 640; // The height is default to 640
            } else if (i == 3) {
                size *= 640; // The width is default to 640
            } else {
                size *= 1; // The other dimensions are default to 1
            }
        } else {
            size *= dims.d[i];
        }
    }
    return size * sizeof(float);
}

void TensorRTSingleFrameInference::debugBindings() {
    if (!FLAGS_log_binding_info) return;
    
    // Suppress the TensorRT API deprecation warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    
    LOG_INFO("=== Single frame inference binding information ===");
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        auto dims = engine_->getBindingDimensions(i);
        std::string binding_info = "Binding " + std::to_string(i) + ": ";
        binding_info += engine_->bindingIsInput(i) ? "INPUT" : "OUTPUT";
        binding_info += " [";
        for (int j = 0; j < dims.nbDims; ++j) {
            if (j > 0) binding_info += ", ";
            binding_info += std::to_string(dims.d[j]);
        }
        binding_info += "]";
        LOG_INFO(binding_info);
    }
    
    #pragma GCC diagnostic pop
    
    LOG_INFO("Input size: " << input_dims_.width << "x" << input_dims_.height);
    LOG_INFO("Input size: " << input_size_ << " bytes");
    LOG_INFO("Output size: " << output_size_ << " bytes");
    LOG_INFO("========================");
}

std::vector<Detection> TensorRTSingleFrameInference::detect(const cv::Mat& image, float conf_threshold) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    if (image.empty()) {
        LOG_WARNING("Input image is empty");
        return {};
    }
    
    // Check if the bindings are initialized
    if (!bindings_initialized_ || bindings_.empty()) {
        LOG_ERROR("Bindings not initialized");
        return {};
    }
    
    if (input_index_ < 0 || input_index_ >= static_cast<int>(bindings_.size()) || 
        bindings_[input_index_] == nullptr) {
        LOG_ERROR("Invalid input binding, index: " << input_index_ << ", binding size: " << bindings_.size());
        return {};
    }
    
    if (detection_output_index_ < 0 || detection_output_index_ >= static_cast<int>(bindings_.size()) || 
        bindings_[detection_output_index_] == nullptr) {
        LOG_ERROR("Invalid output binding, index: " << detection_output_index_ << ", binding size: " << bindings_.size());
        return {};
    }
    
    // Preprocess
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    if (!initPreprocessContext()) {
        LOG_ERROR("Failed to initialize preprocess context");
        return {};
    }
    
    LOG_INFO("🔍 [Single frame inference debug] Starting preprocess, image size: " << image.cols << "x" << image.rows);
    preprocessImage(image, static_cast<float*>(bindings_[input_index_]));
    LOG_INFO("🔍 [Single frame inference debug] Preprocess completed");
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration<double, std::milli>(end_preprocess - start_preprocess).count();
    updateSingleFramePreprocessTime(preprocess_time);
    
    // Inference
    auto start_inference = std::chrono::high_resolution_clock::now();
    
    // 🔥 添加详细的调试信息
    LOG_INFO("🔍 [Single frame inference debug] Starting inference check");
    LOG_INFO("  - Binding size: " << bindings_.size());
    LOG_INFO("  - Input binding[" << input_index_ << "]: " << (bindings_[input_index_] ? "valid" : "invalid"));
    LOG_INFO("  - Output binding[" << detection_output_index_ << "]: " << (bindings_[detection_output_index_] ? "valid" : "invalid"));
    LOG_INFO("  - Input dimensions: " << input_dims_.width << "x" << input_dims_.height);
    LOG_INFO("  - Input size: " << input_size_ << " bytes");
    LOG_INFO("  - Output size: " << output_size_ << " bytes");
    
    // Check if dynamic setting is needed
    if (needs_dynamic_setting_) {
        LOG_INFO("🔍 [Single frame inference debug] Dynamic setting is needed");
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) {
                auto dims = engine_->getBindingDimensions(i);
                LOG_INFO("  - Input binding[" << i << "] original dimensions: [" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ", " << dims.d[3] << "]");
                
                if (dims.d[0] == -1) {
                    nvinfer1::Dims new_dims = dims;
                    new_dims.d[0] = 1;  // Single frame inference
                    LOG_INFO("  - Set input binding[" << i << "] dimensions to: [" << new_dims.d[0] << ", " << new_dims.d[1] << ", " << new_dims.d[2] << ", " << new_dims.d[3] << "]");
                    
                    if (!context_->setBindingDimensions(i, new_dims)) {
                        LOG_ERROR("❌ Set input binding[" << i << "] dimensions failed");
                        return {};
                    }
                    LOG_INFO("  - ✅ Input binding[" << i << "] dimensions set successfully");
                }
            }
        }
        #pragma GCC diagnostic pop
    }
    
    // Check if all input dimensions are specified
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    bool all_dims_specified = context_->allInputDimensionsSpecified();
    LOG_INFO("🔍 [Single frame inference debug] All input dimensions specified: " << (all_dims_specified ? "yes" : "no"));
    
    if (!all_dims_specified) {
        LOG_ERROR("❌ Input dimensions not fully specified, cannot execute inference");
        // Print the current dimensions of all bindings
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) {
                auto dims = context_->getBindingDimensions(i);
                LOG_ERROR("  - Input binding[" << i << "] current dimensions: [" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ", " << dims.d[3] << "]");
            }
        }
        return {};
    }
    #pragma GCC diagnostic pop
    
    // Ensure that all output bindings have valid memory
    for (int i = 0; i < static_cast<int>(bindings_.size()); ++i) {
        if (!engine_->bindingIsInput(i) && bindings_[i] == nullptr) {
            LOG_ERROR("❌ Output binding[" << i << "] memory not allocated");
            return {};
        }
    }
    
    LOG_INFO("🔍 [Single frame inference debug] Starting execution of inference...");
    if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
        LOG_ERROR("❌ TensorRT inference execution failed");
        return {};
    }
    LOG_INFO("🔍 [Single frame inference debug] Inference execution successful");
    
    // Wait for inference to complete
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    auto end_inference = std::chrono::high_resolution_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(end_inference - start_inference).count();
    updateSingleFrameInferenceTime(inference_time);
    
    // Postprocess
    auto start_postprocess = std::chrono::high_resolution_clock::now();
    std::vector<Detection> detections;
    
    LOG_INFO("🔍 [Single frame inference debug] Starting postprocess, using output binding[" << detection_output_index_ << "]");
    
    if (postprocess_mode_ == PostprocessMode::GPU) {
        // GPU postprocess
        detections = gpu::parseYOLOOutputGPU(
            static_cast<float*>(bindings_[detection_output_index_]),
            output_size_ / sizeof(float),
            image,
            conf_threshold
        );
    } else {
        // CPU postprocess - need to copy the GPU data to the CPU memory first
        std::vector<float> cpu_output_data(output_size_ / sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_data.data(), 
                                   bindings_[detection_output_index_], 
                                   output_size_, 
                                   cudaMemcpyDeviceToHost, 
                                   stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
        LOG_INFO("🔍 [Single frame inference debug] GPU data copied to CPU, data size: " << cpu_output_data.size() << " floating point numbers");
        
        detections = parseYOLOOutput(
            cpu_output_data.data(),
            output_size_ / sizeof(float),
            image,
            conf_threshold
        );
    }
    
    LOG_INFO("🔍 [Single frame inference debug] Postprocess completed, detected " << detections.size() << " targets");
    
    auto end_postprocess = std::chrono::high_resolution_clock::now();
    double postprocess_time = std::chrono::duration<double, std::milli>(end_postprocess - start_postprocess).count();
    updateSingleFramePostprocessTime(postprocess_time);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    updateSingleFrameTotalTime(total_time);
    
    return detections;
}

std::vector<std::vector<Detection>> TensorRTSingleFrameInference::detectBatch(
    const std::vector<cv::Mat>& images, float conf_threshold) {
    
    if (images.empty()) {
        LOG_WARNING("Batch detection input is empty");
        return {};
    }
    
    int batch_size = std::min(static_cast<int>(images.size()), getMaxBatchSize());
    if (batch_size != static_cast<int>(images.size())) {
        LOG_WARNING("Batch size exceeds maximum support, truncated to: " << batch_size);
    }
    
    return executeBatchInference(
        std::vector<cv::Mat>(images.begin(), images.begin() + batch_size),
        conf_threshold
    );
}

bool TensorRTSingleFrameInference::supportsBatchDetection() const {
    return true;
}

int TensorRTSingleFrameInference::getMaxBatchSize() const {
    // Suppress the TensorRT API deprecation warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    
    // Get the maximum batch size from the engine
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            if (dims.nbDims >= 4) {
                #pragma GCC diagnostic pop
                return dims.d[0]; // Batch dimension
            }
        }
    }
    
    #pragma GCC diagnostic pop
    return 1;
}

std::vector<std::vector<Detection>> TensorRTSingleFrameInference::executeBatchInference(
    const std::vector<cv::Mat>& batch_images, float conf_threshold) {
    
    int batch_size = batch_images.size();
    
    // Ensure that the batch processing buffer is large enough
    if (!ensureBatchBuffers(batch_size)) {
        LOG_ERROR("Failed to allocate batch processing buffer");
        return {};
    }
    
    // Batch preprocess
    preprocessBatch(batch_images, static_cast<float*>(batch_ctx_->input_buffer));
    
    // Set dynamic batch size
    if (needs_dynamic_setting_) {
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        context_->setBindingDimensions(input_index_, nvinfer1::Dims4{batch_size, 3, input_dims_.height, input_dims_.width});
        #pragma GCC diagnostic pop
    }
    
    // Execute batch inference
    std::vector<void*> batch_bindings = bindings_;
    batch_bindings[input_index_] = batch_ctx_->input_buffer;
    batch_bindings[detection_output_index_] = batch_ctx_->output_buffer;
    
    if (!context_->enqueueV2(batch_bindings.data(), stream_, nullptr)) {
        LOG_ERROR("Batch inference execution failed");
        return {};
    }
    
    // Wait for batch inference to complete
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // Batch postprocess
    return postprocessBatch(
        static_cast<float*>(batch_ctx_->output_buffer),
        batch_images,
        conf_threshold
    );
}

void TensorRTSingleFrameInference::preprocessBatch(const std::vector<cv::Mat>& images, float* batch_input_device) {
    for (size_t i = 0; i < images.size(); ++i) {
        float* img_ptr = batch_input_device + i * 3 * input_dims_.height * input_dims_.width;
        preprocessImage(images[i], img_ptr);
    }
}

std::vector<std::vector<Detection>> TensorRTSingleFrameInference::postprocessBatch(
    float* batch_output, 
    const std::vector<cv::Mat>& original_images, 
    float conf_threshold) {
    
    std::vector<std::vector<Detection>> batch_detections;
    batch_detections.reserve(original_images.size());
    
    int batch_size = original_images.size();
    int output_per_image = output_size_ / sizeof(float) / batch_size;
    
    for (int i = 0; i < batch_size; ++i) {
        float* img_output = batch_output + i * output_per_image;
        auto detections = parseYOLOOutput(img_output, output_per_image, original_images[i], conf_threshold);
        batch_detections.push_back(detections);
    }
    
    return batch_detections;
}

void TensorRTSingleFrameInference::batchDecodeYOLOOutput(
    float* batch_output, 
    int batch_size,
    const std::vector<cv::Mat>& original_images,
    std::vector<std::vector<Detection>>& batch_detections,
    float conf_threshold) {
    
    int output_per_image = output_size_ / sizeof(float) / batch_size;
    
    for (int i = 0; i < batch_size; ++i) {
        float* img_output = batch_output + i * output_per_image;
        auto detections = parseYOLOOutput(img_output, output_per_image, original_images[i], conf_threshold);
        batch_detections.push_back(detections);
    }
}

void TensorRTSingleFrameInference::preprocessImageCPU(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat resized;
    cv::resize(image, resized, input_dims_);
    
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    
    // Convert to CHW format
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    
    int input_h = input_dims_.height;
    int input_w = input_dims_.width;
    
    // Prepare data on the CPU first
    std::vector<float> cpu_data(3 * input_h * input_w);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                int idx = c * input_h * input_w + h * input_w + w;
                cpu_data[idx] = channels[c].at<float>(h, w);
            }
        }
    }
    
    // Copy to the GPU
    CUDA_CHECK(cudaMemcpy(bindings_[input_index_], cpu_data.data(), input_size_, cudaMemcpyHostToDevice));
}

void TensorRTSingleFrameInference::preprocessImageCVAffine(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat letterbox_img = letterbox(image);
    process_input_cv_affine(letterbox_img, input_device_buffer);
}

void TensorRTSingleFrameInference::preprocessImageGPU(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat letterbox_img = letterbox(image);
    process_input_gpu(letterbox_img, input_device_buffer);
}

cv::Mat TensorRTSingleFrameInference::letterbox(const cv::Mat& src) {
    int src_width = src.cols;
    int src_height = src.rows;
    
    float scale = std::min(static_cast<float>(input_dims_.width) / src_width, 
                          static_cast<float>(input_dims_.height) / src_height);
    
    int new_width = static_cast<int>(src_width * scale);
    int new_height = static_cast<int>(src_height * scale);
    
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_width, new_height));
    
    cv::Mat letterbox_img = cv::Mat::zeros(input_dims_.height, input_dims_.width, CV_8UC3);
    
    int x_offset = (input_dims_.width - new_width) / 2;
    int y_offset = (input_dims_.height - new_height) / 2;
    
    cv::Rect roi(x_offset, y_offset, new_width, new_height);
    resized.copyTo(letterbox_img(roi));
    
    return letterbox_img;
}

void TensorRTSingleFrameInference::preprocessImage(const cv::Mat& image, float* input_data) {
    switch (preprocess_mode_) {
        case 0: // CPU
            preprocessImageCPU(image, input_data);
            break;
        case 1: // CPU+GPU混合
            preprocessImageCVAffine(image, input_data);
            break;
        case 2: // GPU
            preprocessImageGPU(image, input_data);
            break;
        default:
            LOG_WARNING("Unknown preprocess mode: " << preprocess_mode_ << ", using CPU mode");
            preprocessImageCPU(image, input_data);
            break;
    }
}

float TensorRTSingleFrameInference::clamp(const float val, const float minVal, const float maxVal) {
    return std::max(minVal, std::min(val, maxVal));
}

cv::Rect TensorRTSingleFrameInference::get_rect(const cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = input_dims_.width / (img.cols * 1.0);
    float r_h = input_dims_.height / (img.rows * 1.0);
    
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (input_dims_.height - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (input_dims_.height - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (input_dims_.width - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (input_dims_.width - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    
    return cv::Rect(static_cast<int>(l), static_cast<int>(t), 
                   static_cast<int>(r - l), static_cast<int>(b - t));
}

std::vector<Detection> TensorRTSingleFrameInference::parseYOLOOutput(float* output, int output_size, 
                                                                    const cv::Mat& original_image, 
                                                                    float conf_threshold) {
    std::vector<Detection> detections;
    
    // 检查输出数据是否有效
    if (output == nullptr) {
        LOG_ERROR("Error: output data is empty");
        return detections;
    }
    
    // Check if the output size is reasonable
    if (output_size <= 0) {
        LOG_ERROR("Error: output size is不合理: " << output_size);
        return detections;
    }
    
    // Get the correct dimensions of the output from the context
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    nvinfer1::Dims output_dims = context_->getBindingDimensions(detection_output_index_);
    #pragma GCC diagnostic pop
    
    // Determine the output format and parameters based on the actual dimensions
    int batch_size = (output_dims.nbDims > 0 && output_dims.d[0] > 0) ? output_dims.d[0] : 1;
    int num_dimensions = output_dims.nbDims;
    
    // Process the YOLOv5 format [batch, 5, num_boxes]
    if (num_dimensions == 3 && output_dims.d[1] == 5) {
        int boxes_dim = output_dims.d[2];  // Number of boxes per channel
        
        // Traverse all detection boxes
        for (int i = 0; i < boxes_dim; i++) {
            float x = output[0 * boxes_dim + i];  // x coordinate
            float y = output[1 * boxes_dim + i];  // y coordinate
            float w = output[2 * boxes_dim + i];  // width
            float h = output[3 * boxes_dim + i];  // height
            float conf = output[4 * boxes_dim + i];  // confidence
            
            // Apply the sigmoid activation function to the confidence (if the confidence is greater than 1.0)
            if (conf > 1.0f) {
                conf = 1.0f / (1.0f + std::exp(-conf));  // sigmoid function
            }
            
            // Filter low confidence and unreasonable boxes
            if (conf < conf_threshold || conf > 1.0f) continue;
            if (w <= 0 || h <= 0 || w > input_dims_.width * 2 || h > input_dims_.height * 2) continue;
            
            // Calculate the scale and offset
            float scale = std::min(float(input_dims_.width) / original_image.cols, float(input_dims_.height) / original_image.rows);
            int offsetx = (input_dims_.width - original_image.cols * scale) / 2;
            int offsety = (input_dims_.height - original_image.rows * scale) / 2;
            
            // Convert the detection box coordinates back to the original image coordinates
            float x1 = (x - w/2 - offsetx) / scale;
            float y1 = (y - h/2 - offsety) / scale;
            float x2 = (x + w/2 - offsetx) / scale;
            float y2 = (y + h/2 - offsety) / scale;
            
            // Ensure that the coordinates are within the image range
            x1 = clamp(x1, 0, original_image.cols);
            y1 = clamp(y1, 0, original_image.rows);
            x2 = clamp(x2, 0, original_image.cols);
            y2 = clamp(y2, 0, original_image.rows);
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            // Check if the final box size is reasonable
            if (width > 5 && height > 5) {
                Detection det;
                det.bbox = cv::Rect2f(x1, y1, width, height);
                det.confidence = conf;
                det.class_id = 0;
                det.is_from_global_model = true;  // Mark as global model detection result
                detections.push_back(det);
            }
        }
    }
    // Process the YOLOv8 output format
    else if (num_dimensions == 3 && output_dims.d[1] > 5) {
        // The YOLOv8 output format is [batch, num_classes+5, num_boxes]
        int num_classes = output_dims.d[1] - 5;
        int num_boxes = output_dims.d[2];
    
        // Traverse all detection boxes
        for (int i = 0; i < num_boxes; i++) {
            float x = output[0 * num_boxes + i];
            float y = output[1 * num_boxes + i];
            float w = output[2 * num_boxes + i];
            float h = output[3 * num_boxes + i];
            float obj_conf = output[4 * num_boxes + i];
            
            if (obj_conf < conf_threshold) continue;
            
            // Find the maximum class confidence
            int class_id = 0;
            float max_class_conf = 0;
            for (int c = 0; c < num_classes; c++) {
                float class_conf = output[(5 + c) * num_boxes + i];
                if (class_conf > max_class_conf) {
                    max_class_conf = class_conf;
                    class_id = c;
                }
            }
        
            // Calculate the final confidence
            float conf = obj_conf * max_class_conf;
            if (conf < conf_threshold) continue;
        
            // Check if the box size is reasonable
            if (w <= 0 || h <= 0 || w > input_dims_.width * 2 || h > input_dims_.height * 2) continue;
            
            // Calculate the scale and offset
            float scale = std::min(float(input_dims_.width) / original_image.cols, float(input_dims_.height) / original_image.rows);
            int offsetx = (input_dims_.width - original_image.cols * scale) / 2;
            int offsety = (input_dims_.height - original_image.rows * scale) / 2;
            
            // Convert the detection box coordinates back to the original image coordinates
            float x1 = (x - w/2 - offsetx) / scale;
            float y1 = (y - h/2 - offsety) / scale;
            float x2 = (x + w/2 - offsetx) / scale;
            float y2 = (y + h/2 - offsety) / scale;
            
            // Ensure that the coordinates are within the image range
            x1 = clamp(x1, 0, original_image.cols);
            y1 = clamp(y1, 0, original_image.rows);
            x2 = clamp(x2, 0, original_image.cols);
            y2 = clamp(y2, 0, original_image.rows);
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            // Check if the final box size is reasonable
            if (width > 5 && height > 5) {
                Detection det;
                det.bbox = cv::Rect2f(x1, y1, width, height);
                det.confidence = conf;
                det.class_id = class_id;
                det.is_from_global_model = true;  // Mark as global model detection result
                detections.push_back(det);
            }
        }
    }
    // Unknown output format, try default processing
    else {
        // Assume the output is in the [batch, num_boxes, 5+num_classes] format
        int box_size = 5;  // Default value: x,y,w,h,obj_conf
        int num_boxes = output_size / box_size;  // Estimated number of boxes
        
        for (int i = 0; i < num_boxes; i++) {
            float* box_data = output + i * box_size;
            float x = box_data[0];
            float y = box_data[1];
            float w = box_data[2];
            float h = box_data[3];
            float conf = box_data[4];
            
            // Filter low confidence and unreasonable boxes
            if (conf < conf_threshold || conf > 1.0f) continue;
            if (w <= 0 || h <= 0 || w > input_dims_.width * 2 || h > input_dims_.height * 2) continue;
        
            // Calculate the scale and offset
            float scale = std::min(float(input_dims_.width) / original_image.cols, float(input_dims_.height) / original_image.rows);
            int offsetx = (input_dims_.width - original_image.cols * scale) / 2;
            int offsety = (input_dims_.height - original_image.rows * scale) / 2;
            
            // Convert the detection box coordinates back to the original image coordinates
            float x1 = (x - w/2 - offsetx) / scale;
            float y1 = (y - h/2 - offsety) / scale;
            float x2 = (x + w/2 - offsetx) / scale;
            float y2 = (y + h/2 - offsety) / scale;
            
            // Ensure that the coordinates are within the image range
            x1 = clamp(x1, 0, original_image.cols);
            y1 = clamp(y1, 0, original_image.rows);
            x2 = clamp(x2, 0, original_image.cols);
            y2 = clamp(y2, 0, original_image.rows);
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            // Check if the final box size is reasonable
            if (width > 5 && height > 5) {
                Detection det;
                det.bbox = cv::Rect2f(x1, y1, width, height);
                det.confidence = conf;
                det.class_id = 0;
                det.is_from_global_model = true;  // Mark as global model detection result
                detections.push_back(det);
            }
        }
    }
    
    // Apply NMS - using the configured threshold instead of hardcoding
    if (!detections.empty()) {
        float used_nms = 0.45f;  // Using the same NMS threshold as the global inference
        nms(detections, nullptr, conf_threshold, used_nms);
    }
    
    return detections;
}

float TensorRTSingleFrameInference::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]),
        std::min(lbox[2], rbox[2]),
        std::max(lbox[1], rbox[1]),
        std::min(lbox[3], rbox[3]),
    };
    if (interBox[2] > interBox[3] || interBox[0] > interBox[1]) return 0.0f;
    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS);
}

void TensorRTSingleFrameInference::nms(std::vector<Detection>& res, float* output, 
                                      float conf_thresh, float nms_thresh) {
    // If output is nullptr, it means detections already contains all the required data
    if (output == nullptr) {
        // Sort by confidence
        std::sort(res.begin(), res.end(), [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
        
        // Execute NMS
        std::vector<Detection> nms_result;
        for (size_t i = 0; i < res.size(); ++i) {
            bool keep = true;
            for (size_t j = 0; j < nms_result.size(); ++j) {
                float lbox[4] = {res[i].bbox.x, res[i].bbox.y, 
                                res[i].bbox.x + res[i].bbox.width, 
                                res[i].bbox.y + res[i].bbox.height};
                float rbox[4] = {nms_result[j].bbox.x, nms_result[j].bbox.y, 
                                nms_result[j].bbox.x + nms_result[j].bbox.width, 
                                nms_result[j].bbox.y + nms_result[j].bbox.height};
                if (iou(lbox, rbox) > nms_thresh) {
                    keep = false;
                    break;
                }
            }
            if (keep) nms_result.push_back(res[i]);
        }
        res = std::move(nms_result);
        return;
    }
    
    // 原有的处理逻辑（用于向后兼容）
    std::vector<Detection> detections;
    
    int num_boxes = output_size_ / sizeof(float) / 6;
    float* output_ptr = output;
    
    for (int i = 0; i < num_boxes; ++i) {
        float confidence = output_ptr[4];
        if (confidence > conf_thresh) {
            Detection det;
            det.confidence = confidence;
            det.class_id = 0;
            
            float bbox[4] = {output_ptr[0], output_ptr[1], output_ptr[2], output_ptr[3]};
            det.bbox = cv::Rect2f(bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2, bbox[2], bbox[3]);
            
            detections.push_back(det);
        }
        output_ptr += 6;
    }
    
    // Sort by confidence
    std::sort(detections.begin(), detections.end(), 
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    // Execute NMS
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        res.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            float lbox[4] = {detections[i].bbox.x + detections[i].bbox.width/2,
                             detections[i].bbox.y + detections[i].bbox.height/2,
                             detections[i].bbox.width, detections[i].bbox.height};
            float rbox[4] = {detections[j].bbox.x + detections[j].bbox.width/2,
                             detections[j].bbox.y + detections[j].bbox.height/2,
                             detections[j].bbox.width, detections[j].bbox.height};
            float iou_val = iou(lbox, rbox);
            
            if (iou_val > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }
}

bool TensorRTSingleFrameInference::initPreprocessContext() {
    if (!preprocess_ctx_) {
        preprocess_ctx_ = std::make_unique<PreprocessContext>();
    }
    
    if (!preprocess_ctx_->stream) {
        cudaError_t status = cudaStreamCreate(&preprocess_ctx_->stream);
        if (status != cudaSuccess) {
            LOG_ERROR("Failed to create preprocess CUDA stream: " << cudaGetErrorString(status));
            return false;
        }
    }
    
    return true;
}

bool TensorRTSingleFrameInference::ensurePreprocessBuffer(size_t required_size) {
    if (!preprocess_ctx_) {
        return false;
    }
    
    if (preprocess_ctx_->buffer_capacity < required_size) {
        if (preprocess_ctx_->device_buffer) {
            cudaError_t status = cudaFree(preprocess_ctx_->device_buffer);
            if (status != cudaSuccess) {
                LOG_WARNING("Failed to release preprocess device buffer: " << cudaGetErrorString(status));
            }
            preprocess_ctx_->device_buffer = nullptr;
        }
        
        cudaError_t status = cudaMalloc(&preprocess_ctx_->device_buffer, required_size);
        if (status != cudaSuccess) {
            LOG_ERROR("Failed to allocate preprocess device buffer: " << cudaGetErrorString(status));
            return false;
        }
        preprocess_ctx_->buffer_capacity = required_size;
    }
    
    return true;
}

bool TensorRTSingleFrameInference::initBatchContext() {
    if (!batch_ctx_) {
        batch_ctx_ = std::make_unique<BatchMemoryContext>();
    }
    return true;
}

bool TensorRTSingleFrameInference::ensureBatchBuffers(int batch_size) {
    if (!initBatchContext()) {
        return false;
    }
    
    size_t required_input_size = batch_size * 3 * input_dims_.height * input_dims_.width * sizeof(float);
    size_t required_output_size = batch_size * output_size_ / sizeof(float) * sizeof(float);
    
    // Allocate input buffer
    if (batch_ctx_->input_capacity < required_input_size) {
        if (batch_ctx_->input_buffer) {
            cudaError_t status = cudaFree(batch_ctx_->input_buffer);
            if (status != cudaSuccess) {
                LOG_WARNING("Failed to release batch input buffer: " << cudaGetErrorString(status));
            }
            batch_ctx_->input_buffer = nullptr;
        }
        cudaError_t status = cudaMalloc(&batch_ctx_->input_buffer, required_input_size);
        if (status != cudaSuccess) {
            LOG_ERROR("Failed to allocate batch input buffer: " << cudaGetErrorString(status));
            return false;
        }
        batch_ctx_->input_capacity = required_input_size;
    }
    
    // Allocate output buffer
    if (batch_ctx_->output_capacity < required_output_size) {
        if (batch_ctx_->output_buffer) {
            cudaError_t status = cudaFree(batch_ctx_->output_buffer);
            if (status != cudaSuccess) {
                LOG_WARNING("Failed to release batch output buffer: " << cudaGetErrorString(status));
            }
            batch_ctx_->output_buffer = nullptr;
        }
        cudaError_t status = cudaMalloc(&batch_ctx_->output_buffer, required_output_size);
        if (status != cudaSuccess) {
            LOG_ERROR("Failed to allocate batch output buffer: " << cudaGetErrorString(status));
            return false;
        }
        batch_ctx_->output_capacity = required_output_size;
    }
    
    batch_ctx_->max_batch_allocated = batch_size;
    return true;
}

// Time statistics method implementation
void TensorRTSingleFrameInference::updateSingleFramePreprocessTime(double preprocess_time) {
    if (preprocess_ctx_) {
        std::lock_guard<std::mutex> lock(preprocess_ctx_->mutex);
        total_preprocess_time_ += preprocess_time;
        preprocess_count_++;
        avg_preprocess_time_ = total_preprocess_time_ / preprocess_count_;
        max_preprocess_time_ = std::max(max_preprocess_time_, preprocess_time);
        min_preprocess_time_ = std::min(min_preprocess_time_, preprocess_time);
    } else {
        // If the preprocess context has been released, update the statistics directly
        total_preprocess_time_ += preprocess_time;
        preprocess_count_++;
        avg_preprocess_time_ = total_preprocess_time_ / preprocess_count_;
        max_preprocess_time_ = std::max(max_preprocess_time_, preprocess_time);
        min_preprocess_time_ = std::min(min_preprocess_time_, preprocess_time);
    }
}

void TensorRTSingleFrameInference::updateSingleFrameInferenceTime(double inference_time) {
    total_inference_time_ += inference_time;
    inference_count_++;
    avg_inference_time_ = total_inference_time_ / inference_count_;
    max_inference_time_ = std::max(max_inference_time_, inference_time);
    min_inference_time_ = std::min(min_inference_time_, inference_time);
}

void TensorRTSingleFrameInference::updateSingleFramePostprocessTime(double postprocess_time) {
    total_postprocess_time_ += postprocess_time;
    postprocess_count_++;
    avg_postprocess_time_ = total_postprocess_time_ / postprocess_count_;
    max_postprocess_time_ = std::max(max_postprocess_time_, postprocess_time);
    min_postprocess_time_ = std::min(min_postprocess_time_, postprocess_time);
}

void TensorRTSingleFrameInference::updateSingleFrameTotalTime(double total_time) {
    total_processing_time_ += total_time;
    avg_total_time_ = total_processing_time_ / (preprocess_count_ + inference_count_ + postprocess_count_) * 3;
    max_total_time_ = std::max(max_total_time_, total_time);
    min_total_time_ = std::min(min_total_time_, total_time);
}

void TensorRTSingleFrameInference::printSingleFrameTimeStatistics() {
    LOG_INFO("=== Single frame inference time statistics ===");
    LOG_INFO("Preprocess - Total count: " << preprocess_count_ 
            << ", Average: " << std::fixed << std::setprecision(2) << avg_preprocess_time_ << "ms"
            << ", Max: " << max_preprocess_time_ << "ms"
            << ", Min: " << min_preprocess_time_ << "ms");
    
    LOG_INFO("Inference - Total count: " << inference_count_
            << ", Average: " << avg_inference_time_ << "ms"
            << ", Max: " << max_inference_time_ << "ms"
            << ", Min: " << min_inference_time_ << "ms");
    
    LOG_INFO("Postprocess - Total count: " << postprocess_count_
            << ", Average: " << avg_postprocess_time_ << "ms"
            << ", Max: " << max_postprocess_time_ << "ms"
            << ", Min: " << min_postprocess_time_ << "ms");
    
    LOG_INFO("Total processing - Average: " << avg_total_time_ << "ms"
            << ", Max: " << max_total_time_ << "ms"
            << ", Min: " << min_total_time_ << "ms");
    LOG_INFO("========================");
}

} // namespace tracking 