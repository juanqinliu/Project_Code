#include "inference/TensorRTGlobalInference.h"
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

const char* kCurrentFrameTensorName = "current_frame";
const char* kPreviousFrameTensorName = "previous_frame";
const char* kGlobalOutputTensorName = "output0";

// GlobalLogger implementation
void GlobalLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        LOG_INFO("[TensorRT Global] " << msg);
    }
}

// Get global preprocess mode from Flags
int TensorRTGlobalInference::getGlobalPreprocessMode() {
    return FLAGS_global_preprocess_mode;
}

// Get global postprocess mode from Flags
PostprocessMode TensorRTGlobalInference::getGlobalPostprocessMode() {
    return static_cast<PostprocessMode>(FLAGS_global_postprocess_mode);
}

TensorRTGlobalInference::TensorRTGlobalInference(const std::string& engine_path)
    : stream_(nullptr), cuda_initialized_(false),
      bindings_initialized_(false), current_frame_index_(-1), previous_frame_index_(-1), detection_output_index_(-1),
      has_cached_frame_(false), host_pinned_output_buffer_(nullptr), pinned_output_size_(0), use_pinned_memory_(true),
      inference_count_(0), preprocess_count_(0), postprocess_count_(0),
      last_processed_frame_id_(-1), cached_previous_preprocessed_(nullptr), has_cached_preprocessed_(false), cached_preprocess_size_(0),
      cache_hit_count_(0), cache_miss_count_(0) {
    // Read engine file
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

    static GlobalLogger logger;
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize CUDA engine");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create execution context");

    LOG_INFO("Global inference engine loaded: " << engine_path);

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // Initialize preprocess context
    preprocess_ctx_ = std::make_unique<PreprocessContext>();
    
    // Initialize bindings
    if (!initializeBindings()) {
        LOG_ERROR("Failed to initialize bindings");
        throw std::runtime_error("Failed to initialize bindings");
    }

    // Initialize pinned memory
    if (use_pinned_memory_ && !allocatePinnedMemory()) {
        LOG_WARNING("Initialize pinned memory failed, use regular memory");
        use_pinned_memory_ = false;
    }
    preprocess_mode_ = getGlobalPreprocessMode();
    LOG_INFO("Global inference engine loaded: " << engine_path << ", preprocess mode: " << preprocess_mode_);
    // 设置后处理模式（只通过gflags）
    postprocess_mode_ = getGlobalPostprocessMode();
    LOG_INFO("Global postprocess mode: " << (postprocess_mode_ == PostprocessMode::CPU ? "CPU" : "GPU"));
}

bool TensorRTGlobalInference::initializeBindings() {
    // Use modern TensorRT API
    int nb_tensors = engine_->getNbIOTensors();
    bindings_.resize(nb_tensors, nullptr);
    
    // Remove detailed binding information log
    
    // Find input and output tensor indices
    for (int i = 0; i < nb_tensors; ++i) {
        std::string name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(name.c_str());
        
        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            if (name == kCurrentFrameTensorName) current_frame_index_ = i;
            if (name == kPreviousFrameTensorName) previous_frame_index_ = i;
        } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (name == kGlobalOutputTensorName) detection_output_index_ = i;
        }
    }
    
    // Check if all necessary bindings are found
    if (current_frame_index_ == -1 || previous_frame_index_ == -1 || detection_output_index_ == -1) {
        LOG_ERROR("Required bindings not found: current_frame_index_=" << current_frame_index_ 
                  << ", previous_frame_index_=" << previous_frame_index_ 
                << ", detection_output_index_=" << detection_output_index_);
        return false;
    }
    
    // Check if there are dynamic shapes
    bool has_dynamic_shapes = false;
    for (int i = 0; i < nb_tensors; ++i) {
        std::string name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                has_dynamic_shapes = true;
                break;
            }
        }
        if (has_dynamic_shapes) break;
    }
    
    // Get input and output dimensions
    auto input_dims = engine_->getTensorShape(kCurrentFrameTensorName);
    input_dims_ = cv::Size(kInputW, kInputH);
    
    // For dynamic shapes, use predefined dimensions
    if (has_dynamic_shapes) {
        LOG_INFO("Use predefined dimensions to allocate memory: " << kInputW << "x" << kInputH);
        input_size_ = 3 * kInputH * kInputW;
    } else {
        // For static shapes, use engine defined dimensions
        input_size_ = 1;
        for (int j = 0; j < input_dims.nbDims; ++j) {
            if (input_dims.d[j] > 0) input_size_ *= input_dims.d[j];
        }
    }
    
    // Get output dimensions and format
    auto output_dims = engine_->getTensorShape(kGlobalOutputTensorName);
    
    // Remove output dimension log
    
    // For dynamic output shapes, try different output formats
    if (has_dynamic_shapes) {
        // Check output dimensions, try to determine correct format
        if (output_dims.nbDims >= 2) {
            // If output is [batch_size, num_boxes, box_info_size]
            if (output_dims.nbDims == 3) {
                int batch_size = output_dims.d[0] > 0 ? output_dims.d[0] : 1;
                int num_boxes_dim = output_dims.d[1] > 0 ? output_dims.d[1] : 25200;  // Common YOLO output box number
                int box_info_size_dim = output_dims.d[2] > 0 ? output_dims.d[2] : kBoxInfoSize;
                
                output_size_ = batch_size * num_boxes_dim * box_info_size_dim;
                num_boxes_ = num_boxes_dim;
                
                // Remove output format detection log
            }
            // If output is [batch_size, num_boxes * box_info_size]
            else if (output_dims.nbDims == 2) {
                int batch_size = output_dims.d[0] > 0 ? output_dims.d[0] : 1;
                int total_elements = output_dims.d[1] > 0 ? output_dims.d[1] : 25200 * kBoxInfoSize;
                
                // Assume interleaved format [x,y,w,h,conf, x,y,w,h,conf, ...]
                num_boxes_ = total_elements / kBoxInfoSize;
                output_size_ = batch_size * total_elements;
                
            }
            // If other format, use default settings
            else {
                // Assume output is [1, 25200, 85] (typical YOLO output)
                num_boxes_ = 25200;
                output_size_ = num_boxes_ * kBoxInfoSize;
                // Remove default output format setting log
            }
        } else {
            // Dimension不足，使用默认设置
            num_boxes_ = 25200;
            output_size_ = num_boxes_ * kBoxInfoSize;
            // Remove dimension不足日志
        }
    } else {
        // For static shapes, use engine defined dimensions
        output_size_ = 1;
        for (int j = 0; j < output_dims.nbDims; ++j) {
            if (output_dims.d[j] > 0) output_size_ *= output_dims.d[j];
        }
        
        // Try to determine correct box number
        if (output_dims.nbDims >= 2) {
            if (output_dims.nbDims == 3 && output_dims.d[2] == kBoxInfoSize) {
                // If output is [batch_size, num_boxes, box_info_size]
                num_boxes_ = output_dims.d[1];
            } else if (output_dims.nbDims == 2) {
                // If output is [batch_size, num_boxes * box_info_size]
                num_boxes_ = output_dims.d[1] / kBoxInfoSize;
            } else {
                // Default calculation
                num_boxes_ = output_size_ / kBoxInfoSize;
            }
        } else {
            // Default calculation
            num_boxes_ = output_size_ / kBoxInfoSize;
        }
    }
    
    // Allocate memory
    cudaError_t status;
    
    // Allocate memory for all bindings
    for (int i = 0; i < nb_tensors; ++i) {
        std::string name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(name.c_str());
        bool isInput = (io_mode == nvinfer1::TensorIOMode::kINPUT);
        nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
        
        // Determine allocation size
        size_t elem_size = sizeof(float);  // Assume all tensors are float type
        size_t size = 1;
        
        if (has_dynamic_shapes) {
            // Allocate reasonable size for dynamic shapes
            if (isInput) {
                // For standard input, use calculated size
                if (i == current_frame_index_ || i == previous_frame_index_) {
                size = input_size_;
            } else {
                    // Other inputs use default size
                    size = 1000000;  // 1M浮点数
                }
            } else {
                // For standard output, use calculated size
                if (i == detection_output_index_) {
                    size = output_size_;
                } else {
                    // Intermediate output uses large enough size, based on image size estimation
                    // Assume maximum is 4 times the input size
                    size = input_size_ * 4;  
                }
            }
        } else {
            // Static shape, directly calculate size
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] > 0) {
                    size *= dims.d[j];
                }
            }
        }
        
        // Allocate memory
        size_t mem_size = size * elem_size;
        status = cudaMalloc(&bindings_[i], mem_size);
        if (status != cudaSuccess) {
            LOG_ERROR("Failed to allocate memory for binding " << i << " (" << name << "): " << cudaGetErrorString(status));
            // Release all previously allocated memory
            for (int j = 0; j < i; ++j) {
                if (bindings_[j]) {
                    cudaFree(bindings_[j]);
                    bindings_[j] = nullptr;
                }
            }
            return false;
        }
        
        // Remove memory allocation log
    }
    
    bindings_initialized_ = true;
    return true;
}

size_t TensorRTGlobalInference::getBindingSize(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] > 0) size *= dims.d[j];
    }
    return size;
}

void TensorRTGlobalInference::debugBindings() {
    // Debug function, only used when explicitly needed
}

TensorRTGlobalInference::~TensorRTGlobalInference() {
    // Release cached frame
    if (has_cached_frame_) {
        cached_previous_frame_.release();
        has_cached_frame_ = false;
    }

    // 🔥 Release preprocess cache
    if (cached_previous_preprocessed_) {
        cudaFree(cached_previous_preprocessed_);
        cached_previous_preprocessed_ = nullptr;
        has_cached_preprocessed_ = false;
    }

    // Release pinned memory
    freePinnedMemory();

    // Release CUDA resources
    for (void* binding : bindings_) {
        if (binding) cudaFree(binding);
    }
    if (stream_) cudaStreamDestroy(stream_);
    if (cuda_initialized_) {
        cuda_preprocess_destroy();
    }
}

// Allocate pinned memory
bool TensorRTGlobalInference::allocatePinnedMemory() {
    // Release existing pinned memory
    freePinnedMemory();
    
    // Use output_size_ as initial size
    size_t size = output_size_ * sizeof(float);
    cudaError_t status = cudaMallocHost((void**)&host_pinned_output_buffer_, size);
    
    if (status != cudaSuccess) {
        LOG_ERROR("Failed to allocate pinned memory: " << cudaGetErrorString(status));
        host_pinned_output_buffer_ = nullptr;
        pinned_output_size_ = 0;
        return false;
    }
    
    pinned_output_size_ = output_size_;
    // Remove pinned memory allocation log
    return true;
}

// Release pinned memory
void TensorRTGlobalInference::freePinnedMemory() {
    if (host_pinned_output_buffer_) {
        cudaFreeHost(host_pinned_output_buffer_);
        host_pinned_output_buffer_ = nullptr;
        pinned_output_size_ = 0;
    }
}

// Implement single frame detection method, using black image as previous frame
std::vector<Detection> TensorRTGlobalInference::detect(const cv::Mat& image, float conf_threshold) {
    // Check if there is a cached previous frame
    if (!has_cached_frame_) {
        // First run, create black image as previous frame
    cv::Mat black_frame = cv::Mat::zeros(image.size(), image.type());
        // Call double frame detection method and start caching
    return detectWithPreviousFrame(image, black_frame, conf_threshold);
    } else {
        // Use cached previous frame
        return detectWithPreviousFrame(image, cached_previous_frame_, conf_threshold);
    }
}

std::vector<Detection> TensorRTGlobalInference::detectWithPreviousFrame(const cv::Mat& current_frame, const cv::Mat& previous_frame, float conf_threshold) {
    if (!bindings_initialized_) {
        LOG_ERROR("Error: bindings not initialized");
        return {};
    }
    
    // 🔥 Global inference time statistics start
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Check if engine uses dynamic shapes
    bool has_dynamic_shapes = false;
    int nb_tensors = engine_->getNbIOTensors();
    for (int i = 0; i < nb_tensors; ++i) {
        std::string name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(name.c_str());
        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    has_dynamic_shapes = true;
                    break;
                }
            }
            if (has_dynamic_shapes) break;
        }
    }
    
    // If there are dynamic shapes, set input dimensions
    if (has_dynamic_shapes) {
        // Set dimensions for all inputs
        for (int i = 0; i < nb_tensors; ++i) {
            std::string name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(name.c_str());
            if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
                nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
                bool has_dynamic_dim = false;
                
                // Check if there are dynamic dimensions
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] == -1) {
                        has_dynamic_dim = true;
                        break;
                    }
                }
                
                if (has_dynamic_dim) {
                    // Set input dimensions
                    if (i == current_frame_index_ || i == previous_frame_index_) {
                        // For image input, set to standard size
                        dims.d[0] = 1;  // Batch size
                        dims.d[1] = 3;  // Channels
                        dims.d[2] = kInputH;  // Height
                        dims.d[3] = kInputW;  // Width
                    } else {
                        // For other inputs, set a default value
                        for (int j = 0; j < dims.nbDims; ++j) {
                            if (dims.d[j] == -1) {
                                dims.d[j] = 1;  // Default set to 1
                            }
                        }
                    }
                    
                    // Apply dimension settings
                    if (!context_->setInputShape(name.c_str(), dims)) {
                        LOG_ERROR("Failed to set dimensions for binding " << i << " (" << name << ")");
                        return {};
                    }
                }
            }
        }
        
        // Check if all input dimensions are specified
        if (!context_->allInputDimensionsSpecified()) {
            LOG_ERROR("Error: not all input dimensions are specified");
            return {};
        }

        // Update memory size for all output bindings
        for (int i = 0; i < nb_tensors; ++i) {
            std::string name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(name.c_str());
            if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
                // This is an output binding, get updated dimensions
                nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
                size_t new_size = 1;
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] > 0) {
                        new_size *= dims.d[j];
                    }
                }
                
                // Calculate memory size (assume float type)
                size_t mem_size = new_size * sizeof(float);
                
                // If this is an output binding we previously processed
                if (i == detection_output_index_) {
                    if (new_size > output_size_) {
                        if (bindings_[i]) {
                            cudaFree(bindings_[i]);
                        }
                        cudaError_t status = cudaMalloc(&bindings_[i], mem_size);
                        if (status != cudaSuccess) {
                            LOG_ERROR("Failed to reallocate memory for output binding " << i << ": " 
                                      << cudaGetErrorString(status));
                            return {};
                        }
                        output_size_ = new_size;
                        
                        // If using pinned memory and size is not enough, reallocate
                        if (use_pinned_memory_ && pinned_output_size_ < output_size_) {
                            if (!allocatePinnedMemory()) {
                                LOG_WARNING("Warning: failed to reallocate pinned memory, using regular memory");
                                use_pinned_memory_ = false;
                            }
                        }
                    }
                }
                // Process other all output bindings
                else {
                    // Release old memory and allocate new memory
                    if (bindings_[i]) {
                        cudaFree(bindings_[i]);
                    }
                    cudaError_t status = cudaMalloc(&bindings_[i], mem_size);
                    if (status != cudaSuccess) {
                        LOG_ERROR("Failed to allocate memory for intermediate output binding " << i << ": " 
                                  << cudaGetErrorString(status));
                        return {};
                    }
                }
            }
        }
    }

    // No need to debug binding information
    
    // Ensure input images are valid
    if (current_frame.empty() || previous_frame.empty()) {
        LOG_ERROR("Error: input images are empty");
        return {};
    }
    
    // 🔥 Global preprocess time statistics start
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    // According to preprocess mode, select different preprocess methods
    try {
    switch (preprocess_mode_) {
        case 0: // CPU full preprocess
            preprocessImageCPU(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
            preprocessImageCPU(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
            break;
        case 1: // CV affine transform + GPU preprocess
            preprocessImageCVAffine(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
            preprocessImageCVAffine(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
            break;
        case 2: // Full GPU preprocess
            preprocessImageGPU(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
            preprocessImageGPU(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
            break;
        default:
                LOG_ERROR("Unknown preprocess mode: " << preprocess_mode_ << ", using CPU preprocess");
            preprocessImageCPU(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
            preprocessImageCPU(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
            break;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Exception occurred during preprocess: " << e.what());
        return {};
    }
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    updateGlobalPreprocessTime(preprocess_time);
    
    // Check CUDA errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("CUDA error after preprocess: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    // Check if bindings are valid
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        if (bindings_[i] == nullptr) {
            std::string name = engine_->getIOTensorName(i);
            LOG_ERROR("Error: binding " << i << " (" << name << ") is empty");
            return {};
        }
    }

    // Ensure CUDA device is synchronized
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("CUDA stream synchronization failed: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    // 🔥 Global inference time statistics start
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    // Inference
    bool success = context_->enqueueV2(bindings_.data(), stream_, nullptr);
    if (!success) {
        LOG_ERROR("❌ Inference execution failed");
        
        // Try to get the last CUDA error
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            LOG_ERROR("CUDA error: " << cudaGetErrorString(cuda_status));
        }
        
        return {};
    }
    
    // Synchronize stream and check errors
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("Inference stream synchronization failed: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    auto inference_end = std::chrono::high_resolution_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    updateGlobalInferenceTime(inference_time);
    
    // Use pinned memory or regular vector to get output data
    float* output_data_ptr = nullptr;
    std::vector<float> temp_output_data;
    
    if (use_pinned_memory_ && host_pinned_output_buffer_) {
        // Ensure pinned memory size is enough
        if (pinned_output_size_ < output_size_) {
            // Reallocate
            if (!allocatePinnedMemory()) {
                LOG_WARNING("Warning: failed to reallocate pinned memory, using regular memory");
                use_pinned_memory_ = false;
                temp_output_data.resize(output_size_);
                output_data_ptr = temp_output_data.data();
            } else {
                output_data_ptr = host_pinned_output_buffer_;
            }
        } else {
            output_data_ptr = host_pinned_output_buffer_;
        }
    } else {
        // Use regular vector
        temp_output_data.resize(output_size_);
        output_data_ptr = temp_output_data.data();
    }
    
    // Copy output data
    cuda_status = cudaMemcpyAsync(output_data_ptr, bindings_[detection_output_index_], 
                               output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("Failed to copy output data from device: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("CUDA stream synchronization failed after data copy: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    // Cache current frame as previous frame for next time
    cached_previous_frame_ = current_frame.clone();
    has_cached_frame_ = true;
    
    // 🔥 Global postprocess time statistics start
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    
    // According to postprocess mode, select CPU or GPU postprocess
    std::vector<Detection> detections;
    
    if (postprocess_mode_ == PostprocessMode::GPU) {
        try {
            // Use GPU postprocess
            detections = gpu::parseYOLOOutputGPU(
                output_data_ptr, output_size_, current_frame, conf_threshold);
        } catch (const std::exception& e) {
            LOG_ERROR("GPU postprocess failed, falling back to CPU: " << e.what());
            detections = parseYOLOOutput(output_data_ptr, output_size_, current_frame, conf_threshold);
        }
    } else {
        // Use CPU postprocess
        detections = parseYOLOOutput(output_data_ptr, output_size_, current_frame, conf_threshold);
    }
    
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    updateGlobalPostprocessTime(postprocess_time);
    
    // 🔥 Global total time statistics
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    updateGlobalTotalTime(total_time);
    
    return detections;
}

// 🔥 New: optimized version with frame number, supporting preprocess cache reuse
std::vector<Detection> TensorRTGlobalInference::detectWithPreviousFrame(const cv::Mat& current_frame, const cv::Mat& previous_frame, float conf_threshold, int frame_id) {
    if (!bindings_initialized_) {
        LOG_ERROR("Error: bindings not initialized");
        return {};
    }
    
    // 🔥 Global inference time statistics start
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Check if it is a continuous frame (can reuse previous frame preprocess result)
    bool is_continuous_frame = has_cached_preprocessed_ && (frame_id == last_processed_frame_id_ + 1);
    
    if (frame_id % 50 == 0) {  // Print debug information every 50 frames
        LOG_INFO("🔥 [Frame continuity check] Frame " << frame_id << ", previous frame " << last_processed_frame_id_ 
                 << ", continuous: " << (is_continuous_frame ? "yes" : "no") 
                 << ", has cache: " << (has_cached_preprocessed_ ? "yes" : "no"));
    }
    
    // Check if engine uses dynamic shapes (reuse original logic)
    bool has_dynamic_shapes = false;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (engine_->bindingIsInput(i)) {
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    has_dynamic_shapes = true;
                    break;
                }
            }
            if (has_dynamic_shapes) break;
        }
    }
    
    // If there are dynamic shapes, set input dimensions (reuse original logic)
    if (has_dynamic_shapes) {
        int nb_bindings = engine_->getNbBindings();
        
        for (int i = 0; i < nb_bindings; ++i) {
            if (engine_->bindingIsInput(i)) {
                nvinfer1::Dims dims = engine_->getBindingDimensions(i);
                bool has_dynamic_dim = false;
                
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] == -1) {
                        has_dynamic_dim = true;
                        break;
                    }
                }
                
                if (has_dynamic_dim) {
                    if (i == current_frame_index_ || i == previous_frame_index_) {
                        dims.d[0] = 1;  // Batch size
                        dims.d[1] = 3;  // Channels
                        dims.d[2] = kInputH;  // Height
                        dims.d[3] = kInputW;  // Width
                    } else {
                        for (int j = 0; j < dims.nbDims; ++j) {
                            if (dims.d[j] == -1) {
                                dims.d[j] = 1;  // Default set to 1
                            }
                        }
                    }
                    
                    if (!context_->setBindingDimensions(i, dims)) {
                        LOG_ERROR("Failed to set dimensions for binding " << i << " (" << engine_->getBindingName(i) << ")");
                        return {};
                    }
                }
            }
        }
        
        if (!context_->allInputDimensionsSpecified()) {
            LOG_ERROR("Error: not all input dimensions are specified");
            return {};
        }

        // Update output binding memory (reuse original logic, simplify)
        for (int i = 0; i < nb_bindings; ++i) {
            if (!engine_->bindingIsInput(i) && i == detection_output_index_) {
                nvinfer1::Dims dims = context_->getBindingDimensions(i);
                size_t new_size = 1;
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] > 0) {
                        new_size *= dims.d[j];
                    }
                }
                
                if (new_size > output_size_) {
                    if (bindings_[i]) {
                        cudaFree(bindings_[i]);
                    }
                    size_t mem_size = new_size * sizeof(float);
                    cudaError_t status = cudaMalloc(&bindings_[i], mem_size);
                    if (status != cudaSuccess) {
                        LOG_ERROR("Failed to reallocate memory for output binding: " << cudaGetErrorString(status));
                        return {};
                    }
                    output_size_ = new_size;
                    
                    if (use_pinned_memory_ && pinned_output_size_ < output_size_) {
                        if (!allocatePinnedMemory()) {
                            LOG_WARNING("Warning: failed to reallocate pinned memory, using regular memory");
                            use_pinned_memory_ = false;
                        }
                    }
                }
            }
        }
    }

    // Ensure input images are valid
    if (current_frame.empty() || previous_frame.empty()) {
        LOG_ERROR("Error: input images are empty");
        return {};
    }
    
    // 🔥 Global preprocess time statistics start
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    size_t frame_preprocess_size = input_size_ * sizeof(float);
    
    // According to preprocess mode and frame continuity, select different preprocess strategies
    try {
        // Always process current frame
        switch (preprocess_mode_) {
            case 0: // CPU full preprocess
                preprocessImageCPU(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
                break;
            case 1: // CV affine transform + GPU preprocess
                preprocessImageCVAffine(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
                break;
            case 2: // Full GPU preprocess
                preprocessImageGPU(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
                break;
            default:
                LOG_ERROR("Unknown preprocess mode: " << preprocess_mode_ << ", using CPU preprocess");
                preprocessImageCPU(current_frame, static_cast<float*>(bindings_[current_frame_index_]));
                break;
        }
        
        // Process previous frame: if it is a continuous frame and has cache, reuse it; otherwise reprocess
        if (is_continuous_frame && cached_previous_preprocessed_) {
            // 🔥 Reuse cached previous frame preprocess result
            cudaError_t status = cudaMemcpyAsync(
                bindings_[previous_frame_index_], 
                cached_previous_preprocessed_, 
                frame_preprocess_size, 
                cudaMemcpyDeviceToDevice, 
                stream_
            );
            if (status != cudaSuccess) {
                LOG_ERROR("Failed to reuse previous frame preprocess cache: " << cudaGetErrorString(status));
                // Fall back to reprocess
                is_continuous_frame = false;
                         } else {
                 cache_hit_count_++;  // 🔥 Update cache hit statistics
                 if (frame_id % 50 == 0) {
                     LOG_INFO("✅ [Preprocess optimization] Frame " << frame_id << ": reuse previous frame preprocess cache, save preprocess time");
                 }
             }
        }
        
        if (!is_continuous_frame || !cached_previous_preprocessed_) {
            cache_miss_count_++;  // 🔥 Update cache miss statistics
            // Reprocess previous frame
            switch (preprocess_mode_) {
                case 0:
                    preprocessImageCPU(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
                    break;
                case 1:
                    preprocessImageCVAffine(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
                    break;
                case 2:
                    preprocessImageGPU(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
                    break;
                default:
                    preprocessImageCPU(previous_frame, static_cast<float*>(bindings_[previous_frame_index_]));
                    break;
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception occurred during preprocess: " << e.what());
        return {};
    }
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    updateGlobalPreprocessTime(preprocess_time);
    
    // 🔥 Cache current frame preprocess result for next time
    if (!cached_previous_preprocessed_ || cached_preprocess_size_ < frame_preprocess_size) {
        // Reallocate cache
        if (cached_previous_preprocessed_) {
            cudaFree(cached_previous_preprocessed_);
        }
        cudaError_t status = cudaMalloc(&cached_previous_preprocessed_, frame_preprocess_size);
        if (status == cudaSuccess) {
            cached_preprocess_size_ = frame_preprocess_size;
        } else {
            LOG_WARNING("Failed to allocate preprocess cache: " << cudaGetErrorString(status));
            cached_previous_preprocessed_ = nullptr;
            cached_preprocess_size_ = 0;
        }
    }
    
    if (cached_previous_preprocessed_) {
        // Copy current frame preprocess result to cache
        cudaError_t status = cudaMemcpyAsync(
            cached_previous_preprocessed_, 
            bindings_[current_frame_index_], 
            frame_preprocess_size, 
            cudaMemcpyDeviceToDevice, 
            stream_
        );
        if (status == cudaSuccess) {
            has_cached_preprocessed_ = true;
            last_processed_frame_id_ = frame_id;
        } else {
            LOG_WARNING("Failed to cache current frame preprocess result: " << cudaGetErrorString(status));
        }
    }
    
    // Check CUDA errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("CUDA error after preprocess: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    // Ensure CUDA device is synchronized
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("CUDA stream synchronization failed: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    // 🔥 Inference execution (reuse original inference and postprocess logic)
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    bool success = context_->enqueueV2(bindings_.data(), stream_, nullptr);
    if (!success) {
        LOG_ERROR("❌ Inference execution failed");
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            LOG_ERROR("CUDA error: " << cudaGetErrorString(cuda_status));
        }
        return {};
    }
    
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("Inference after CUDA stream synchronization failed: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    auto inference_end = std::chrono::high_resolution_clock::now();
    double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    updateGlobalInferenceTime(inference_time);
    
    // Use pinned memory or regular vector to get output data (reuse original logic)
    float* output_data_ptr = nullptr;
    std::vector<float> temp_output_data;
    
    if (use_pinned_memory_ && host_pinned_output_buffer_) {
        if (pinned_output_size_ < output_size_) {
            if (!allocatePinnedMemory()) {
                LOG_WARNING("Warning: failed to reallocate pinned memory, using regular memory");
                use_pinned_memory_ = false;
                temp_output_data.resize(output_size_);
                output_data_ptr = temp_output_data.data();
            } else {
                output_data_ptr = host_pinned_output_buffer_;
            }
        } else {
            output_data_ptr = host_pinned_output_buffer_;
        }
    } else {
        temp_output_data.resize(output_size_);
        output_data_ptr = temp_output_data.data();
    }
    
    // Copy output data
    cuda_status = cudaMemcpyAsync(output_data_ptr, bindings_[detection_output_index_], 
                               output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("Failed to copy output data from device: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    cuda_status = cudaStreamSynchronize(stream_);
    if (cuda_status != cudaSuccess) {
        LOG_ERROR("CUDA stream synchronization failed after data copy: " << cudaGetErrorString(cuda_status));
        return {};
    }
    
    // Cache current frame as previous frame for next time (keep original cache logic)
    cached_previous_frame_ = current_frame.clone();
    has_cached_frame_ = true;
    
    // 🔥 Global postprocess time statistics start
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    
    // According to postprocess mode, select CPU or GPU postprocess
    std::vector<Detection> detections;
    
    if (postprocess_mode_ == PostprocessMode::GPU) {
        try {
            detections = gpu::parseYOLOOutputGPU(
                output_data_ptr, output_size_, current_frame, conf_threshold);
        } catch (const std::exception& e) {
            LOG_ERROR("GPU postprocess failed, fall back to CPU: " << e.what());
            detections = parseYOLOOutput(output_data_ptr, output_size_, current_frame, conf_threshold);
        }
    } else {
        detections = parseYOLOOutput(output_data_ptr, output_size_, current_frame, conf_threshold);
    }
    
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    updateGlobalPostprocessTime(postprocess_time);
    
    // 🔥 Global total time statistics
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    updateGlobalTotalTime(total_time);
    
    return detections;
}

// Preprocess related implementation (can reuse TensorRTInference.cpp implementation)
void TensorRTGlobalInference::preprocessImageCPU(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat frame = image.clone();
    process_input_cpu(frame, input_device_buffer);
}
void TensorRTGlobalInference::preprocessImageCVAffine(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat frame = image.clone();
    process_input_cv_affine(frame, input_device_buffer);
}
void TensorRTGlobalInference::preprocessImageGPU(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat frame = image.clone();
    process_input_gpu(frame, input_device_buffer);
}
cv::Mat TensorRTGlobalInference::letterbox(const cv::Mat& src) {
    float scale = std::min(kInputH / (float)src.rows, kInputW / (float)src.cols);
    int offsetx = (kInputW - src.cols * scale) / 2;
    int offsety = (kInputH - src.rows * scale) / 2;
    cv::Point2f srcTri[3];
    srcTri[0] = cv::Point2f(0.f, 0.f);
    srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
    cv::Point2f dstTri[3];
    dstTri[0] = cv::Point2f(offsetx, offsety);
    dstTri[1] = cv::Point2f(src.cols * scale - 1.f + offsetx, offsety);
    dstTri[2] = cv::Point2f(offsetx, src.rows * scale - 1.f + offsety);
    cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
    cv::Mat warp_dst = cv::Mat::zeros(kInputH, kInputW, src.type());
    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());
    return warp_dst;
}
void TensorRTGlobalInference::preprocessImage(const cv::Mat& image, float* input_data) {
    float scale = std::min(float(kInputW) / image.cols, float(kInputH) / image.rows);
    int new_unpad_w = static_cast<int>(image.cols * scale);
    int new_unpad_h = static_cast<int>(image.rows * scale);
    int dw = kInputW - new_unpad_w;
    int dh = kInputH - new_unpad_h;
    dw /= 2;
    dh /= 2;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, dh, dw, dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (padded.size() != cv::Size(kInputW, kInputH)) {
        cv::resize(padded, padded, cv::Size(kInputW, kInputH));
    }
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    int channel_size = kInputW * kInputH;
    std::memcpy(input_data + 0 * channel_size, channels[2].data, channel_size * sizeof(float));
    std::memcpy(input_data + 1 * channel_size, channels[1].data, channel_size * sizeof(float));
    std::memcpy(input_data + 2 * channel_size, channels[0].data, channel_size * sizeof(float));
}
float TensorRTGlobalInference::clamp(const float val, const float minVal, const float maxVal) {
    return std::min(maxVal, std::max(minVal, val));
}
cv::Rect TensorRTGlobalInference::get_rect(const cv::Mat& img, float bbox[4]) {
    float scale = std::min(kInputH / float(img.rows), kInputW / float(img.cols));
    int offsetx = (kInputW - img.cols * scale) / 2; 
    int offsety = (kInputH - img.rows * scale) / 2; 
    float x1 = (bbox[0] - offsetx) / scale;
    float y1 = (bbox[1] - offsety) / scale;
    float x2 = (bbox[2] - offsetx) / scale;
    float y2 = (bbox[3] - offsety) / scale;
    x1 = clamp(x1, 0, img.cols);
    y1 = clamp(y1, 0, img.rows);
    x2 = clamp(x2, 0, img.cols);
    y2 = clamp(y2, 0, img.rows);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}
std::vector<Detection> TensorRTGlobalInference::parseYOLOOutput(float* output, int output_size, const cv::Mat& original_image, float conf_threshold) {
    std::vector<Detection> detections;
    
    // Check if output data is valid
    if (output == nullptr) {
        LOG_ERROR("Error: output data is empty");
        return detections;
    }
    
    // Check if output size is reasonable
    if (output_size <= 0) {
        LOG_ERROR("Error: output size is unreasonable: " << output_size);
        return detections;
    }
    
    // Get correct dimensions of output from context
    nvinfer1::Dims output_dims = context_->getTensorShape(kGlobalOutputTensorName);
    
    // Remove parsing output data log
    
    // Determine output format and parameters based on actual dimensions
    int batch_size = (output_dims.nbDims > 0 && output_dims.d[0] > 0) ? output_dims.d[0] : 1;
    int num_dimensions = output_dims.nbDims;
    
    // Process YOLOv5 format [batch, 5, 34000]
    if (num_dimensions == 3 && output_dims.d[1] == 5) {
        int boxes_dim = output_dims.d[2];  // Number of boxes per channel
        
        // Iterate through all detection boxes
        for (int i = 0; i < boxes_dim; i++) {
            float x = output[0 * boxes_dim + i];  // x coordinate
            float y = output[1 * boxes_dim + i];  // y coordinate
            float w = output[2 * boxes_dim + i];  // width
            float h = output[3 * boxes_dim + i];  // height
            float conf = output[4 * boxes_dim + i];  // confidence
            
            // Filter low confidence and unreasonable boxes
            if (conf < conf_threshold || conf > 1.0f) continue;
            if (w <= 0 || h <= 0 || w > kInputW * 2 || h > kInputH * 2) continue;
            
            // Calculate scale and offset
            float scale = std::min(float(kInputW) / original_image.cols, float(kInputH) / original_image.rows);
            int offsetx = (kInputW - original_image.cols * scale) / 2;
            int offsety = (kInputH - original_image.rows * scale) / 2;
            
            // Convert detection box coordinates back to original image coordinates
            float x1 = (x - w/2 - offsetx) / scale;
            float y1 = (y - h/2 - offsety) / scale;
            float x2 = (x + w/2 - offsetx) / scale;
            float y2 = (y + h/2 - offsety) / scale;
            
            // Ensure coordinates are within image range
            x1 = clamp(x1, 0, original_image.cols);
            y1 = clamp(y1, 0, original_image.rows);
            x2 = clamp(x2, 0, original_image.cols);
            y2 = clamp(y2, 0, original_image.rows);
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            // Check if final box size is reasonable
            if (width > 5 && height > 5) {
                Detection det;
                det.bbox = cv::Rect2f(x1, y1, width, height);
                det.confidence = conf;
                det.class_id = 0;
                det.is_from_global_model = true;  // Mark as global model detection result
                detections.push_back(det);
            }
        }
        
        // Remove YOLOv5 format detection result log
    }
    // YOLOv8 output format processing
    else if (num_dimensions == 3 && output_dims.d[1] > 5) {
        // YOLOv8 output format is [batch, num_classes+5, num_boxes]
        int num_classes = output_dims.d[1] - 5;
        int num_boxes = output_dims.d[2];
        
        // Remove YOLOv8 format output detection log
    
        // Iterate through all detection boxes
        for (int i = 0; i < num_boxes; i++) {
            float x = output[0 * num_boxes + i];
            float y = output[1 * num_boxes + i];
            float w = output[2 * num_boxes + i];
            float h = output[3 * num_boxes + i];
            float obj_conf = output[4 * num_boxes + i];
            
            if (obj_conf < conf_threshold) continue;
            
            // Find maximum class confidence
            int class_id = 0;
            float max_class_conf = 0;
            for (int c = 0; c < num_classes; c++) {
                float class_conf = output[(5 + c) * num_boxes + i];
                if (class_conf > max_class_conf) {
                    max_class_conf = class_conf;
                    class_id = c;
                }
            }
        
            // Calculate final confidence
            float conf = obj_conf * max_class_conf;
            if (conf < conf_threshold) continue;
        
        // Check if box size is reasonable
            if (w <= 0 || h <= 0 || w > kInputW * 2 || h > kInputH * 2) continue;
            
            // Calculate scale and offset
            float scale = std::min(float(kInputW) / original_image.cols, float(kInputH) / original_image.rows);
            int offsetx = (kInputW - original_image.cols * scale) / 2;
            int offsety = (kInputH - original_image.rows * scale) / 2;
            
            // Convert detection box coordinates back to original image coordinates
            float x1 = (x - w/2 - offsetx) / scale;
            float y1 = (y - h/2 - offsety) / scale;
            float x2 = (x + w/2 - offsetx) / scale;
            float y2 = (y + h/2 - offsety) / scale;
            
            // Ensure coordinates are within image range
            x1 = clamp(x1, 0, original_image.cols);
            y1 = clamp(y1, 0, original_image.rows);
            x2 = clamp(x2, 0, original_image.cols);
            y2 = clamp(y2, 0, original_image.rows);
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            // Check if final box size is reasonable
            if (width > 5 && height > 5) {
                Detection det;
                det.bbox = cv::Rect2f(x1, y1, width, height);
                det.confidence = conf;
                det.class_id = class_id;
                det.is_from_global_model = true;  // Mark as global model detection result
                detections.push_back(det);
            }
        }
        
        // Remove YOLOv8 format detection result log
    }
    // Unknown output format, try default processing
    else {
        // Assume output is [batch, num_boxes, 5+num_classes] format
        int box_size = 5;  // Default value: x,y,w,h,obj_conf
        int num_boxes = output_size / box_size;  // Estimated number of boxes
        
        int valid_detections = 0;
        for (int i = 0; i < num_boxes; i++) {
            float* box_data = output + i * box_size;
            float x = box_data[0];
            float y = box_data[1];
            float w = box_data[2];
            float h = box_data[3];
            float conf = box_data[4];
            
            // Filter low confidence and unreasonable boxes
            if (conf < conf_threshold || conf > 1.0f) continue;
        if (w <= 0 || h <= 0 || w > kInputW * 2 || h > kInputH * 2) continue;
        
        // Calculate scale and offset
        float scale = std::min(float(kInputW) / original_image.cols, float(kInputH) / original_image.rows);
        int offsetx = (kInputW - original_image.cols * scale) / 2;
        int offsety = (kInputH - original_image.rows * scale) / 2;
        
        // Convert detection box coordinates back to original image coordinates
        float x1 = (x - w/2 - offsetx) / scale;
        float y1 = (y - h/2 - offsety) / scale;
        float x2 = (x + w/2 - offsetx) / scale;
        float y2 = (y + h/2 - offsety) / scale;
        
        // Ensure coordinates are within image range
        x1 = clamp(x1, 0, original_image.cols);
        y1 = clamp(y1, 0, original_image.rows);
        x2 = clamp(x2, 0, original_image.cols);
        y2 = clamp(y2, 0, original_image.rows);
        
        float width = x2 - x1;
        float height = y2 - y1;
        
        // Check if final box size is reasonable
        if (width > 5 && height > 5) {
            Detection det;
            det.bbox = cv::Rect2f(x1, y1, width, height);
            det.confidence = conf;
            det.class_id = 0;
            det.is_from_global_model = true;  // Mark as global model detection result
            detections.push_back(det);
            valid_detections++;
        }
    }
    
        // Remove default processing detection result log
    }
    
    // Apply NMS - use configuration threshold instead of hardcode
    if (!detections.empty()) {
        float used_nms = static_cast<float>(FLAGS_global_nms_threshold);
        used_nms = std::max(0.3f, std::min(0.7f, used_nms));
        nms(detections, nullptr, conf_threshold, used_nms);
    }
    
    return detections;
}
float TensorRTGlobalInference::iou(float lbox[4], float rbox[4]) {
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
void TensorRTGlobalInference::nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    std::sort(res.begin(), res.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<Detection> nms_result;
    for (size_t i = 0; i < res.size(); ++i) {
        bool keep = true;
        for (size_t j = 0; j < nms_result.size(); ++j) {
            float lbox[4] = {res[i].bbox.x, res[i].bbox.y, res[i].bbox.x + res[i].bbox.width, res[i].bbox.y + res[i].bbox.height};
            float rbox[4] = {nms_result[j].bbox.x, nms_result[j].bbox.y, nms_result[j].bbox.x + nms_result[j].bbox.width, nms_result[j].bbox.y + nms_result[j].bbox.height};
            if (iou(lbox, rbox) > nms_thresh) {
                keep = false;
                break;
            }
        }
        if (keep) nms_result.push_back(res[i]);
    }
    res = std::move(nms_result);
}

// 🔥 Global inference time statistics method implementation
void TensorRTGlobalInference::updateGlobalPreprocessTime(double preprocess_time) {
    total_preprocess_time_ += preprocess_time;
    preprocess_count_++;
    avg_preprocess_time_ = total_preprocess_time_ / preprocess_count_;
    max_preprocess_time_ = std::max(max_preprocess_time_, preprocess_time);
    min_preprocess_time_ = std::min(min_preprocess_time_, preprocess_time);
    
    // Output statistics every 10 times
    if (preprocess_count_ % 10 == 0) {
        LOG_INFO("🔥 Global preprocess time statistics triggered, count: " << preprocess_count_);
        printGlobalTimeStatistics();
    }
}

void TensorRTGlobalInference::updateGlobalInferenceTime(double inference_time) {
    total_inference_time_ += inference_time;
    inference_count_++;
    avg_inference_time_ = total_inference_time_ / inference_count_;
    max_inference_time_ = std::max(max_inference_time_, inference_time);
    min_inference_time_ = std::min(min_inference_time_, inference_time);
}

void TensorRTGlobalInference::updateGlobalPostprocessTime(double postprocess_time) {
    total_postprocess_time_ += postprocess_time;
    postprocess_count_++;
    avg_postprocess_time_ = total_postprocess_time_ / postprocess_count_;
    max_postprocess_time_ = std::max(max_postprocess_time_, postprocess_time);
    min_postprocess_time_ = std::min(min_postprocess_time_, postprocess_time);
}

void TensorRTGlobalInference::updateGlobalTotalTime(double total_time) {
    total_processing_time_ += total_time;
    avg_total_time_ = total_processing_time_ / inference_count_;
    max_total_time_ = std::max(max_total_time_, total_time);
    min_total_time_ = std::min(min_total_time_, total_time);
}

void TensorRTGlobalInference::printGlobalTimeStatistics() {
    LOG_INFO("=== Global inference time statistics report ===");
    LOG_INFO("Preprocess time statistics:");
    LOG_INFO("  Total preprocess time: " << std::fixed << std::setprecision(2) << total_preprocess_time_ << "ms");
    LOG_INFO("  Average preprocess time: " << std::fixed << std::setprecision(2) << avg_preprocess_time_ << "ms");
    LOG_INFO("  Max preprocess time: " << std::fixed << std::setprecision(2) << max_preprocess_time_ << "ms");
    LOG_INFO("  Min preprocess time: " << std::fixed << std::setprecision(2) << min_preprocess_time_ << "ms");
    LOG_INFO("  Preprocess count: " << preprocess_count_);
    
    LOG_INFO("Inference time statistics:");
    LOG_INFO("  Total inference time: " << std::fixed << std::setprecision(2) << total_inference_time_ << "ms");
    LOG_INFO("  Average inference time: " << std::fixed << std::setprecision(2) << avg_inference_time_ << "ms");
    LOG_INFO("  Max inference time: " << std::fixed << std::setprecision(2) << max_inference_time_ << "ms");
    LOG_INFO("  Min inference time: " << std::fixed << std::setprecision(2) << min_inference_time_ << "ms");
    LOG_INFO("  Inference count: " << inference_count_);
    
    LOG_INFO("Postprocess time statistics:");
    LOG_INFO("  Total postprocess time: " << std::fixed << std::setprecision(2) << total_postprocess_time_ << "ms");
    LOG_INFO("  Average postprocess time: " << std::fixed << std::setprecision(2) << avg_postprocess_time_ << "ms");
    LOG_INFO("  Max postprocess time: " << std::fixed << std::setprecision(2) << max_postprocess_time_ << "ms");
    LOG_INFO("  Min postprocess time: " << std::fixed << std::setprecision(2) << min_postprocess_time_ << "ms");
    LOG_INFO("  Postprocess count: " << postprocess_count_);
    
    LOG_INFO("Total processing time statistics:");
    LOG_INFO("  Total processing time: " << std::fixed << std::setprecision(2) << total_processing_time_ << "ms");
    LOG_INFO("  Average total time: " << std::fixed << std::setprecision(2) << avg_total_time_ << "ms");
    LOG_INFO("  Max total time: " << std::fixed << std::setprecision(2) << max_total_time_ << "ms");
    LOG_INFO("  Min total time: " << std::fixed << std::setprecision(2) << min_total_time_ << "ms");
    
    // Calculate the proportion of each part of the time
    if (total_processing_time_ > 0) {
        double preprocess_ratio = (total_preprocess_time_ / total_processing_time_) * 100.0;
        double inference_ratio = (total_inference_time_ / total_processing_time_) * 100.0;
        double postprocess_ratio = (total_postprocess_time_ / total_processing_time_) * 100.0;
        double other_ratio = 100.0 - preprocess_ratio - inference_ratio - postprocess_ratio;
        
        LOG_INFO("Global inference time proportion analysis:");
        LOG_INFO("  Preprocess time proportion: " << std::fixed << std::setprecision(1) << preprocess_ratio << "%");
        LOG_INFO("  Inference time proportion: " << std::fixed << std::setprecision(1) << inference_ratio << "%");
        LOG_INFO("  Postprocess time proportion: " << std::fixed << std::setprecision(1) << postprocess_ratio << "%");
        LOG_INFO("  Other time proportion: " << std::fixed << std::setprecision(1) << other_ratio << "%");
    }
    
    // 🔥 Preprocess optimization effect statistics
    int total_cache_attempts = cache_hit_count_ + cache_miss_count_;
    if (total_cache_attempts > 0) {
        double cache_hit_rate = (double)cache_hit_count_ / total_cache_attempts * 100.0;
        LOG_INFO("🚀 Preprocess cache optimization statistics:");
        LOG_INFO("  Cache hit count: " << cache_hit_count_);
        LOG_INFO("  Cache miss count: " << cache_miss_count_);
        LOG_INFO("  Cache hit rate: " << std::fixed << std::setprecision(1) << cache_hit_rate << "%");
        if (avg_preprocess_time_ > 0) {
            double time_saved = (cache_hit_rate / 100.0) * (avg_preprocess_time_ / 2.0);  // Save about half of the preprocess time
            LOG_INFO("  Preprocess optimization saved: ~" << std::fixed << std::setprecision(2) 
                     << time_saved << "ms/frame (about" 
                     << std::fixed << std::setprecision(1) << (time_saved / avg_preprocess_time_ * 100.0) << "%)");
        }
    }
    
    LOG_INFO("==================");
}

} // namespace tracking