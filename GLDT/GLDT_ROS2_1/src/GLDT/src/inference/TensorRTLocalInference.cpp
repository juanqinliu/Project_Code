#include "inference/TensorRTLocalInference.h"
#include "common/Detection.h" 
#include "inference/preprocess.h"
#include "common/Logger.h"    
#include "common/Flags.h" 
#include <chrono>     
#include <iomanip>  

#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstring>

// Add CUDA error check macro definition
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

// Define tensor name
const char* kLocalInputTensorName = "images";
const char* kLocalOutputTensorName = "output0"; 


// Get local preprocess mode from Flags
int TensorRTLocalInference::getLocalPreprocessMode() {
    return FLAGS_local_preprocess_mode;
}

// Get local postprocess mode from Flags
PostprocessMode TensorRTLocalInference::getLocalPostprocessMode() {
    return static_cast<PostprocessMode>(FLAGS_local_postprocess_mode);
}

// Batch preprocess mode related functions have been removed

// Add default constructor, get preprocess mode from Flags
TensorRTLocalInference::TensorRTLocalInference(const std::string& engine_path) 
    : stream_(nullptr), cuda_initialized_(false), bindings_initialized_(false),
      inference_count_(0), preprocess_count_(0), postprocess_count_(0) {
    // Verify preprocess mode
    preprocess_mode_ = getLocalPreprocessMode();
    if (preprocess_mode_ < 0 || preprocess_mode_ > 2) {
        preprocess_mode_ = 0;
    }
    // Initialize preprocess and batch processing context
    preprocess_ctx_ = std::make_unique<PreprocessContext>();
    batch_ctx_ = std::make_unique<BatchMemoryContext>();
    multi_stream_ctx_ = std::make_unique<MultiStreamPreprocessContext>();  // 🔥 初始化多stream上下文
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
    // Create runtime and engine
    static LocalLogger logger;
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
    
    // 🔥 Check number of optimization profiles
    int num_profiles = engine_->getNbOptimizationProfiles();
    LOG_INFO("✅ Number of optimization profiles: " << num_profiles);
    
    // 🔥 For INT8 models with multiple profiles, select the runtime profile (usually profile 1)
    // Profile 0 is for calibration, Profile 1 is for runtime inference
    if (num_profiles > 1) {
        // Set execution context to use profile 1 (runtime profile)
        context_->setOptimizationProfileAsync(1, nullptr);
        cudaDeviceSynchronize();
        LOG_INFO("✅ Selected optimization profile 1 for runtime inference");
    } else if (num_profiles == 1) {
        // FP16/FP32 models typically have only one profile
        context_->setOptimizationProfileAsync(0, nullptr);
        cudaDeviceSynchronize();
        LOG_INFO("✅ Selected optimization profile 0 (single profile mode)");
    }
    
    // Initialize binding
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
    // Initialize binding
    if (!initializeBindings()) {
        LOG_ERROR("Failed to initialize bindings");
        throw std::runtime_error("Failed to initialize bindings");
    }
    
    // 🔥 Store the active profile index for later use (reuse num_profiles from above)
    active_profile_index_ = (num_profiles > 1) ? 1 : 0;
    
    // 🔥 Check engine batch configuration - handle API deprecation warning
    int engine_max_batch_size = 1;
    #if NV_TENSORRT_MAJOR >= 8
    // getMaxBatchSize() in TensorRT 8+ has been deprecated, but still can be used for old engines
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    engine_max_batch_size = engine_->getMaxBatchSize();
    #pragma GCC diagnostic pop
    #else
    engine_max_batch_size = engine_->getMaxBatchSize();
    #endif
    
    // Check if dynamic shape is supported
    bool has_dynamic_shapes = false;
    
    // Get and verify input and output binding
    int input_index = -1;
    int output_index = -1;
    
    for (int i = 0; i < nb_bindings; ++i) {
        const char* name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        bool is_input = engine_->bindingIsInput(i);
        
        // Check dynamic dimension
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                has_dynamic_shapes = true;
                break;
            }
        }
        
        if (is_input) {
            input_index = i;
            // Verify input dimension (just check, don't store unused variables)
            if (dims.nbDims == 4) {
                // Input dimensions are: [batch, channels, height, width]
                // We verify but don't need to store them as separate variables
            }
        } else {
            output_index = i;
            // Calculate output size
            output_size_ = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] > 0) {  // Ignore dynamic dimension
                    output_size_ *= dims.d[j];
                }
            }
        }
    }
    
    // Save detection output binding index (last output binding)
    detection_output_index_ = output_index;

    if (input_index == -1 || detection_output_index_ == -1) {
        LOG_ERROR("Cannot find valid input or output binding");
        throw std::runtime_error("Cannot find valid input or output binding");
    }
    
    // Calculate model's num_boxes value (YOLO output format is [box_info_size, num_boxes])
    // box_info_size usually is 5 (x,y,w,h,conf) or more (contains category information)
    if (output_size_ > 0) {
        LOG_INFO("✅ [Model analysis] Output size: " << output_size_);
        // Use constant defined box_info_size
        // Calculate num_boxes and store as member variable
        num_boxes_ = output_size_ / kBoxInfoSize;
        LOG_INFO("✅ [Model analysis] Estimated detection box number: " << num_boxes_ 
                << " (Output size " << output_size_ << " / Box info size " << kBoxInfoSize << ")");
    }
    
    // Set input size
    input_dims_ = cv::Size(kInputW, kInputH);
    input_size_ = 3 * kInputH * kInputW; // C*H*W
    
    // 🔥 Remove duplicate memory allocation, handle in initializeBindings()
    // Only create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Initialize CUDA preprocess according to preprocess mode
    if (preprocess_mode_ == 1 || preprocess_mode_ == 2) {
        // Initialize CUDA preprocess (mode 1 and 2 need)
        const int kMaxInputImageSize = 4096 * 3112;
        cuda_preprocess_init(kMaxInputImageSize);
        cuda_initialized_ = true;
    }
    
    // Set postprocess mode (only through gflags)
    postprocess_mode_ = getLocalPostprocessMode();
    LOG_INFO("Local postprocess mode: " << (postprocess_mode_ == PostprocessMode::CPU ? "CPU" : "GPU"));
}

TensorRTLocalInference::~TensorRTLocalInference() {
    // 🔥 Ensure all CUDA operations are completed
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
    cudaDeviceSynchronize();
    
    // Clean up binding memory
    for (void* binding : bindings_) {
        if (binding) {
            cudaFree(binding);
        }
    }
    bindings_.clear();
    
    // Clean up CUDA stream
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    // Clean up CUDA preprocess resources
    if (cuda_initialized_) {
        cuda_preprocess_destroy();
        cuda_initialized_ = false;
    }
    
    // Preprocess and batch processing context will be automatically cleaned (smart pointer destructor)
    // But ensure they are correctly reset
    preprocess_ctx_.reset();
    batch_ctx_.reset();
    multi_stream_ctx_.reset();  // 🔥 清理多stream上下文
}

std::vector<Detection> TensorRTLocalInference::detect(const cv::Mat& image, float conf_threshold) {
    // 🔥 Local inference time statistics start
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Ensure binding is initialized
    if (!bindings_initialized_ && !initializeBindings()) {
        LOG_ERROR("❌ Binding initialization failed");
        return {};
    }
    
    // Verify all bindings are valid
    bool all_bindings_valid = true;
    for (size_t i = 0; i < bindings_.size(); i++) {
        if (!bindings_[i]) {
            LOG_ERROR("❌ Binding[" << i << "] is empty");
            all_bindings_valid = false;
        }
    }
    
    if (!all_bindings_valid) {
        LOG_ERROR("❌ Invalid binding, trying to reinitialize...");
        bindings_initialized_ = false;
        if (!initializeBindings()) {
            return {};
        }
    }
    
    try {
        // Set dynamic dimension (if needed)
        if (needs_dynamic_setting_) {
            for (int i = 0; i < engine_->getNbBindings(); i++) {
                if (engine_->bindingIsInput(i)) {
                    auto dims = engine_->getBindingDimensions(i);
                    if (dims.d[0] == -1) {
                        nvinfer1::Dims new_dims = dims;
                        new_dims.d[0] = 1;  // Single image inference
                        if (!context_->setBindingDimensions(i, new_dims)) {
                            LOG_ERROR("❌ Set dynamic dimension failed");
                            return {};
                        }
                    }
                }
            }
        }
        
        // 🔥 Local preprocess time statistics start
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        
        // Preprocess and copy input data
        if (preprocess_mode_ == 0) {
            preprocessImageCPU(image, static_cast<float*>(bindings_[0]));
        } else if (preprocess_mode_ == 1) {
            preprocessImageCVAffine(image, static_cast<float*>(bindings_[0]));
        } else if (preprocess_mode_ == 2) {
            preprocessImageGPU(image, static_cast<float*>(bindings_[0]));
        }
        
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
        updateLocalPreprocessTime(preprocess_time);
        
        // 🔥 Local inference time statistics start
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // Execute inference
        bool success = context_->enqueueV2(bindings_.data(), stream_, nullptr);
        if (!success) {
            LOG_ERROR("❌ Inference execution failed");
            return {};
        }
        
        // Wait for completion
        cudaStreamSynchronize(stream_);
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        updateLocalInferenceTime(inference_time);
        
        // Get output - add safety check
        std::vector<float> output_data(output_size_);
        if (bindings_[1] && output_size_ > 0) {
            cudaMemcpyAsync(output_data.data(), bindings_[1], 
                           output_size_ * sizeof(float),
                           cudaMemcpyDeviceToHost, stream_);
            
            cudaStreamSynchronize(stream_);
        } else {
            LOG_ERROR("❌ Output binding invalid or output size is 0");
            return {};
        }
        
        // 🔥 Local postprocess time statistics start
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        
        // Select CPU or GPU postprocess according to postprocess mode
        std::vector<Detection> detections;
        
        if (postprocess_mode_ == PostprocessMode::GPU) {
            try {
                // Use GPU postprocess
                detections = gpu::parseLocalYOLOOutputGPU(
                    output_data.data(), output_size_, image, conf_threshold);
            } catch (const std::exception& e) {
                LOG_ERROR("Local model GPU postprocess failed, fallback to CPU: " << e.what());
                detections = parseYOLOOutput(output_data.data(), output_size_, 
                                         image, conf_threshold);
            }
        } else {
            // Use CPU postprocess
            detections = parseYOLOOutput(output_data.data(), output_size_, 
                             image, conf_threshold);
        }
        
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        updateLocalPostprocessTime(postprocess_time);
        
        // 🔥 Local total time statistics
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        updateLocalTotalTime(total_time);
        
        // Apply NMS - use configuration threshold instead of hardcode
        float used_nms = static_cast<float>(FLAGS_local_nms_threshold);
        used_nms = std::max(0.3f, std::min(0.7f, used_nms));
        nms(detections, nullptr, conf_threshold, used_nms);
        
        // 🔥 Debug: Print LOCAL detection results
        if (Logger::isVLogEnabled(1) && !detections.empty()) {
            LOG_VERBOSE(1, "🎯 [LOCAL] Detected " << detections.size() << " objects:");
            for (size_t i = 0; i < std::min(detections.size(), size_t(5)); i++) {
                const auto& det = detections[i];
                LOG_VERBOSE(1, "  [" << i << "] bbox=[" << det.bbox.x << "," << det.bbox.y 
                          << "," << det.bbox.width << "," << det.bbox.height 
                          << "] conf=" << det.confidence 
                          << " class=" << det.class_id
                          << " from_local=" << det.is_from_local_model);
            }
        }
        
        return detections;
                             
    } catch (const std::exception& e) {
        LOG_ERROR("❌ Inference process failed: " << e.what());
        return {};
    }
}

// Mode 0: Use CPU to do letterbox, normalization, BGR2RGB, NHWC to NCHW
void TensorRTLocalInference::preprocessImageCPU(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat frame = image.clone();
    process_input_cpu(frame, input_device_buffer);
}

// Mode 1: Use CPU to do letterbox, GPU to do normalization, BGR2RGB, NHWC to NCHW
void TensorRTLocalInference::preprocessImageCVAffine(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat frame = image.clone();
    process_input_cv_affine(frame, input_device_buffer);
}

// Mode 2: Use GPU to do all preprocess steps
void TensorRTLocalInference::preprocessImageGPU(const cv::Mat& image, float* input_device_buffer) {
    cv::Mat frame = image.clone();
    process_input_gpu(frame, input_device_buffer);
}

// letterbox helper function (for CPU mode)
cv::Mat TensorRTLocalInference::letterbox(const cv::Mat& src) {
    float scale = std::min(kInputH / (float)src.rows, kInputW / (float)src.cols);

    int offsetx = (kInputW - src.cols * scale) / 2;
    int offsety = (kInputH - src.rows * scale) / 2;

    cv::Point2f srcTri[3]; // Calculate three points of the original image: top-left, top-right, bottom-left
    srcTri[0] = cv::Point2f(0.f, 0.f);
    srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
    srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
    cv::Point2f dstTri[3]; // Calculate three points of the target image: top-left, top-right, bottom-left
    dstTri[0] = cv::Point2f(offsetx, offsety);
    dstTri[1] = cv::Point2f(src.cols * scale - 1.f + offsetx, offsety);
    dstTri[2] = cv::Point2f(offsetx, src.rows * scale - 1.f + offsety);
    cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);       // Calculate affine transformation matrix
    cv::Mat warp_dst = cv::Mat::zeros(kInputH, kInputW, src.type()); // Create target image
    cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());        // Perform affine transformation
    return warp_dst;
}

// Original CPU preprocess method (for compatibility)
void TensorRTLocalInference::preprocessImage(const cv::Mat& image, float* input_data) {
    // Keep aspect ratio resize
    float scale = std::min(float(kInputW) / image.cols, float(kInputH) / image.rows);
    int new_unpad_w = static_cast<int>(image.cols * scale);
    int new_unpad_h = static_cast<int>(image.rows * scale);
    
    // Calculate padding
    int dw = kInputW - new_unpad_w;
    int dh = kInputH - new_unpad_h;
    dw /= 2;
    dh /= 2;
    
    // Resize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));
    
    // Add padding
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, dh, dw, dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // Ensure size is correct
    if (padded.size() != cv::Size(kInputW, kInputH)) {
        cv::resize(padded, padded, cv::Size(kInputW, kInputH));
    }
    
    // Convert to float and normalize to [0,1]
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    
    // Convert BGR to RGB and HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    
    int channel_size = kInputW * kInputH;
    // Note: OpenCV is BGR, convert to RGB
    std::memcpy(input_data + 0 * channel_size, channels[2].data, channel_size * sizeof(float)); // R
    std::memcpy(input_data + 1 * channel_size, channels[1].data, channel_size * sizeof(float)); // G
    std::memcpy(input_data + 2 * channel_size, channels[0].data, channel_size * sizeof(float)); // B
}

float TensorRTLocalInference::clamp(const float val, const float minVal, const float maxVal) {
    return std::min(maxVal, std::max(minVal, val));
}

cv::Rect TensorRTLocalInference::get_rect(const cv::Mat& img, float bbox[4]) {
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

std::vector<Detection> TensorRTLocalInference::parseYOLOOutput(float* output, int output_size, 
                                                         const cv::Mat& original_image, 
                                                         float conf_threshold) {
    std::vector<Detection> detections;
    
    // Use already calculated num_boxes_ member variable, if not initialized then calculate
    int num_boxes = (num_boxes_ > 0) ? num_boxes_ : (output_size / kBoxInfoSize);
    int box_info_size = kBoxInfoSize;
    
    // 🔥 Debug: Check raw output data
    if (Logger::isVLogEnabled(2)) {
        LOG_VERBOSE(2, "🔍 [YOLO Parse] Output size: " << output_size << ", num_boxes: " << num_boxes);
        LOG_VERBOSE(2, "🔍 [YOLO Parse] Image size: " << original_image.cols << "x" << original_image.rows);
        
        // Sample first few boxes
        for (int i = 0; i < std::min(3, num_boxes); i++) {
            float x = output[0 * num_boxes + i];
            float y = output[1 * num_boxes + i];
            float w = output[2 * num_boxes + i];
            float h = output[3 * num_boxes + i];
            float conf = output[4 * num_boxes + i];
            LOG_VERBOSE(2, "  Box[" << i << "] raw: x=" << x << " y=" << y << " w=" << w 
                      << " h=" << h << " conf=" << conf);
        }
    }
    
    // Count valid detections
    int valid_conf_count = 0;
    int valid_bbox_count = 0;
    
    for (int i = 0; i < num_boxes; ++i) {
        float x = output[0 * num_boxes + i];
        float y = output[1 * num_boxes + i];
        float w = output[2 * num_boxes + i];
        float h = output[3 * num_boxes + i];
        float conf = output[4 * num_boxes + i];
        
        // Count confidence distribution
        if (conf > 0.01f && conf <= 1.0f) {
            valid_conf_count++;
        }
        
        // Check confidence
        if (conf < conf_threshold || conf > 1.0f) {
            continue;
        }
        
        // Verify the reasonableness of the bounding box - loosen conditions
        if (w <= 0 || h <= 0 || w > kInputW * 2 || h > kInputH * 2) {
            continue;
        }
        
        valid_bbox_count++;
        
        // Convert coordinates (from network input size to original image size)
        float scale = std::min(float(kInputW) / original_image.cols, 
                              float(kInputH) / original_image.rows);
        int offsetx = (kInputW - original_image.cols * scale) / 2;
        int offsety = (kInputH - original_image.rows * scale) / 2;
        
        // Convert center point coordinates to top-left coordinates
        float x1 = (x - w/2 - offsetx) / scale;
        float y1 = (y - h/2 - offsety) / scale;
        float x2 = (x + w/2 - offsetx) / scale;
        float y2 = (y + h/2 - offsety) / scale;
        
        // Constrain to original image range
        x1 = clamp(x1, 0, original_image.cols);
        y1 = clamp(y1, 0, original_image.rows);
        x2 = clamp(x2, 0, original_image.cols);
        y2 = clamp(y2, 0, original_image.rows);
        
        float width = x2 - x1;
        float height = y2 - y1;
        
        if (width > 5 && height > 5) {  // Minimum size requirement lowered
            Detection det;
            det.bbox = cv::Rect2f(x1, y1, width, height);
            det.confidence = conf;
            det.class_id = 0;
            det.is_from_local_model = true;  // Mark as local model detection result
            
            detections.push_back(det);
            
        }
    }
    
    // Apply NMS - use configuration threshold instead of hardcode
    float used_nms = static_cast<float>(FLAGS_local_nms_threshold);
    used_nms = std::max(0.3f, std::min(0.7f, used_nms));
    nms(detections, nullptr, conf_threshold, used_nms);
    
    // std::cout << "NMS after detection number: " << detections.size() << std::endl;
    return detections;
}

float TensorRTLocalInference::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + 
                        (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS);
}

void TensorRTLocalInference::nms(std::vector<Detection>& res, float* output, 
                           float conf_thresh, float nms_thresh) {
    // Sort by confidence in descending order
    std::sort(res.begin(), res.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    
    std::vector<Detection> nms_result;
    
    for (size_t i = 0; i < res.size(); ++i) {
        bool keep = true;
        
        for (size_t j = 0; j < nms_result.size(); ++j) {
            // Calculate IoU
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
        
        if (keep) {
            nms_result.push_back(res[i]);
        }
    }
    
    res = std::move(nms_result);
}

// 🔥 新增：批量检测实现
std::vector<std::vector<Detection>> TensorRTLocalInference::detectBatch(const std::vector<cv::Mat>& images, float conf_threshold) {
    std::vector<std::vector<Detection>> all_results;
    all_results.reserve(images.size());
    
    if (images.empty()) {
        LOG_WARNING("⚠️ [TensorRT batch] Input image is empty");
        return all_results;
    }

    
    // Get maximum batch size limit
    int max_batch_size = getMaxBatchSize();
    
    // 🔥 使用配置的最大batch size，不再硬编码限制
    int safe_batch_size = std::min(max_batch_size, static_cast<int>(images.size()));
    const int total_images = images.size();
    
    if (FLAGS_log_batch_timing) {
        LOG_INFO("📦 [批量检测] 总ROI数=" << total_images << ", 最大batch=" << max_batch_size << ", 实际batch=" << safe_batch_size);
    }
    
    // Batch processing
    for (int start_idx = 0; start_idx < total_images; start_idx += safe_batch_size) {
        int end_idx = std::min(start_idx + safe_batch_size, total_images);
        int current_batch_size = end_idx - start_idx;
        
        // Create current batch
        std::vector<cv::Mat> current_batch(images.begin() + start_idx, images.begin() + end_idx);
        
        // Execute真正的批量推理
        auto batch_results = executeBatchInference(current_batch, conf_threshold);
        
        // Add results to total results
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }
    
    return all_results;
}

// 🔥 新增：执行真正的批量推理 - 使用实例级内存管理
std::vector<std::vector<Detection>> TensorRTLocalInference::executeBatchInference(const std::vector<cv::Mat>& batch_images, float conf_threshold) {
    const int batch_size = batch_images.size();
    std::vector<std::vector<Detection>> results;
    results.reserve(batch_size);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    
    // 检查批次大小是否有效
    if (batch_size <= 0 || batch_size > getMaxBatchSize()) {
        LOG_WARNING("⚠️ [TensorRT batch inference] Batch size invalid, fallback to single detection");
        
        // Fallback to single detection
        for (const auto& image : batch_images) {
            results.push_back(detect(image, conf_threshold));
        }
        return results;
    }
    
    try {
        // 🔥 Use instance-level memory management, replace static global variable
        if (!initBatchContext()) {
            throw std::runtime_error("Batch context initialization failed");
        }
        
        if (!ensureBatchBuffers(batch_size)) {
            throw std::runtime_error("Batch buffer allocation failed");
        }
        
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        // 🔥 Batch preprocess - use instance-level buffer
        preprocessBatch(batch_images, static_cast<float*>(batch_ctx_->input_buffer));
        cudaStreamSynchronize(stream_); // 确保前处理完成
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
        updateLocalPreprocessTime(preprocess_time);
        
        if (FLAGS_log_batch_timing) {
            LOG_INFO("⏱️  [批量推理时间-前处理] batch=" << batch_size << ", 前处理=" << preprocess_time << "ms");
        }
        
        // 🔥 Create batch binding - use instance-level buffer (full binding)
        int nb_bindings = engine_->getNbBindings();
        std::vector<void*> batch_bindings(nb_bindings, nullptr);
        std::vector<void*> extra_output_buffers; // Additional output buffer

        for (int i = 0; i < nb_bindings; ++i) {
            if (engine_->bindingIsInput(i)) {
                batch_bindings[i] = batch_ctx_->input_buffer;
            } else if (i == detection_output_index_) {
                batch_bindings[i] = batch_ctx_->output_buffer;
            } else {
                // Allocate memory for other outputs temporarily
                auto dims = engine_->getBindingDimensions(i);
                size_t bytes = sizeof(float);
                for (int d = 0; d < dims.nbDims; ++d) {
                    int dim = dims.d[d];
                    if (dim == -1) {
                        dim = (d == 0 ? batch_size : 1);
                    }
                    bytes *= dim;
                }
                void* buffer = nullptr;
                cudaError_t status = cudaMalloc(&buffer, bytes);
                if (status != cudaSuccess) {
                    LOG_ERROR("❌ [TensorRT batch inference] Output binding memory allocation failed: "
                              << cudaGetErrorString(status));
                    for (void* ptr : extra_output_buffers) {
                        if (ptr) cudaFree(ptr);
                    }
                    throw std::runtime_error("batch output malloc failed");
                }
                batch_bindings[i] = buffer;
                extra_output_buffers.push_back(buffer);
            }
        }
        
        // 🔥 For EXPLICIT_BATCH mode, must set input dimensions before inference
        // Get the number of bindings per profile
        int num_profiles = engine_->getNbOptimizationProfiles();
        int bindings_per_profile = nb_bindings / std::max(1, num_profiles);
        
        // Calculate the binding index offset for the active profile
        int binding_offset = active_profile_index_ * bindings_per_profile;
        
        for (int i = 0; i < bindings_per_profile; ++i) {
            int binding_idx = binding_offset + i;
            if (engine_->bindingIsInput(binding_idx)) {
                auto dims = engine_->getBindingDimensions(binding_idx);
                if (dims.d[0] == -1) {
                    nvinfer1::Dims input_dims = dims;
                    input_dims.d[0] = batch_size;
                    bool dim_ok = context_->setBindingDimensions(binding_idx, input_dims);
                    if (!dim_ok) {
                        for (void* ptr : extra_output_buffers) {
                            if (ptr) cudaFree(ptr);
                        }
                        throw std::runtime_error("Set batch inference input dimensions failed");
                    }
                }
            }
        }
        
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // 🔥 Execute batch inference
        bool success = context_->enqueueV2(batch_bindings.data(), stream_, nullptr);
        if (!success) {
            LOG_ERROR("❌ [TensorRT batch inference] Inference execution failed");
            throw std::runtime_error("TensorRT batch inference failed");
        }
        
        cudaStreamSynchronize(stream_); // Ensure inference is complete
        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        updateLocalInferenceTime(inference_time);
        
        if (FLAGS_log_batch_timing) {
            LOG_INFO("⏱️  [批量推理时间-推理] batch=" << batch_size << ", 推理=" << inference_time << "ms");
        }
        
        // 🔥 Immediately release temporary output buffer, avoid memory leak
        for (void* ptr : extra_output_buffers) {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        extra_output_buffers.clear();
        
        auto copy_start = std::chrono::high_resolution_clock::now();
        // 🔥 Copy output data to CPU
        const int total_output_size = batch_size * output_size_;
        std::vector<float> batch_output_host(total_output_size);
        cudaMemcpyAsync(batch_output_host.data(), batch_ctx_->output_buffer, 
                        total_output_size * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream_);
        
        cudaStreamSynchronize(stream_);
        auto copy_end = std::chrono::high_resolution_clock::now();
        double copy_time = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
        
        if (FLAGS_log_batch_timing) {
            LOG_INFO("⏱️  [批量推理时间-拷贝] batch=" << batch_size << ", GPU->CPU拷贝=" << copy_time << "ms");
        }
        
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        
        // Batch postprocess - use CPU or GPU based on the selected postprocess mode
        if (postprocess_mode_ == PostprocessMode::GPU) {
            // Local detection GPU postprocess directly use serial mode, process each sample
            results.resize(batch_size);
            
            try {
                for (int b = 0; b < batch_size; b++) {
                    // Calculate the output pointer and size of the current sample
                    float* output = batch_output_host.data() + b * output_size_;
                    
                    // Use GPU postprocess for single sample
                    results[b] = gpu::parseLocalYOLOOutputGPU(
                        output, output_size_, batch_images[b], conf_threshold);
                        
                    // Ensure CUDA synchronization, avoid memory interference between multiple samples
                    cudaDeviceSynchronize();
                    
                    // Clean up CUDA memory
                    cudaFree(0);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Local model GPU batch postprocess failed, fallback to CPU: " << e.what());
                results = postprocessBatch(batch_output_host.data(), batch_images, conf_threshold);
            }
        } else {
            // Use CPU batch postprocess
            results = postprocessBatch(batch_output_host.data(), batch_images, conf_threshold);
        }
        
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        updateLocalPostprocessTime(postprocess_time);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        updateLocalTotalTime(total_time);
        
        // Count detection results
        int total_detections = 0;
        for (const auto& result : results) {
            total_detections += result.size();
        }
        
        if (FLAGS_log_batch_timing) {
            LOG_INFO("⏱️  [批量推理时间-后处理] batch=" << batch_size << ", 后处理=" << postprocess_time << "ms");
            LOG_INFO("⏱️  [批量推理时间-总计] batch=" << batch_size 
                     << ", 前处理=" << preprocess_time << "ms"
                     << ", 推理=" << inference_time << "ms"
                     << ", 拷贝=" << copy_time << "ms"
                     << ", 后处理=" << postprocess_time << "ms"
                     << ", 总计=" << total_time << "ms"
                     << ", 检测数=" << total_detections 
                     << " | 平均每ROI=" << (total_time/batch_size) << "ms");
        }
        // Remove detailed performance statistics
        
    } catch (const std::exception& e) {
        LOG_ERROR("❌ [TensorRT batch inference] Exception: " << e.what() << ", fallback to single detection");
        
        // Fallback to single detection
        for (const auto& image : batch_images) {
            results.push_back(detect(image, conf_threshold));
        }
    }
    
    return results;
}

// 🔥 MultiStreamPreprocessContext实现
void TensorRTLocalInference::MultiStreamPreprocessContext::cleanup() {
    // 不在锁内进行CUDA同步，避免长时间阻塞
    std::vector<cudaStream_t> streams_to_destroy;
    std::vector<void*> buffers_to_free;
    
    {
        std::lock_guard<std::mutex> lock(mutex);
        streams_to_destroy = streams;
        buffers_to_free = temp_buffers;
        streams.clear();
        temp_buffers.clear();
        buffer_capacities.clear();
        num_streams = 0;
    }
    
    // 在锁外进行CUDA操作
    for (auto stream : streams_to_destroy) {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
    for (auto buffer : buffers_to_free) {
        if (buffer) cudaFree(buffer);
    }
}

bool TensorRTLocalInference::MultiStreamPreprocessContext::ensure_streams(
    int required_streams, size_t buffer_size_per_stream) {
    
    std::lock_guard<std::mutex> lock(mutex);
    
    // 检查是否需要重新分配
    if (required_streams <= num_streams) {
        bool all_buffers_ok = true;
        for (int i = 0; i < required_streams; i++) {
            if (buffer_capacities[i] < buffer_size_per_stream) {
                all_buffers_ok = false;
                break;
            }
        }
        if (all_buffers_ok) return true;
    }
    
    // 需要重新分配
    // 先清理旧资源（不在锁内进行同步）
    std::vector<cudaStream_t> old_streams = streams;
    std::vector<void*> old_buffers = temp_buffers;
    
    streams.clear();
    temp_buffers.clear();
    buffer_capacities.clear();
    
    // 在锁外清理旧资源
    mutex.unlock();
    for (auto stream : old_streams) {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
    for (auto buffer : old_buffers) {
        if (buffer) cudaFree(buffer);
    }
    mutex.lock();
    
    // 创建新的streams和buffers
    streams.resize(required_streams);
    temp_buffers.resize(required_streams);
    buffer_capacities.resize(required_streams);
    
    for (int i = 0; i < required_streams; i++) {
        // 创建stream
        cudaError_t stream_status = cudaStreamCreate(&streams[i]);
        if (stream_status != cudaSuccess) {
            LOG_ERROR("❌ 创建CUDA stream " << i << " 失败: " << cudaGetErrorString(stream_status));
            return false;
        }
        
        // 分配临时缓冲区
        cudaError_t malloc_status = cudaMalloc(&temp_buffers[i], buffer_size_per_stream);
        if (malloc_status != cudaSuccess) {
            LOG_ERROR("❌ 分配CUDA临时缓冲区 " << i << " 失败: " << cudaGetErrorString(malloc_status));
            return false;
        }
        
        buffer_capacities[i] = buffer_size_per_stream;
    }
    
    num_streams = required_streams;
    return true;
}

// 🔥 新增：批量预处理（真正的并行版本）
void TensorRTLocalInference::preprocessBatch(const std::vector<cv::Mat>& images, float* batch_input_device) {
    const int batch_size = images.size();
    
    // CPU preprocess mode
    if (preprocess_mode_ == 0) {
        std::vector<float> batch_input_host(batch_size * input_size_);
        
        #pragma omp parallel for
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> input_data(input_size_);
            preprocessImage(images[i], input_data.data());
            std::memcpy(batch_input_host.data() + i * input_size_, 
                    input_data.data(), input_size_ * sizeof(float));
        }
        
        cudaMemcpyAsync(batch_input_device, batch_input_host.data(), 
                    batch_size * input_size_ * sizeof(float), 
                    cudaMemcpyHostToDevice, stream_);
                    
        if (FLAGS_log_batch_timing) {
            LOG_INFO("🔧 [前处理] CPU并行模式, batch=" << batch_size << ", 使用OpenMP并行");
        }
    } 
    // 🔥 GPU preprocess mode - 使用多CUDA stream真正并行
    else if (preprocess_mode_ == 2) {
        // 1. 初始化多stream上下文
        if (!multi_stream_ctx_) {
            multi_stream_ctx_ = std::make_unique<MultiStreamPreprocessContext>();
        }
        
        // 2. 计算每个ROI需要的临时缓冲区大小
        size_t max_image_size = 0;
        for (const auto& img : images) {
            size_t img_size = img.step * img.rows;
            max_image_size = std::max(max_image_size, img_size);
        }
        
        // 3. 确保有足够的stream和临时缓冲区
        if (!multi_stream_ctx_->ensure_streams(batch_size, max_image_size)) {
            LOG_WARNING("⚠️ CUDA stream分配失败，回退到串行处理");
            for (int i = 0; i < batch_size; ++i) {
                float* current_input_device = batch_input_device + i * input_size_;
                preprocessImageGPU(images[i], current_input_device);
            }
            return;
        }
        
        // 4. 🔥 并行提交所有ROI到不同stream（真正的并行）
        for (int i = 0; i < batch_size; ++i) {
            float* current_input_device = batch_input_device + i * input_size_;
            cudaStream_t current_stream = multi_stream_ctx_->streams[i];
            void* temp_buffer = multi_stream_ctx_->temp_buffers[i];
            
            // 在独立stream上处理
            process_input_gpu_stream(images[i], current_input_device, temp_buffer, current_stream);
        }
        
        // 5. 等待所有stream完成
        for (int i = 0; i < batch_size; ++i) {
            cudaStreamSynchronize(multi_stream_ctx_->streams[i]);
        }
        
        if (FLAGS_log_batch_timing) {
            LOG_INFO("🔧 [前处理] GPU多Stream并行模式, batch=" << batch_size << ", 使用" << batch_size << "个独立CUDA stream");
        }
    }
    // 混合模式
    else {
        for (int i = 0; i < batch_size; ++i) {
            float* current_input_device = batch_input_device + i * input_size_;
            preprocessImageCVAffine(images[i], current_input_device);
        }
        
        if (FLAGS_log_batch_timing) {
            LOG_INFO("🔧 [前处理] CV仿射变换模式, batch=" << batch_size);
        }
    }
}

// 🔥 Modify: improve batch postprocess function implementation
std::vector<std::vector<Detection>> TensorRTLocalInference::postprocessBatch(
    float* batch_output, 
    const std::vector<cv::Mat>& original_images, 
    float conf_threshold) {
    
    const int batch_size = original_images.size();
    std::vector<std::vector<Detection>> all_results;
    all_results.reserve(batch_size);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use optimized batch postprocess method
    batchDecodeYOLOOutput(batch_output, batch_size, original_images, all_results, conf_threshold);
    
    return all_results;
}

// 🔥 New: batch YOLO decode function implementation
void TensorRTLocalInference::batchDecodeYOLOOutput(
    float* batch_output, 
    int batch_size,
    const std::vector<cv::Mat>& original_images,
    std::vector<std::vector<Detection>>& batch_detections,
    float conf_threshold) {
    
    // Ensure output container size is correct
    batch_detections.resize(batch_size);
    
    // Use already calculated num_boxes_ member variable, if not initialized then calculate
    int num_boxes = (num_boxes_ > 0) ? num_boxes_ : (output_size_ / kBoxInfoSize);
    int box_info_size = kBoxInfoSize;  // x, y, w, h, conf
    
    // LOG_INFO("🔍 [batch YOLO decode] Each sample output size: " << output_size_ << ", number of detection boxes: " << num_boxes 
    //           << ", number of box information: " << box_info_size << ", batch size: " << batch_size);
    
    // Batch process the output of each sample
    for (int b = 0; b < batch_size; ++b) {
        std::vector<Detection> detections;
        const cv::Mat& original_image = original_images[b];
        float* output = batch_output + b * output_size_;  // Pointer to the output of the current batch sample
        
        // Process all prediction boxes
        for (int i = 0; i < num_boxes; ++i) {
            float x = output[0 * num_boxes + i];
            float y = output[1 * num_boxes + i];
            float w = output[2 * num_boxes + i];
            float h = output[3 * num_boxes + i];
            float conf = output[4 * num_boxes + i];
            
            // Check confidence
            if (conf < conf_threshold || conf > 1.0f) {
                continue;
            }
            
            // Verify the reasonableness of the bounding box
            if (w <= 0 || h <= 0 || w > kInputW * 2 || h > kInputH * 2) {
                continue;
            }
            
            // Convert coordinates (from network input size to original image size)
            float scale = std::min(float(kInputW) / original_image.cols, 
                                  float(kInputH) / original_image.rows);
            int offsetx = (kInputW - original_image.cols * scale) / 2;
            int offsety = (kInputH - original_image.rows * scale) / 2;
            
            // Convert the center point coordinates to the upper left corner coordinates
            float x1 = (x - w/2 - offsetx) / scale;
            float y1 = (y - h/2 - offsety) / scale;
            float x2 = (x + w/2 - offsetx) / scale;
            float y2 = (y + h/2 - offsety) / scale;
            
            // Constrain to the original image range
            x1 = clamp(x1, 0, original_image.cols);
            y1 = clamp(y1, 0, original_image.rows);
            x2 = clamp(x2, 0, original_image.cols);
            y2 = clamp(y2, 0, original_image.rows);
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            if (width > 5 && height > 5) {  // Minimum size requirement
                Detection det;
                det.bbox = cv::Rect2f(x1, y1, width, height);
                det.confidence = conf;
                det.class_id = 0;
                
                detections.push_back(det);
            }
        }
        
        // Apply NMS
        float used_nms = static_cast<float>(FLAGS_local_nms_threshold);
        used_nms = std::max(0.3f, std::min(0.7f, used_nms));
        nms(detections, nullptr, conf_threshold, used_nms);
        
        // Store the detection results of the current batch
        batch_detections[b] = std::move(detections);
    }
}

// 🔥 Enhance: check if batch detection is supported - add detailed diagnosis
bool TensorRTLocalInference::supportsBatchDetection() const {
    if (!engine_) {
        LOG_ERROR("❌ [batch support check] Engine not initialized");
        return false;
    }
    
    // 🔥 Optimize: cache the check result, avoid repeated check
    static bool supports_cached = false;
    static bool cached_result = false;
    static bool first_check = true;
    
    if (first_check) {
        first_check = false;
        
        int actual_max_batch = getMaxBatchSize();
        
        // Check the input dimension to determine if dynamic batch is supported
        int nb_bindings = engine_->getNbBindings();
        bool has_dynamic_batch = false;
        bool has_fixed_batch_gt1 = false;
        int total_input_bindings = 0;
        int total_output_bindings = 0;
        
        for (int i = 0; i < nb_bindings; ++i) {
            const char* binding_name = engine_->getBindingName(i);
            bool is_input = engine_->bindingIsInput(i);
            auto dims = engine_->getBindingDimensions(i);
            
            if (is_input) {
                total_input_bindings++;
                if (Logger::isVLogEnabled(2)) {
                    std::stringstream ss;
                    ss << "📥 [input binding " << i << "] " << binding_name << " dimension: [";
                for (int j = 0; j < dims.nbDims; ++j) {
                        ss << dims.d[j];
                        if (j < dims.nbDims - 1) ss << ", ";
                }
                    ss << "]";
                    LOG_VERBOSE(2, ss.str());
                }
                
                if (dims.nbDims >= 1) {
                    int batch_dim = dims.d[0];
                    
                    if (batch_dim == -1) {
                        has_dynamic_batch = true;
                        LOG_INFO("   ✅ Detected dynamic batch dimension!");
                        
                        // 检查优化配置文件中的批次范围
                        int num_profiles = engine_->getNbOptimizationProfiles();
                        LOG_INFO("   🎯 Number of optimization profiles: " << num_profiles);
                        
                        // 🔥 Only check the active profile for multi-profile engines
                        int profile_start = (num_profiles > 1) ? active_profile_index_ : 0;
                        int profile_end = (num_profiles > 1) ? active_profile_index_ + 1 : num_profiles;
                        
                        for (int profile_idx = profile_start; profile_idx < profile_end; ++profile_idx) {
                            // Calculate the correct binding index for this profile
                            int bindings_per_profile = nb_bindings / std::max(1, num_profiles);
                            int profile_binding_idx = profile_idx * bindings_per_profile + (i % bindings_per_profile);
                            
                            if (profile_binding_idx < nb_bindings) {
                                auto min_dims = engine_->getProfileDimensions(profile_binding_idx, profile_idx, nvinfer1::OptProfileSelector::kMIN);
                                auto opt_dims = engine_->getProfileDimensions(profile_binding_idx, profile_idx, nvinfer1::OptProfileSelector::kOPT);
                                auto max_dims = engine_->getProfileDimensions(profile_binding_idx, profile_idx, nvinfer1::OptProfileSelector::kMAX);
                                
                                LOG_INFO("     profile " << profile_idx << " (active) batch range: [" 
                                      << min_dims.d[0] << ", " << opt_dims.d[0] << ", " << max_dims.d[0] << "]");
                            }
                        }
                    } else if (batch_dim > 1) {
                        has_fixed_batch_gt1 = true;
                        LOG_INFO("   ✅ Fixed batch size: " << batch_dim << " (>1)");
                    } else {
                        LOG_INFO("   ⚠️ Fixed batch size: " << batch_dim << " (=1)");
                    }
                }
            } else {
                total_output_bindings++;
            }
        }
        
        // Check the configuration of the engine at build time
        bool engine_supports_batch = (actual_max_batch > 1);
 
        // If neither dynamic batch nor fixed batch greater than 1 is detected
        if (!has_dynamic_batch && !has_fixed_batch_gt1) {
            LOG_WARNING("   ⚠️ Note: getMaxBatchSize() is greater than 1, but neither dynamic batch nor fixed batch greater than 1 is detected.\n"
                      << "      This usually means that the batch dimension of bindingDims is 1 before setBindingDimensions.\n"
                      << "      If you need to confirm further, please check if the kMAX value of each profile is >1, or call supportsBatchDetection() after setBindingDimensions.");
        }
        
        // Additional engine feature check
        #if NV_TENSORRT_MAJOR >= 8
        LOG_INFO("🏗️ [engine feature] TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR);
        
        // Check if the engine is built with dynamic shapes
        bool has_dynamic_shapes = false;
        for (int i = 0; i < nb_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    has_dynamic_shapes = true;
                    break;
                }
            }
            if (has_dynamic_shapes) break;
        }
        LOG_INFO("   - Dynamic shape support: " << (has_dynamic_shapes ? "✅" : "❌"));
        
        // Check the device memory size (indirectly reflect the engine complexity)
        size_t device_memory = engine_->getDeviceMemorySize();
        LOG_INFO("   - Device memory demand: " << device_memory / 1024 / 1024 << "MB");
        
        if (device_memory > 500 * 1024 * 1024) { // >500MB
            LOG_INFO("     (Large model, batch processing may have better benefits)");
        } else if (device_memory < 50 * 1024 * 1024) { // <50MB
            LOG_INFO("     (Lightweight model, batch processing may have limited benefits)");
        }
        #endif
        
        cached_result = engine_supports_batch;
        supports_cached = true;
        
        // Output the final conclusion and suggestions
        LOG_INFO("🏁 [Final conclusion] Batch detection support: " << (cached_result ? "✅ Supported" : "❌ Not supported"));
        
        if (!cached_result) {
            LOG_INFO("💡 [Suggestions for solutions]");
            
            if (actual_max_batch <= 1) {
                LOG_INFO("   1. Rebuild the TensorRT engine, set max_batch_size > 1");
                LOG_INFO("      trtexec --onnx=model.onnx --saveEngine=model.engine --maxBatch=8");
            }
            
            if (!has_dynamic_batch && !has_fixed_batch_gt1) {
                LOG_INFO("   2. Enable dynamic batch dimension building");
                LOG_INFO("      trtexec --onnx=model.onnx --saveEngine=model.engine \\");
                LOG_INFO("              --minShapes=input:1x3x640x640 \\");
                LOG_INFO("              --optShapes=input:4x3x640x640 \\");
                LOG_INFO("              --maxShapes=input:8x3x640x640");
            }
        } else {
            LOG_INFO("🎉 [Optimization suggestions]");
            LOG_INFO("   - Suggested batch size: 2-" << std::min(8, actual_max_batch) << " (adjust according to GPU memory)");
            LOG_INFO("   - Monitor GPU utilization, ensure batch processing充分利用GPU");
            
            if (has_dynamic_batch) {
                LOG_INFO("   - Can dynamically adjust the batch size to adapt to different scenarios");
            }
        }
        
        LOG_INFO("======================================\n");
    }
    
    return cached_result;
}

// 🔥 新增：获取最大批次大小
int TensorRTLocalInference::getMaxBatchSize() const {
    if (!engine_) {
        return 1;
    }

    // Always recalculate, to ensure the dynamic profile is ready
    int final_max_batch = 1;  // Default at least 1

    // Determine the engine batch mode
    if (engine_->hasImplicitBatchDimension()) {
        // Old Implicit-Batch, directly use the first value of the dimension
        for (int i = 0; i < engine_->getNbBindings(); i++) {
            if (engine_->bindingIsInput(i)) {
                auto dims = engine_->getBindingDimensions(i);
                if (dims.nbDims >= 1) {
                    final_max_batch = dims.d[0];
                    break; // Find the first input binding
                }
            }
        }
    } else {
        // Explicit-Batch: no longer call getMaxBatchSize()
        int nb_bindings = engine_->getNbBindings();
        int num_profiles = engine_->getNbOptimizationProfiles();

        // 🔥 For multi-profile engines, check only the active profile's bindings
        int bindings_per_profile = nb_bindings / std::max(1, num_profiles);
        int binding_offset = active_profile_index_ * bindings_per_profile;
        
        // Check input bindings in the active profile
        for (int i = 0; i < bindings_per_profile; i++) {
            int binding_idx = binding_offset + i;
            if (binding_idx >= nb_bindings || !engine_->bindingIsInput(binding_idx)) continue;
            
            // First check the binding dimension of the context
            if (context_ && context_->allInputDimensionsSpecified()) {
                auto dims = context_->getBindingDimensions(binding_idx);
                if (dims.nbDims > 0 && dims.d[0] > final_max_batch) {
                    final_max_batch = dims.d[0];
                }
            }

            // Then check the static dimension of the engine
            auto dims = engine_->getBindingDimensions(binding_idx);
            if (dims.nbDims > 0 && dims.d[0] > 0 && dims.d[0] > final_max_batch) {
                final_max_batch = dims.d[0];
            }
            
            // Finally check the maximum value of the active profile
            if (num_profiles > 0) {
                auto max_dims = engine_->getProfileDimensions(binding_idx, active_profile_index_, 
                                              nvinfer1::OptProfileSelector::kMAX);
                if (max_dims.nbDims > 0 && max_dims.d[0] > final_max_batch) {
                    final_max_batch = max_dims.d[0];
                }
            }
        }

        // Check if there is a dynamic batch dimension
        bool has_dynamic_batch = false;
        for (int i = 0; i < nb_bindings; i++) {
            if (!engine_->bindingIsInput(i)) continue;
            auto dims = engine_->getBindingDimensions(i);
            if (dims.nbDims > 0 && dims.d[0] == -1) {
                has_dynamic_batch = true;
                break;
            }
        }
        
        // For dynamic batch, if the maximum batch is still 1, use the default value 4
        if (has_dynamic_batch && final_max_batch == 1) {
            final_max_batch = 4;  // For dynamic batch, the reasonable default value is 4
            LOG_INFO("Detected dynamic batch dimension, but cannot determine the maximum batch size, using default value: " << final_max_batch);
        }
    }

    // Defensive check, ensure the return value is at least 1
    if (final_max_batch < 1) final_max_batch = 1;

    return final_max_batch;
}

bool TensorRTLocalInference::initializeBindings() {
    if (bindings_initialized_) return true;
    
    try {
        int nb_bindings = engine_->getNbBindings();
        bindings_.resize(nb_bindings, nullptr);
        
        for (int i = 0; i < nb_bindings; i++) {
            if (engine_->bindingIsInput(i)) {
                // Input binding
                auto dims = engine_->getBindingDimensions(i);
                size_t size = getBindingSize(dims);
                
                void* device_memory = nullptr;
                cudaError_t cuda_status = cudaMalloc(&device_memory, size);
                if (cuda_status != cudaSuccess) {
                    LOG_ERROR("❌ Input binding memory allocation failed: " << cudaGetErrorString(cuda_status));
                    return false;
                }
                bindings_[i] = device_memory;
                
                if (Logger::isVLogEnabled(1)) {
                    LOG_VERBOSE(1, "✅ Input binding[" << i << "] memory allocation successful: " << (size / 1024.0 / 1024.0) << "MB");
                }
            } else {
                // Output binding
                auto dims = engine_->getBindingDimensions(i);
                size_t size = getBindingSize(dims);
                
                void* device_memory = nullptr;
                cudaError_t cuda_status = cudaMalloc(&device_memory, size);
                if (cuda_status != cudaSuccess) {
                    LOG_ERROR("❌ Output binding memory allocation failed: " << cudaGetErrorString(cuda_status));
                    return false;
                }
                bindings_[i] = device_memory;
                
                if (Logger::isVLogEnabled(1)) {
                    LOG_VERBOSE(1, "✅ Output binding[" << i << "] memory allocation successful: " << (size / 1024.0 / 1024.0) << "MB");
                }
            }
        }
        
        bindings_initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("❌ Binding initialization failed: " << e.what());
        return false;
    }
}

size_t TensorRTLocalInference::getBindingSize(const nvinfer1::Dims& dims) {
    size_t size = sizeof(float);  // Assume using float type
    for (int j = 0; j < dims.nbDims; j++) {
        if (dims.d[j] < 0) {
            // Dynamic dimension, use the maximum possible value
            auto max_dims = engine_->getProfileDimensions(
                0, 0, nvinfer1::OptProfileSelector::kMAX);
            size *= max_dims.d[j];
        } else {
            size *= dims.d[j];
        }
    }
    return size;
}

void TensorRTLocalInference::debugBindings() {
    LOG_INFO("\n🔍 Debug binding information:");
    
    int nb_bindings = engine_->getNbBindings();
    LOG_INFO("Total binding number: " << nb_bindings);
    
    for (int i = 0; i < nb_bindings; i++) {
        LOG_INFO("\nBinding[" << i << "]:");
        LOG_INFO("  Name: " << engine_->getBindingName(i));
        LOG_INFO("  Type: " << (engine_->bindingIsInput(i) ? "Input" : "Output"));
        LOG_INFO("  Pointer: " << bindings_[i]);
        
        auto dims = engine_->getBindingDimensions(i);
        std::stringstream ss;
        ss << "  Dimension: [";
        for (int j = 0; j < dims.nbDims; j++) {
            ss << dims.d[j];
            if (j < dims.nbDims - 1) ss << ", ";
        }
        ss << "]";
        LOG_INFO(ss.str());
        
        size_t size = getBindingSize(dims);
        LOG_INFO("  Memory size: " << (size / 1024.0 / 1024.0) << "MB");
        
        // Verify if the memory is accessible
        if (bindings_[i]) {
            cudaPointerAttributes attrs;
            cudaError_t status = cudaPointerGetAttributes(&attrs, bindings_[i]);
            LOG_INFO("  Memory status: " << 
                (status == cudaSuccess ? "✅ Valid" : "❌ Invalid"));
        } else {
            LOG_INFO("  Memory status: ❌ Null pointer");
        }
    }
    LOG_INFO("");
}

// Initialize the preprocess context
bool TensorRTLocalInference::initPreprocessContext() {
    if (!preprocess_ctx_) {
        preprocess_ctx_ = std::make_unique<PreprocessContext>();
    }
    
    std::lock_guard<std::mutex> lock(preprocess_ctx_->mutex);
    
    if (!preprocess_ctx_->stream) {
        cudaError_t status = cudaStreamCreate(&preprocess_ctx_->stream);
        if (status != cudaSuccess) {
            std::cerr << "❌ Create preprocess CUDA stream failed: " << cudaGetErrorString(status) << std::endl;
            return false;
        }
    }
    
    return true;
}

// Ensure the preprocess buffer is large enough
bool TensorRTLocalInference::ensurePreprocessBuffer(size_t required_size) {
    if (!preprocess_ctx_) return false;
    
    std::lock_guard<std::mutex> lock(preprocess_ctx_->mutex);
    
    if (required_size <= preprocess_ctx_->buffer_capacity) {
        return true;
    }
    
    // Wait for all operations to complete
    if (preprocess_ctx_->stream) {
        CUDA_CHECK(cudaStreamSynchronize(preprocess_ctx_->stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Release the old buffer
    if (preprocess_ctx_->device_buffer) {
        CUDA_CHECK(cudaFree(preprocess_ctx_->device_buffer));
    }
    
    // Allocate new buffer
    cudaError_t status = cudaMalloc(&preprocess_ctx_->device_buffer, required_size);
    if (status != cudaSuccess) {
        LOG_ERROR("❌ Allocate preprocess buffer failed: " << cudaGetErrorString(status));
        preprocess_ctx_->device_buffer = nullptr;
        preprocess_ctx_->buffer_capacity = 0;
        return false;
    }
    
    preprocess_ctx_->buffer_capacity = required_size;
    LOG_INFO("✅ Preprocess buffer has been expanded to " << required_size / 1024.0 / 1024.0 << " MB");
    
    return true;
}

// Initialize the batch context
bool TensorRTLocalInference::initBatchContext() {
    if (!batch_ctx_) {
        batch_ctx_ = std::make_unique<BatchMemoryContext>();
    }
    return true;
}

// Ensure the batch buffer is large enough
bool TensorRTLocalInference::ensureBatchBuffers(int batch_size) {
    if (!batch_ctx_) return false;
    
    std::lock_guard<std::mutex> lock(batch_ctx_->mutex);
    
    if (batch_size <= batch_ctx_->max_batch_allocated) {
        return true;
    }
    
    // Calculate the required buffer size
    size_t input_size = batch_size * input_size_ * sizeof(float);
    size_t output_size = batch_size * output_size_ * sizeof(float);
    
    // Wait for all operations to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Release the old buffer
    if (batch_ctx_->input_buffer) {
        CUDA_CHECK(cudaFree(batch_ctx_->input_buffer));
    }
    if (batch_ctx_->output_buffer) {
        CUDA_CHECK(cudaFree(batch_ctx_->output_buffer));
    }
    
    // Allocate new buffer
    cudaError_t status1 = cudaMalloc(&batch_ctx_->input_buffer, input_size);
    cudaError_t status2 = cudaMalloc(&batch_ctx_->output_buffer, output_size);
    
    if (status1 != cudaSuccess || status2 != cudaSuccess) {
        LOG_ERROR("❌ Allocate batch buffer failed: " 
                  << cudaGetErrorString(status1) << " / " 
                  << cudaGetErrorString(status2));
        
        if (batch_ctx_->input_buffer) cudaFree(batch_ctx_->input_buffer);
        if (batch_ctx_->output_buffer) cudaFree(batch_ctx_->output_buffer);
        
        batch_ctx_->input_buffer = nullptr;
        batch_ctx_->output_buffer = nullptr;
        batch_ctx_->max_batch_allocated = 0;
        return false;
    }
    
    batch_ctx_->input_capacity = input_size;
    batch_ctx_->output_capacity = output_size;
    batch_ctx_->max_batch_allocated = batch_size;
    
    if (Logger::isVLogEnabled(1)) {
        LOG_VERBOSE(1, "✅ Batch buffer has been expanded to batch=" << batch_size 
                  << " (Input:" << input_size / 1024.0 / 1024.0 
                  << "MB, Output:" << output_size / 1024.0 / 1024.0 << "MB)");
    }
    
    return true;
}

// Safe ROI detection implementation
std::vector<Detection> TensorRTLocalInference::detectROIs(const std::vector<cv::Mat>& rois, float conf_threshold) {
    if (rois.empty()) {
        LOG_WARNING("⚠️ [ROI detection] Input ROI list is empty");
        return {};
    }
    
    LOG_INFO("🎯 [ROI detection] Start processing " << rois.size() << " ROI regions");
    
    try {
        // 1. Initialize the preprocess context
        if (!initPreprocessContext()) {
            LOG_ERROR("❌ [ROI detection] Preprocess context initialization failed");
            return {};
        }
        
        // 2. Initialize the batch context
        if (!initBatchContext()) {
            LOG_ERROR("❌ [ROI detection] Batch context initialization failed");
            return {};
        }
        
        int batch_size = rois.size();
        
        // 3. Ensure the batch buffer is large enough
        if (!ensureBatchBuffers(batch_size)) {
            LOG_ERROR("❌ [ROI detection] Batch buffer allocation failed");
            return {};
        }
        
        // 4. Calculate the maximum ROI size
        size_t max_roi_size = 0;
        for (const auto& roi : rois) {
            size_t roi_size = roi.step * roi.rows;  // Use step to ensure correct byte number
            max_roi_size = std::max(max_roi_size, roi_size);
        }
        
        // 5. Ensure the preprocess buffer is large enough
        if (!ensurePreprocessBuffer(max_roi_size)) {
            LOG_ERROR("❌ [ROI detection] Preprocess buffer allocation failed");
            return {};
        }
        
        LOG_INFO("📋 [ROI detection] Buffer preparation completed, maximum ROI=" << max_roi_size / 1024.0 / 1024.0 << "MB");
        
        // 6. Preprocess all ROI
        {
            std::lock_guard<std::mutex> preprocess_lock(preprocess_ctx_->mutex);
            std::lock_guard<std::mutex> batch_lock(batch_ctx_->mutex);
            
            for (int i = 0; i < batch_size; i++) {
                float* current_input = static_cast<float*>(batch_ctx_->input_buffer) + 
                                     i * input_size_;
                
                // Use the safe ROI preprocess function (already guaranteed continuous memory)
                process_roi_gpu(rois[i], current_input, 
                              preprocess_ctx_->device_buffer, 
                              preprocess_ctx_->stream);
            }
            
            // Wait for all preprocess to complete
            CUDA_CHECK(cudaStreamSynchronize(preprocess_ctx_->stream));
        }
        
        LOG_INFO("✅ [ROI detection] Preprocess completed");
        
        // 7. Set dynamic batch dimension
        if (needs_dynamic_setting_) {
            for (int i = 0; i < engine_->getNbBindings(); i++) {
                if (engine_->bindingIsInput(i)) {
                    auto dims = engine_->getBindingDimensions(i);
                    if (dims.d[0] == -1) {
                        nvinfer1::Dims input_dims = dims;
                        input_dims.d[0] = batch_size;
                        
                        if (!context_->setBindingDimensions(i, input_dims)) {
                            LOG_ERROR("❌ [ROI detection] Set dynamic dimension failed");
                            return {};
                        }
                    }
                }
            }
        }
        
        // 8. Prepare complete bindings and execute inference
        int nb_bindings = engine_->getNbBindings();
        std::vector<void*> batch_bindings(nb_bindings, nullptr);

        // Use the real detection output binding index
        int primary_output_index = detection_output_index_;
        std::vector<void*> extra_output_buffers;  // Record temporary output buffers, released after inference

        for (int i = 0; i < nb_bindings; ++i) {
            if (engine_->bindingIsInput(i)) {
                batch_bindings[i] = batch_ctx_->input_buffer;
            } else if (i == primary_output_index) {
                batch_bindings[i] = batch_ctx_->output_buffer;
            } else {
                // The other outputs (valid detection count, class, etc.) are smaller, here dynamically allocate according to the current batch
                auto dims = engine_->getBindingDimensions(i);
                size_t bytes = sizeof(float);
                for (int d = 0; d < dims.nbDims; ++d) {
                    int dim = dims.d[d];
                    if (dim == -1) {
                        // Only process the dynamic dimension of the batch, the other dynamic dimensions are processed as 1
                        dim = (d == 0 ? batch_size : 1);
                    }
                    bytes *= dim;
                }

                void* buffer = nullptr;
                cudaError_t status = cudaMalloc(&buffer, bytes);
                if (status != cudaSuccess) {
                    std::cerr << "❌ [ROI detection] Output binding memory allocation failed: "
                              << cudaGetErrorString(status) << std::endl;
                    // Release the allocated temporary buffer
                    for (void* ptr : extra_output_buffers) {
                        if (ptr) cudaFree(ptr);
                    }
                    return {};
                }
                batch_bindings[i] = buffer;
                extra_output_buffers.push_back(buffer);
            }
        }

        bool success = context_->enqueueV2(batch_bindings.data(), stream_, nullptr);
        if (!success) {
            std::cerr << "❌ [ROI detection] Inference execution failed" << std::endl;
            // Release the temporary buffer
            for (void* ptr : extra_output_buffers) {
                if (ptr) cudaFree(ptr);
            }
            return {};
        }
         
         // Wait for inference to complete
         CUDA_CHECK(cudaStreamSynchronize(stream_));

        // After inference, immediately release the extra output buffer
        for (void* ptr : extra_output_buffers) {
            if (ptr) cudaFree(ptr);
        }
        extra_output_buffers.clear();
 
        LOG_INFO("✅ [ROI detection] Inference completed");
        
        // 9. Process the result
        std::vector<float> batch_output_host(batch_size * output_size_);
        CUDA_CHECK(cudaMemcpy(batch_output_host.data(), batch_ctx_->output_buffer,
                             batch_size * output_size_ * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        // 10. Parse the output (here simplified to process the result of the first ROI)
        std::vector<Detection> results = parseYOLOOutput(
            batch_output_host.data(), output_size_, 
            rois[0], conf_threshold);
        
        LOG_INFO("🎯 [ROI detection] Completed, detected " << results.size() << " targets");
        
        // 🔥 Debug: Print ROI detection details
        if (Logger::isVLogEnabled(1) && !results.empty()) {
            LOG_VERBOSE(1, "🎯 [ROI] ROI size: " << rois[0].cols << "x" << rois[0].rows);
            LOG_VERBOSE(1, "🎯 [ROI] Detected " << results.size() << " objects:");
            for (size_t i = 0; i < std::min(results.size(), size_t(5)); i++) {
                const auto& det = results[i];
                LOG_VERBOSE(1, "  [" << i << "] bbox=[" << det.bbox.x << "," << det.bbox.y 
                          << "," << det.bbox.width << "," << det.bbox.height 
                          << "] conf=" << det.confidence);
            }
        }
        
        return results;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ [ROI detection] Exception: " << e.what() << std::endl;
        return {};
    }
}

// Batch preprocess mode related functions have been removed

// Local inference time statistics method implementation
void TensorRTLocalInference::updateLocalPreprocessTime(double preprocess_time) {
    total_preprocess_time_ += preprocess_time;
    preprocess_count_++;
    avg_preprocess_time_ = total_preprocess_time_ / preprocess_count_;
    max_preprocess_time_ = std::max(max_preprocess_time_, preprocess_time);
    min_preprocess_time_ = std::min(min_preprocess_time_, preprocess_time);
    
    // Output statistics every 10 times
    if (preprocess_count_ % 10 == 0) {
        LOG_INFO("🔥 Local preprocess time statistics triggered, count: " << preprocess_count_);
        printLocalTimeStatistics();
    }
}

void TensorRTLocalInference::updateLocalInferenceTime(double inference_time) {
    total_inference_time_ += inference_time;
    inference_count_++;
    avg_inference_time_ = total_inference_time_ / inference_count_;
    max_inference_time_ = std::max(max_inference_time_, inference_time);
    min_inference_time_ = std::min(min_inference_time_, inference_time);
}

void TensorRTLocalInference::updateLocalPostprocessTime(double postprocess_time) {
    total_postprocess_time_ += postprocess_time;
    postprocess_count_++;
    avg_postprocess_time_ = total_postprocess_time_ / postprocess_count_;
    max_postprocess_time_ = std::max(max_postprocess_time_, postprocess_time);
    min_postprocess_time_ = std::min(min_postprocess_time_, postprocess_time);
}

void TensorRTLocalInference::updateLocalTotalTime(double total_time) {
    total_processing_time_ += total_time;
    avg_total_time_ = total_processing_time_ / inference_count_;
    max_total_time_ = std::max(max_total_time_, total_time);
    min_total_time_ = std::min(min_total_time_, total_time);
}

void TensorRTLocalInference::printLocalTimeStatistics() {
    LOG_INFO("=== Local inference time statistics report ===");
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
        
        LOG_INFO("Local inference time proportion analysis:");
        LOG_INFO("  Preprocess time proportion: " << std::fixed << std::setprecision(1) << preprocess_ratio << "%");
        LOG_INFO("  Inference time proportion: " << std::fixed << std::setprecision(1) << inference_ratio << "%");
        LOG_INFO("  Postprocess time proportion: " << std::fixed << std::setprecision(1) << postprocess_ratio << "%");
        LOG_INFO("  Other time proportion: " << std::fixed << std::setprecision(1) << other_ratio << "%");
    }
    
    LOG_INFO("==================");
}

} // namespace tracking