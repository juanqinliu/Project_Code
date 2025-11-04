#include "inference/local_postprocess.h"
#include "common/Logger.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

namespace tracking {
namespace gpu {

// Define constants
constexpr int kInputW = 640;
constexpr int kInputH = 640;
constexpr int kBoxInfoSize = 5;   // x, y, w, h, conf 
constexpr int kLocalMinBoxWidth = 2;  // Local model minimum box width
constexpr int kLocalMinBoxHeight = 2; // Local model minimum box height
constexpr int kLocalMaxDetections = 100; // Local model maximum detections (decreased to avoid memory issues)

// CUDA check macro
#define CUDA_CHECK(callstr) \
    { \
        cudaError_t error_code = callstr; \
        if (error_code != cudaSuccess) { \
            LOG_ERROR("CUDA Error " << error_code << ": " << cudaGetErrorString(error_code)); \
            throw std::runtime_error("CUDA error"); \
        } \
    }

// CUDA kernel function: convert local YOLO detection box format to physical coordinates [x,y,w,h] -> [x1,y1,x2,y2]
__global__ void transformLocalBoxesKernel(
    float* boxes,        // [num_boxes, 4] 格式 (x,y,w,h)
    int num_boxes,
    int img_width,
    int img_height,
    int input_width,
    int input_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    // Get pointer to current box (x,y,w,h format)
    float* box = boxes + idx * 4;
    
    // Get original coordinates
    float x = box[0];
    float y = box[1];
    float w = box[2];
    float h = box[3];
    
    // Check coordinate validity
    if (!isfinite(x) || !isfinite(y) || !isfinite(w) || !isfinite(h)) {
        // If coordinate is invalid, set to zero area box
        box[0] = 0.0f;
        box[1] = 0.0f;
        box[2] = 0.0f;
        box[3] = 0.0f;
        return;
    }
    
    // // Calculate scale (local detection model)
    // float scale = min(float(input_width) / img_width, float(input_height) / img_height);
    // Note: offset is not used in local model processing, coordinates are directly divided by scale factor
    
    // Local model processing - ROI extracted image, no need to consider global image offset
    // But need to map coordinates back to ROI real size relative to input size (640x640)
    // Calculate letterbox scale and padding used in preprocessing
    // r = min(input_w / img_w, input_h / img_h)
    float r = min(float(input_width) / float(img_width), float(input_height) / float(img_height));
    float new_w = float(img_width) * r;
    float new_h = float(img_height) * r;
    float dw = (float(input_width) - new_w) * 0.5f;   // padding on width (left/right)
    float dh = (float(input_height) - new_h) * 0.5f;  // padding on height (top/bottom)

    // Map from network input space (with padding) back to ROI image space
    // Boxes from model are in cx,cy,w,h on input canvas; need to remove padding then divide by r
    float x1 = (x - w * 0.5f - dw) / r;
    float y1 = (y - h * 0.5f - dh) / r;
    float x2 = (x + w * 0.5f - dw) / r;
    float y2 = (y + h * 0.5f - dh) / r;
    
    // Limit within image range
    x1 = max(0.0f, min(float(img_width), x1));
    y1 = max(0.0f, min(float(img_height), y1));
    x2 = max(0.0f, min(float(img_width), x2));
    y2 = max(0.0f, min(float(img_height), y2));
    // Check validity of converted coordinates
    if (!isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2)) {
        // If coordinate is invalid, set to zero area box
        box[0] = 0.0f;
        box[1] = 0.0f;
        box[2] = 0.0f;
        box[3] = 0.0f;
        return;
    }
    
    // Store result (变为x1,y1,x2,y2格式)
    box[0] = x1;
    box[1] = y1;
    box[2] = x2;
    box[3] = y2;
}

// CUDA kernel function: compute IoU
__device__ float computeLocalIoU(float* box1, float* box2) {
    float x1 = max(box1[0], box2[0]);
    float y1 = max(box1[1], box2[1]);
    float x2 = min(box1[2], box2[2]);
    float y2 = min(box1[3], box2[3]);
    
    float width = max(0.0f, x2 - x1);
    float height = max(0.0f, y2 - y1);
    float intersection = width * height;
    
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float unionArea = area1 + area2 - intersection;
    
    return intersection / unionArea;
}

// CUDA kernel function: execute NMS
__global__ void localNmsKernel(
    float* boxes,         // [num_boxes, 4] format (x1,y1,x2,y2)
    float* scores,        // [num_boxes]
    int* indices,         // [num_boxes] input is index, output is index
    int* count,           // atomic counter, used to record number of retained boxes
    int num_boxes,
    float threshold
) {
    // Get index of current processed box
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    // Get index of current box
    int box_idx = indices[idx];
    
    // If index is marked as -1, it means it has been suppressed, skip
    if (box_idx == -1) return;
    
    // Get current box
    float* current_box = boxes + box_idx * 4;
    // Score is only used for logging, not used here
    
    // Compute IoU with subsequent all boxes and suppress overlapping boxes
    for (int i = idx + 1; i < num_boxes; i++) {
        int other_idx = indices[i];
        if (other_idx == -1) continue;  // Has been suppressed
        
        float* other_box = boxes + other_idx * 4;
        float iou = computeLocalIoU(current_box, other_box);
        
        if (iou > threshold) {
            // Atomic operation, mark suppressed box as -1
            atomicExch(&indices[i], -1);
        }
    }
}

// GPU postprocess implementation
void transformLocalBoxesGPU(
    float* d_boxes,
    int num_boxes,
    const cv::Mat& original_image,
    int input_width,
    int input_height
) {
    // Determine CUDA grid and block size
    int block_size = 256;
    int grid_size = (num_boxes + block_size - 1) / block_size;
    
    // Call CUDA kernel function to convert box coordinates
    transformLocalBoxesKernel<<<grid_size, block_size>>>(
        d_boxes,
        num_boxes,
        original_image.cols,
        original_image.rows,
        input_width,
        input_height
    );
    
    // Check CUDA error
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Execute NMS on GPU
void localNmsGPU(float* boxes, float* scores, int* indices, int& nboxes, int boxes_num, float threshold) {
    // Allocate device memory
    float* d_boxes;
    float* d_scores;
    int* d_indices;
    int* d_count;
    
    CUDA_CHECK(cudaMalloc(&d_boxes, boxes_num * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, boxes_num * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, boxes_num * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    
    // Copy data to device memory
    CUDA_CHECK(cudaMemcpy(d_boxes, boxes, boxes_num * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, scores, boxes_num * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize index array
    std::vector<int> h_indices(boxes_num);
    for (int i = 0; i < boxes_num; i++) {
        h_indices[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), boxes_num * sizeof(int), cudaMemcpyHostToDevice));
    
    // Use Thrust to sort scores (index sorting)
    thrust::sort_by_key(
        thrust::device,
        thrust::device_pointer_cast(d_scores),
        thrust::device_pointer_cast(d_scores) + boxes_num,
        thrust::device_pointer_cast(d_indices),
        thrust::greater<float>()  // Descending order
    );
    
    // Initialize counter to 0
    int count = 0;
    CUDA_CHECK(cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice));
    
    // Execute NMS
    int block_size = 256;
    int grid_size = (boxes_num + block_size - 1) / block_size;
    
    localNmsKernel<<<grid_size, block_size>>>(
        d_boxes, d_scores, d_indices, d_count, boxes_num, threshold
    );
    
    // Check CUDA error
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(indices, d_indices, boxes_num * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate number of valid boxes
    nboxes = 0;
    for (int i = 0; i < boxes_num; i++) {
        if (indices[i] != -1) nboxes++;
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_boxes));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_count));
}

// CUDA kernel function: extract valid detections (local model专用）
__global__ void extractLocalDetectionsKernel(
    float* output,        // YOLO original output [5, num_boxes]
    int num_boxes,
    float conf_threshold,
    float* valid_boxes,   // Output valid boxes [max_detections, 5]
    int* valid_count,     // Atomic counter
    int max_detections    // Maximum number of detections
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    // Local model detection needs stronger filtering - first check confidence then access other data
    float conf = output[4 * num_boxes + idx];
    
    // First check confidence, quickly filter obviously invalid boxes
    if (isnan(conf) || isinf(conf) || conf < conf_threshold || conf > 1.0f) return;
    
    // If passed confidence check, then extract coordinates
    float x = output[0 * num_boxes + idx];
    float y = output[1 * num_boxes + idx];
    float w = output[2 * num_boxes + idx];
    float h = output[3 * num_boxes + idx];
    
    // Enhanced check: exclude obviously invalid values
    if (isnan(x) || isnan(y) || isnan(w) || isnan(h)) return;
    if (isinf(x) || isinf(y) || isinf(w) || isinf(h)) return;
    
    // More strict size check (local model)
    if (w <= 0 || h <= 0 || w > kInputW || h > kInputH) return;
    
    // Size ratio check - exclude strange aspect ratios
    float aspect_ratio = w / max(h, 0.1f);
    if (aspect_ratio > 20.0f || aspect_ratio < 0.05f) return;  // Exclude extreme aspect ratios
    
    // Atomic increase counter and get index, ensure不超过最大检测框数
    int pos = atomicAdd(valid_count, 1);
    if (pos >= max_detections) return;  // If exceeds maximum number of detections, ignore this box
    
    // Store detection box (x,y,w,h,conf)
    valid_boxes[pos * 5 + 0] = x;
    valid_boxes[pos * 5 + 1] = y;
    valid_boxes[pos * 5 + 2] = w;
    valid_boxes[pos * 5 + 3] = h;
    valid_boxes[pos * 5 + 4] = conf;
}

// Parse local YOLO output and execute NMS on GPU
std::vector<Detection> parseLocalYOLOOutputGPU(
    float* output,
    int output_size,
    const cv::Mat& original_image,
    float conf_threshold,
    float nms_threshold
) {
    std::vector<Detection> detections;
    
    // Used to track CUDA memory allocation, ensure all memory is released
    std::vector<void*> allocated_memory;
    
    try {
        // Parameter validation
        if (output == nullptr) {
            LOG_ERROR("Local YOLO output pointer is null");
            return detections;
        }
        
        if (output_size <= 0) {
            LOG_ERROR("Invalid output size of local YOLO: " << output_size);
            return detections;
        }
        
        if (original_image.empty()) {
            LOG_ERROR("Original image is empty");
            return detections;
        }
        
        // Ensure valid value size
        if (output_size > 20000000) { // 20MB数据的浮点数
            LOG_ERROR("Local YOLO output size is too large: " << output_size << ",可能导致内存问题");
            return detections;
        }
        
        // Calculate number of boxes
        int num_boxes = output_size / kBoxInfoSize;
        // In extreme case, limit number of boxes
        int max_detections = std::min(std::min(num_boxes, kLocalMaxDetections), 50); // More strictly limit to 50
        
                 // Remove local model postprocess log
        
        // Allocate temporary GPU memory and track
        float* d_output = nullptr;
        float* d_valid_boxes = nullptr;  // Store valid detections [x,y,w,h,conf]
        int* d_valid_count = nullptr;    // Valid box counter
        
        // Protective memory allocation
        try {
            // Allocate and copy input
            cudaError_t err = cudaMalloc(&d_output, output_size * sizeof(float));
            if (err != cudaSuccess) {
                LOG_ERROR("Local detection GPU memory allocation failed (d_output): " << cudaGetErrorString(err));
                throw std::runtime_error("GPU memory allocation failed");
            }
            allocated_memory.push_back(d_output);
            
            CUDA_CHECK(cudaMemcpy(d_output, output, output_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Allocate memory for valid detections (最多保留max_detections个框）
            err = cudaMalloc(&d_valid_boxes, max_detections * 5 * sizeof(float));
            if (err != cudaSuccess) {
                LOG_ERROR("Local detection GPU memory allocation failed (d_valid_boxes): " << cudaGetErrorString(err));
                throw std::runtime_error("GPU memory allocation failed");
            }
            allocated_memory.push_back(d_valid_boxes);
            
            // Initialize valid detection memory to 0
            CUDA_CHECK(cudaMemset(d_valid_boxes, 0, max_detections * 5 * sizeof(float)));
            
            // Initialize counter to 0
            int valid_count = 0;
            err = cudaMalloc(&d_valid_count, sizeof(int));
            if (err != cudaSuccess) {
                LOG_ERROR("Local detection GPU memory allocation failed (d_valid_count): " << cudaGetErrorString(err));
                throw std::runtime_error("GPU memory allocation failed");
            }
            allocated_memory.push_back(d_valid_count);
            
            CUDA_CHECK(cudaMemcpy(d_valid_count, &valid_count, sizeof(int), cudaMemcpyHostToDevice));
            
            // Extract valid detections - using more secure configuration
            int block_size = 128; // Decreased to avoid长时间占用GPU
            int grid_size = (num_boxes + block_size - 1) / block_size;
            
            // Limit grid size, prevent large input causing problems
            if (grid_size > 1000) {
                LOG_WARNING("Local detection grid size is too large, limited to 1000");
                grid_size = 1000;
            }
            
            // Call local detection专用提取核函数
            extractLocalDetectionsKernel<<<grid_size, block_size>>>(
                d_output, num_boxes, conf_threshold, d_valid_boxes, d_valid_count, max_detections
            );
            
            // Check CUDA error
            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                LOG_ERROR("Local detection CUDA kernel execution error: " << cudaGetErrorString(cuda_status));
                throw std::runtime_error("CUDA kernel execution error");
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Get number of valid boxes
            valid_count = 0;  // Reuse declared variable
            CUDA_CHECK(cudaMemcpy(&valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost));
            
            // Ensure不超过最大检测框数量
            valid_count = std::min(valid_count, max_detections);
            
            if (valid_count == 0) {
                return detections;  // Return directly, destructor will release memory
            }
            
            if (valid_count > max_detections) {
                valid_count = max_detections;
            }
            
            // Allocate host memory and copy valid boxes
            std::vector<float> h_valid_boxes(valid_count * 5);
            CUDA_CHECK(cudaMemcpy(h_valid_boxes.data(), d_valid_boxes, valid_count * 5 * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Extract coordinates and confidence
            std::vector<float> boxes(valid_count * 4);  // x,y,w,h -> x1,y1,x2,y2
            std::vector<float> scores(valid_count);
            
            int valid_box_count = 0; // Track actual number of valid boxes
            
            for (int i = 0; i < valid_count; i++) {
                // Additional check for coordinate validity
                bool valid_coord = true;
                for (int j = 0; j < 5; j++) {
                    if (!std::isfinite(h_valid_boxes[i*5 + j])) {
                        valid_coord = false;
                        break;
                    }
                }
                
                if (!valid_coord) {
                    LOG_WARNING("Local detection detected invalid coordinates, index: " << i);
                    continue;
                }
                
                // Check the validity of the box
                float x = h_valid_boxes[i*5 + 0];
                float y = h_valid_boxes[i*5 + 1];
                float w = h_valid_boxes[i*5 + 2];
                float h = h_valid_boxes[i*5 + 3];
                float conf = h_valid_boxes[i*5 + 4];
                
                // Additional check for aspect ratio and size
                if (w <= 0 || h <= 0 || w > kInputW || h > kInputH || 
                    w/h > 20 || h/w > 20) { // 排除极端长宽比
                    continue;
                }
                
                // Copy original coordinates
                boxes[valid_box_count*4 + 0] = x;  // x
                boxes[valid_box_count*4 + 1] = y;  // y
                boxes[valid_box_count*4 + 2] = w;  // w
                boxes[valid_box_count*4 + 3] = h;  // h
                scores[valid_box_count] = conf;    // conf
                valid_box_count++;
            }
            
            // Adjust size to match actual number of valid boxes
            if (valid_box_count < valid_count) {
                boxes.resize(valid_box_count * 4);
                scores.resize(valid_box_count);
                valid_count = valid_box_count;
            }
            
            // Ensure valid data
            if (valid_count == 0) {
                return detections;  // Return directly, destructor will release memory
            }
            
            // Convert box coordinates to original image space
            float* d_boxes = nullptr;
            cudaError_t box_err = cudaMalloc(&d_boxes, valid_count * 4 * sizeof(float));
            if (box_err != cudaSuccess) {
                LOG_ERROR("Local detection GPU memory allocation failed (d_boxes): " << cudaGetErrorString(box_err));
                throw std::runtime_error("GPU memory allocation failed");
            }
            allocated_memory.push_back(d_boxes);
            
            CUDA_CHECK(cudaMemcpy(d_boxes, boxes.data(), valid_count * 4 * sizeof(float), cudaMemcpyHostToDevice));
            
            // Convert box coordinates (using local detection专用转换函数)
            try {
                transformLocalBoxesGPU(d_boxes, valid_count, original_image, kInputW, kInputH);
            } catch (const std::exception& e) {
                LOG_ERROR("Local coordinate conversion failed: " << e.what());
                throw;
            }
            
            // Copy converted coordinates back to host
            CUDA_CHECK(cudaMemcpy(boxes.data(), d_boxes, valid_count * 4 * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Additional check for converted coordinates validity
            int valid_after_transform = 0;
            for (int i = 0; i < valid_count; i++) {
                bool valid_box = true;
                for (int j = 0; j < 4; j++) {
                    if (!std::isfinite(boxes[i*4 + j])) {
                        valid_box = false;
                        break;
                    }
                }
                
                if (!valid_box) {
                    scores[i] = -1.0f;  // Mark as invalid
                } else {
                    valid_after_transform++;
                }
            }
            
            if (valid_after_transform == 0) {
                return detections;  // Return directly, destructor will release memory
            }
            
            // Execute NMS
            std::vector<int> indices(valid_count);
            int num_after_nms = 0;
            
            try {
                localNmsGPU(boxes.data(), scores.data(), indices.data(), num_after_nms, valid_count, nms_threshold);
            } catch (const std::exception& e) {
                LOG_ERROR("Local NMS execution failed: " << e.what());
                return detections;
            }
            
            // Remove NMS result log
            
            // Create final detection results
            detections.reserve(num_after_nms);
            
            for (int i = 0; i < valid_count; i++) {
                if (indices[i] != -1) {
                    int box_idx = indices[i];
                    
                    // Additional check for validity
                    if (scores[box_idx] < 0) continue;  // 跳过标记为无效的框
                    
                    float x1 = boxes[box_idx*4 + 0];
                    float y1 = boxes[box_idx*4 + 1];
                    float x2 = boxes[box_idx*4 + 2];
                    float y2 = boxes[box_idx*4 + 3];
                    
                    // Check coordinate range
                    if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 || 
                        x1 > original_image.cols || y1 > original_image.rows ||
                        x2 > original_image.cols || y2 > original_image.rows) {
                        continue;  // Skip boxes out of range
                    }
                    
                    float width = x2 - x1;
                    float height = y2 - y1;
                    float conf = scores[box_idx];
                    
                    // Check the validity of width and height
                    if (width <= 0 || height <= 0) continue;
                    
                    // Local model uses smaller minimum box size
                    if (width > kLocalMinBoxWidth && height > kLocalMinBoxHeight) {
                        // Additional check for coordinate validity 
                        if (std::isfinite(x1) && std::isfinite(y1) && std::isfinite(width) && std::isfinite(height)) {
                            Detection det;
                            det.bbox = cv::Rect2f(x1, y1, width, height);
                            det.confidence = conf;
                            det.class_id = 0;
                            det.is_from_global_model = false;  // Mark as local model detection result
                            
                            detections.push_back(det);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Local detection processing exception: " << e.what());
            // Exception handling - ensure memory is released
        }
        
        // Ensure all allocated device memory is released
        for (void* ptr : allocated_memory) {
            if (ptr) {
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    LOG_WARNING("Release local detection GPU memory warning: " << cudaGetErrorString(err));
                }
            }
        }
        
        // Ensure synchronization is complete
        cudaDeviceSynchronize();
        
    } catch (const std::exception& e) {
        LOG_ERROR("Local detection GPU postprocess failed: " << e.what());
        
        // Try to release any memory that may still be allocated
        for (void* ptr : allocated_memory) {
            if (ptr) cudaFree(ptr);
        }
        
        // Reset device
        cudaDeviceReset();
    }
    
    return detections;
}

// 直接从设备端输出解析，避免整块GPU->CPU拷贝
std::vector<Detection> parseLocalYOLOOutputGPUFromDevice(
    float* d_output,
    int output_size,
    const cv::Mat& original_image,
    float conf_threshold,
    float nms_threshold,
    cudaStream_t stream
) {
    std::vector<Detection> detections;

    // 记录需要释放的设备内存
    std::vector<void*> allocated_memory;

    try {
        if (d_output == nullptr) {
            LOG_ERROR("Local YOLO device output pointer is null");
            return detections;
        }
        if (output_size <= 0) {
            LOG_ERROR("Invalid output size of local YOLO: " << output_size);
            return detections;
        }
        if (original_image.empty()) {
            LOG_ERROR("Original image is empty");
            return detections;
        }

        int num_boxes = output_size / kBoxInfoSize;
        int max_detections = std::min(std::min(num_boxes, kLocalMaxDetections), 50);

        // 为有效框/计数分配设备内存
        float* d_valid_boxes = nullptr;
        int* d_valid_count = nullptr;
        cudaError_t err = cudaMalloc(&d_valid_boxes, max_detections * 5 * sizeof(float));
        if (err != cudaSuccess) {
            LOG_ERROR("Local detection GPU memory allocation failed (d_valid_boxes): " << cudaGetErrorString(err));
            throw std::runtime_error("GPU memory allocation failed");
        }
        allocated_memory.push_back(d_valid_boxes);
        CUDA_CHECK(cudaMemsetAsync(d_valid_boxes, 0, max_detections * 5 * sizeof(float), stream));

        int zero = 0;
        err = cudaMalloc(&d_valid_count, sizeof(int));
        if (err != cudaSuccess) {
            LOG_ERROR("Local detection GPU memory allocation failed (d_valid_count): " << cudaGetErrorString(err));
            throw std::runtime_error("GPU memory allocation failed");
        }
        allocated_memory.push_back(d_valid_count);
        CUDA_CHECK(cudaMemcpyAsync(d_valid_count, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));

        // 提取有效检测（直接使用设备端输出）
        int block_size = 128;
        int grid_size = (num_boxes + block_size - 1) / block_size;
        if (grid_size > 1000) grid_size = 1000;

        extractLocalDetectionsKernel<<<grid_size, block_size, 0, stream>>>(
            d_output, num_boxes, conf_threshold, d_valid_boxes, d_valid_count, max_detections);
        CUDA_CHECK(cudaGetLastError());

        // 拷回有效框数量
        int valid_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        valid_count = std::min(valid_count, max_detections);
        if (valid_count <= 0) {
            // 释放内存
            for (void* ptr : allocated_memory) { if (ptr) cudaFree(ptr); }
            return detections;
        }

        // 拷回有效框 (x,y,w,h,conf)
        std::vector<float> h_valid_boxes(valid_count * 5);
        CUDA_CHECK(cudaMemcpyAsync(h_valid_boxes.data(), d_valid_boxes, valid_count * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 拆分出 boxes 和 scores
        std::vector<float> boxes(valid_count * 4);
        std::vector<float> scores(valid_count);
        int keep = 0;
        for (int i = 0; i < valid_count; ++i) {
            float x = h_valid_boxes[i*5 + 0];
            float y = h_valid_boxes[i*5 + 1];
            float w = h_valid_boxes[i*5 + 2];
            float h = h_valid_boxes[i*5 + 3];
            float conf = h_valid_boxes[i*5 + 4];
            if (w <= 0 || h <= 0 || w > kInputW || h > kInputH || w/h > 20 || h/w > 20) continue;
            boxes[keep*4 + 0] = x;
            boxes[keep*4 + 1] = y;
            boxes[keep*4 + 2] = w;
            boxes[keep*4 + 3] = h;
            scores[keep] = conf;
            keep++;
        }
        if (keep == 0) {
            for (void* ptr : allocated_memory) { if (ptr) cudaFree(ptr); }
            return detections;
        }
        boxes.resize(keep * 4);
        scores.resize(keep);

        // 坐标转换到原图尺度（使用现有GPU函数，内部会同步）
        float* d_boxes_xyxy = nullptr;
        err = cudaMalloc(&d_boxes_xyxy, keep * 4 * sizeof(float));
        if (err != cudaSuccess) {
            LOG_ERROR("Local detection GPU memory allocation failed (d_boxes_xyxy): " << cudaGetErrorString(err));
            throw std::runtime_error("GPU memory allocation failed");
        }
        allocated_memory.push_back(d_boxes_xyxy);
        CUDA_CHECK(cudaMemcpyAsync(d_boxes_xyxy, boxes.data(), keep * 4 * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 复用已有转换（使用默认流同步），精度上无影响；如需进一步优化，可改为带stream版本
        transformLocalBoxesGPU(d_boxes_xyxy, keep, original_image, kInputW, kInputH);

        // 拷回转换后的坐标
        CUDA_CHECK(cudaMemcpyAsync(boxes.data(), d_boxes_xyxy, keep * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // GPU NMS（现有实现使用默认流并设备同步）
        std::vector<int> indices(keep);
        int num_after_nms = 0;
        localNmsGPU(boxes.data(), scores.data(), indices.data(), num_after_nms, keep, nms_threshold);

        detections.reserve(num_after_nms);
        for (int i = 0; i < keep; ++i) {
            if (indices[i] == -1) continue;
            int bi = indices[i];
            float x1 = boxes[bi*4 + 0];
            float y1 = boxes[bi*4 + 1];
            float x2 = boxes[bi*4 + 2];
            float y2 = boxes[bi*4 + 3];
            if (x1 < 0 || y1 < 0 || x2 <= x1 || y2 <= y1) continue;
            if (x2 > original_image.cols || y2 > original_image.rows) continue;
            Detection det;
            det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            det.confidence = scores[bi];
            det.class_id = 0;
            det.is_from_global_model = false;
            detections.push_back(det);
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Local detection GPU device-parse failed: " << e.what());
    }

    for (void* ptr : allocated_memory) {
        if (ptr) cudaFree(ptr);
    }

    return detections;
}

// Batch process local YOLO output and execute NMS on GPU
std::vector<std::vector<Detection>> batchDecodeLocalYOLOOutputGPU(
    float* batch_output,
    int batch_size,
    const std::vector<cv::Mat>& original_images,
    float conf_threshold,
    float nms_threshold
) {
    // Deprecated batch processing function
    std::vector<std::vector<Detection>> batch_detections(batch_size);
    
    // Ensure parameter validity
    if (batch_output == nullptr || batch_size <= 0 || original_images.empty()) {
        return batch_detections;
    }
    
    
    return batch_detections;
}

} // namespace gpu
} // namespace tracking 