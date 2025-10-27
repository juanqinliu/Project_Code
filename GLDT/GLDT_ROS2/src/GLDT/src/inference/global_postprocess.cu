#include "inference/global_postprocess.h"
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

// CUDA check macro
#define CUDA_CHECK(callstr) \
    { \
        cudaError_t error_code = callstr; \
        if (error_code != cudaSuccess) { \
            LOG_ERROR("CUDA Error " << error_code << ": " << cudaGetErrorString(error_code)); \
            throw std::runtime_error("CUDA error"); \
        } \
    }

// CUDA kernel function: convert YOLO detection box format to physical coordinates [x,y,w,h] -> [x1,y1,x2,y2]
__global__ void transformBoxesKernel(
    float* boxes,        // [num_boxes, 4] 格式 (x,y,w,h)
    int num_boxes,
    int img_width,
    int img_height,
    int input_width,
    int input_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    // Get current box pointer (x,y,w,h format)
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
    
    // Calculate scale and offset
    float scale = min(float(input_width) / img_width, float(input_height) / img_height);
    int offsetx = (input_width - img_width * scale) / 2;
    int offsety = (input_height - img_height * scale) / 2;
    
    // Convert to top-left and bottom-right coordinates (standard YOLO output with letterbox offset)
    float x1 = (x - w/2.0f - offsetx) / scale;
    float y1 = (y - h/2.0f - offsety) / scale;
    float x2 = (x + w/2.0f - offsetx) / scale;
    float y2 = (y + h/2.0f - offsety) / scale;
    
    // Limit within image range
    x1 = max(0.0f, min(float(img_width), x1));
    y1 = max(0.0f, min(float(img_height), y1));
    x2 = max(0.0f, min(float(img_width), x2));
    y2 = max(0.0f, min(float(img_height), y2));
    
    // Check converted coordinates validity
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
__device__ float computeIoU(float* box1, float* box2) {
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
__global__ void nmsKernel(
    float* boxes,         // [num_boxes, 4] 格式 (x1,y1,x2,y2)
    float* scores,        // [num_boxes]
    int* indices,         // [num_boxes] input is index, output is index
    int* count,           // atomic counter, used to record number of retained boxes
    int num_boxes,
    float threshold
) {
    // Get current index being processed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    // Get current box index
    int box_idx = indices[idx];
    
    // If index is marked as -1, it means it has been suppressed, skip
    if (box_idx == -1) return;
    
    // Get current box
    float* current_box = boxes + box_idx * 4;
    float current_score = scores[box_idx];
    
    // Compute IoU with subsequent all boxes and suppress overlapping boxes
    for (int i = idx + 1; i < num_boxes; i++) {
        int other_idx = indices[i];
        if (other_idx == -1) continue;  // Has been suppressed
        
        float* other_box = boxes + other_idx * 4;
        float iou = computeIoU(current_box, other_box);
        
        if (iou > threshold) {
            // Atomic operation, mark suppressed box as -1
            atomicExch(&indices[i], -1);
        }
    }
}

// GPU postprocess implementation
void transformBoxesGPU(
    float* d_boxes,
    int num_boxes,
    const cv::Mat& original_image,
    int input_width,
    int input_height
) {
    // Determine CUDA grid and block size
    int block_size = 256;
    int grid_size = (num_boxes + block_size - 1) / block_size;
    
    // Call CUDA kernel function to transform box coordinates
    transformBoxesKernel<<<grid_size, block_size>>>(
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
void nmsGPU(float* boxes, float* scores, int* indices, int& nboxes, int boxes_num, float threshold) {
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
    
    nmsKernel<<<grid_size, block_size>>>(
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

// CUDA kernel function: extract valid detections (global model)
__global__ void extractDetectionsKernel(
    float* output,        // YOLO original output [5, num_boxes]
    int num_boxes,
    float conf_threshold,
    float* valid_boxes,   // Output valid boxes [max_detections, 5]
    int* valid_count      // Atomic counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;
    
    float x = output[0 * num_boxes + idx];
    float y = output[1 * num_boxes + idx];
    float w = output[2 * num_boxes + idx];
    float h = output[3 * num_boxes + idx];
    float conf = output[4 * num_boxes + idx];
    
    // Filter low confidence and unreasonable boxes
    if (conf < conf_threshold || conf > 1.0f) return;
    if (w <= 0 || h <= 0 || w > kInputW * 2 || h > kInputH * 2) return;
    
    // Atomic increase counter, get current box storage position
    int pos = atomicAdd(valid_count, 1);
    
    // Store detection box (x,y,w,h,conf)
    valid_boxes[pos * 5 + 0] = x;
    valid_boxes[pos * 5 + 1] = y;
    valid_boxes[pos * 5 + 2] = w;
    valid_boxes[pos * 5 + 3] = h;
    valid_boxes[pos * 5 + 4] = conf;
}

// Parse YOLO output and execute NMS on GPU (global detection)
std::vector<Detection> parseYOLOOutputGPU(
    float* output,
    int output_size,
    const cv::Mat& original_image,
    float conf_threshold,
    float nms_threshold
) {
    std::vector<Detection> detections;
    
    try {
        // Parameter validation
        if (output == nullptr) {
            LOG_ERROR("YOLO output pointer is null");
            return detections;
        }
        
        if (output_size <= 0) {
            LOG_ERROR("Invalid output size: " << output_size);
            return detections;
        }
        
        if (original_image.empty()) {
            LOG_ERROR("Original image is empty");
            return detections;
        }
        
        // Calculate number of boxes
        int num_boxes = output_size / kBoxInfoSize;
        int max_detections = num_boxes;
        
        // Allocate temporary GPU memory
        float* d_output;
        float* d_valid_boxes;  // Store valid detections [x,y,w,h,conf]
        int* d_valid_count;    // Valid box counter
        
        // Allocate and copy input
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_output, output, output_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Allocate memory for valid detections (最多保留max_detections个框）
        CUDA_CHECK(cudaMalloc(&d_valid_boxes, max_detections * 5 * sizeof(float)));  // 每框5个值
        
        // Initialize valid detection memory to 0
        CUDA_CHECK(cudaMemset(d_valid_boxes, 0, max_detections * 5 * sizeof(float)));
        
        // Initialize counter to 0
        int valid_count = 0;
        CUDA_CHECK(cudaMalloc(&d_valid_count, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_valid_count, &valid_count, sizeof(int), cudaMemcpyHostToDevice));
        
        // Extract valid detections
        int block_size = 256;
        int grid_size = (num_boxes + block_size - 1) / block_size;
        
        // Extract valid boxes
        extractDetectionsKernel<<<grid_size, block_size>>>(
            d_output, num_boxes, conf_threshold, d_valid_boxes, d_valid_count
        );
        
        // Check CUDA error
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            LOG_ERROR("CUDA kernel function execution error: " << cudaGetErrorString(cuda_status));
            
            // Free resources
            cudaFree(d_output);
            cudaFree(d_valid_boxes);
            cudaFree(d_valid_count);
            return detections;
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get number of valid boxes
        CUDA_CHECK(cudaMemcpy(&valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Ensure不超过最大检测框数量
        valid_count = std::min(valid_count, max_detections);
        
        if (valid_count == 0) {
            // No valid detections, free resources and return empty result
            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaFree(d_valid_boxes));
            CUDA_CHECK(cudaFree(d_valid_count));
            return detections;
        }
        
        // Remove GPU extraction box statistics log
        
        // Allocate host memory and copy valid boxes
        std::vector<float> h_valid_boxes(valid_count * 5);
        CUDA_CHECK(cudaMemcpy(h_valid_boxes.data(), d_valid_boxes, valid_count * 5 * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Extract coordinates and confidence
        std::vector<float> boxes(valid_count * 4);  // x,y,w,h -> x1,y1,x2,y2
        std::vector<float> scores(valid_count);
        
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
                LOG_WARNING("Invalid coordinates detected, index: " << i);
                continue;
            }
            
            // Copy original coordinates
            boxes[i*4 + 0] = h_valid_boxes[i*5 + 0];  // x
            boxes[i*4 + 1] = h_valid_boxes[i*5 + 1];  // y
            boxes[i*4 + 2] = h_valid_boxes[i*5 + 2];  // w
            boxes[i*4 + 3] = h_valid_boxes[i*5 + 3];  // h
            scores[i] = h_valid_boxes[i*5 + 4];       // conf
        }
        
        // Ensure valid data
        if (boxes.empty()) {
            // No valid boxes, free resources and return empty result
            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaFree(d_valid_boxes));
            CUDA_CHECK(cudaFree(d_valid_count));
            return detections;
        }
        
        // Convert box coordinates to original image space
        float* d_boxes;
        CUDA_CHECK(cudaMalloc(&d_boxes, valid_count * 4 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_boxes, boxes.data(), valid_count * 4 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Convert box coordinates
        transformBoxesGPU(d_boxes, valid_count, original_image, kInputW, kInputH);
        
        // Copy converted coordinates back to host
        CUDA_CHECK(cudaMemcpy(boxes.data(), d_boxes, valid_count * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Additional check for converted coordinates validity
        for (int i = 0; i < valid_count; i++) {
            for (int j = 0; j < 4; j++) {
                if (!std::isfinite(boxes[i*4 + j])) {
                    // If coordinate is invalid, set to invalid score
                    scores[i] = -1.0f;
                    break;
                }
            }
        }
        
        // Execute NMS
        std::vector<int> indices(valid_count);
        int num_after_nms = 0;
        nmsGPU(boxes.data(), scores.data(), indices.data(), num_after_nms, valid_count, nms_threshold);
        
        // Remove NMS result statistics log
        
        // Create final detection results
        for (int i = 0; i < valid_count; i++) {
            if (indices[i] != -1) {
                int box_idx = indices[i];
                
                // Additional check for validity
                if (scores[box_idx] < 0) continue;  // Skip marked as invalid boxes
                
                float x1 = boxes[box_idx*4 + 0];
                float y1 = boxes[box_idx*4 + 1];
                float x2 = boxes[box_idx*4 + 2];
                float y2 = boxes[box_idx*4 + 3];
                
                // Check coordinate range
                if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 || 
                    x1 > original_image.cols || y1 > original_image.rows ||
                    x2 > original_image.cols || y2 > original_image.rows) {
                    continue;  // Skip boxes超出范围的框
                }
                
                float width = x2 - x1;
                float height = y2 - y1;
                float conf = scores[box_idx];
                
                // Check width and height validity
                if (width <= 0 || height <= 0) continue;
                
                // For global model, set minimum box size threshold
                float min_width = 5.0f;
                float min_height = 5.0f;
                
                // Check boundary box validity
                if (width > min_width && height > min_height) {
                    // Additional check for coordinate validity 
                    if (std::isfinite(x1) && std::isfinite(y1) && std::isfinite(width) && std::isfinite(height)) {
                        Detection det;
                        det.bbox = cv::Rect2f(x1, y1, width, height);
                        det.confidence = conf;
                        det.class_id = 0;
                        det.is_from_global_model = true;  // Mark as global model detection result
                        
                        detections.push_back(det);
                    }
                }
            }
        }
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_valid_boxes));
        CUDA_CHECK(cudaFree(d_valid_count));
        CUDA_CHECK(cudaFree(d_boxes));
        
    } catch (const std::exception& e) {
        LOG_ERROR("GPU postprocess failed: " << e.what() << ", fallback to CPU processing");
        // Here we can fallback to CPU processing, but for now we return empty result directly
    }
    
    return detections;
}

} // namespace gpu
} // namespace tracking 