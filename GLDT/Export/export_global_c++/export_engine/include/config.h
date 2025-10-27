#ifndef CONFIG_H
#define CONFIG_H

#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>

// ==================== Model Configuration ====================

// Input settings
constexpr int kInputH = 640;
constexpr int kInputW = 640;
constexpr const char* kInputTensorName = "current_frame";
constexpr const char* kPrevTensorName = "previous_frame";

// Output settings
constexpr const char* kOutputTensorName = "output0";

// Model parameters
constexpr float kConfThresh = 0.25f;
constexpr float kNmsThresh = 0.45f;

// Batch processing settings
constexpr int kBatchSize = 1;

// Use FP16 precision
constexpr bool kUseFp16 = false;

// ==================== CUDA Utilities ====================

// CUDA error checking macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#endif // CONFIG_H

