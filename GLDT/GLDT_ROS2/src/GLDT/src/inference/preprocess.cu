#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <memory>

// CUDA error check macro - fix: use throw rather than assert
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error_code) << std::endl;\
            throw std::runtime_error("CUDA error");\
        }\
    }
#endif

// Constant definition - consistent with constants in TensorRTInference.cpp
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// 🔥 Thread-safe preprocess context - each thread has independent resource management
struct ThreadPreprocessContext {
    uint8_t* device_buffer = nullptr;
    size_t buffer_capacity = 0;
    float* staging_buffer = nullptr;   // Staging buffer for storing preprocess results
    size_t staging_capacity = 0;
    cudaStream_t stream = nullptr;
    bool initialized = false;
    std::mutex context_mutex;
    
    ~ThreadPreprocessContext() {
        cleanup();
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(context_mutex);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (device_buffer) {
            cudaFree(device_buffer);
            device_buffer = nullptr;
        }
        if (staging_buffer) {
            cudaFree(staging_buffer);
            staging_buffer = nullptr;
        }
        buffer_capacity = 0;
        staging_capacity = 0;
        initialized = false;
    }
    
    bool ensure_buffer(size_t required_size) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        if (!initialized) {
            // Create independent CUDA stream
            CUDA_CHECK(cudaStreamCreate(&stream));
            initialized = true;
        }
        
        if (required_size <= buffer_capacity && device_buffer) {
            return true;
        }
        
        // Wait for current stream to complete
        if (stream) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Release old buffer
        if (device_buffer) {
            CUDA_CHECK(cudaFree(device_buffer));
            device_buffer = nullptr;
        }
        
        // Allocate new buffer
        cudaError_t status = cudaMalloc(&device_buffer, required_size);
        if (status != cudaSuccess) {
            std::cerr << "❌ [Thread" << std::this_thread::get_id() << "] Allocate buffer failed: " 
                      << cudaGetErrorString(status) << std::endl;
            buffer_capacity = 0;
            return false;
        }
        
        buffer_capacity = required_size;
        return true;
    }

    bool ensure_staging(size_t required_size) {
        std::lock_guard<std::mutex> lock(context_mutex);

        if (!initialized) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            initialized = true;
        }

        if (required_size <= staging_capacity && staging_buffer) {
            return true;
        }

        if (stream) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        if (staging_buffer) {
            CUDA_CHECK(cudaFree(staging_buffer));
            staging_buffer = nullptr;
        }

        cudaError_t status = cudaMalloc(&staging_buffer, required_size);
        if (status != cudaSuccess) {
            std::cerr << "❌ [Thread" << std::this_thread::get_id() << "] Allocate staging buffer failed: "
                      << cudaGetErrorString(status) << std::endl;
            staging_capacity = 0;
            return false;
        }

        staging_capacity = required_size;
        return true;
    }
};

// 🔥 Thread-safe context manager
class ThreadSafePreprocessManager {
private:
    static std::mutex manager_mutex_;
    static std::unordered_map<std::thread::id, std::unique_ptr<ThreadPreprocessContext>> contexts_;
    
public:
    static ThreadPreprocessContext* get_context() {
        std::thread::id tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(manager_mutex_);
        
        auto it = contexts_.find(tid);
        if (it == contexts_.end()) {
            contexts_[tid] = std::make_unique<ThreadPreprocessContext>();
        }
        
        return contexts_[tid].get();
    }
    
    static void cleanup_thread() {
        std::thread::id tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(manager_mutex_);
        
        auto it = contexts_.find(tid);
        if (it != contexts_.end()) {
            it->second->cleanup();
            contexts_.erase(it);
        }
    }
    
    static void cleanup_all() {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        for (auto& pair : contexts_) {
            pair.second->cleanup();
        }
        contexts_.clear();
    }
};

// Static member definition
std::mutex ThreadSafePreprocessManager::manager_mutex_;
std::unordered_map<std::thread::id, std::unique_ptr<ThreadPreprocessContext>> ThreadSafePreprocessManager::contexts_;

// 🔥 Backward compatible global variable - but no longer recommended
static uint8_t *img_buffer_device = nullptr;
static bool cuda_initialized = false;
static size_t img_buffer_capacity = 0;
static std::mutex g_cuda_pre_mutex;

// Affine transformation matrix structure
struct AffineMatrix {
    float value[6];
};

// CUDA kernel function: preprocess (normalization, BGR2RGB, NHWC to NCHW)
__global__ void preprocess_kernel(
    uint8_t *src, float *dst, int dst_width,
    int dst_height, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge)
        return;

    int dx = position % dst_width;
    int dy = position / dst_width;

    // normalization
    float c0 = src[dy * dst_width * 3 + dx * 3 + 0] / 255.0f;
    float c1 = src[dy * dst_width * 3 + dx * 3 + 1] / 255.0f;
    float c2 = src[dy * dst_width * 3 + dx * 3 + 2] / 255.0f;

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // rgbrgbrgb to rrrgggbbb (NHWC to NCHW)
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// CUDA kernel function: affine transformation
__global__ void warpaffine_kernel(
    uint8_t *src, int src_line_size, int src_width,
    int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge)
        return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;

    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t *v1 = const_value;
        uint8_t *v2 = const_value;
        uint8_t *v3 = const_value;
        uint8_t *v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // rgbrgbrgb to rrrgggbbb (NHWC to NCHW)
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// letterbox helper function (CPU side)
inline cv::Mat letterbox_cpu(const cv::Mat& src) {
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

int cuda_preprocess_thread_safe(uint8_t *src, int src_width, int src_height,
                                float *dst, int dst_width, int dst_height) {
    try {
        ThreadPreprocessContext* ctx = ThreadSafePreprocessManager::get_context();
        if (!ctx) {
            std::cerr << "❌ Get thread context failed" << std::endl;
            return -1;
        }
        
        int img_size = src_width * src_height * 3;
        if (!ctx->ensure_buffer(img_size)) {
            std::cerr << "❌ Allocate buffer failed" << std::endl;
            return -1;
        }

        // Allocate staging buffer for output (float)
        size_t output_bytes = dst_width * dst_height * 3 * sizeof(float);
        if (!ctx->ensure_staging(output_bytes)) {
            std::cerr << "❌ Allocate staging buffer failed" << std::endl;
            return -1;
        }
        
        // Use thread-independent stream for asynchronous memory copy
        CUDA_CHECK(cudaMemcpyAsync(ctx->device_buffer, src, img_size, 
                                  cudaMemcpyHostToDevice, ctx->stream));
        
        // Calculate transformation matrix
        AffineMatrix s2d, d2s;
        float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

        s2d.value[0] = scale;
        s2d.value[1] = 0;
        s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
        s2d.value[3] = 0;
        s2d.value[4] = scale;
        s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

        cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
        cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
        cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

        memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

        // Start kernel on thread-independent stream
        int jobs = dst_height * dst_width;
        int threads = 256;
        int blocks = (jobs + threads - 1) / threads;

        // First write to staging_buffer, avoid race condition when writing to TensorRT input memory
        warpaffine_kernel<<<blocks, threads, 0, ctx->stream>>>(
            ctx->device_buffer, src_width * 3, src_width,
            src_height, ctx->staging_buffer, dst_width,
            dst_height, 128, d2s, jobs);
        
        CUDA_CHECK(cudaGetLastError());

        // Copy result to actual dst (TensorRT input memory)
        CUDA_CHECK(cudaMemcpyAsync(dst, ctx->staging_buffer, output_bytes,
                                  cudaMemcpyDeviceToDevice, ctx->stream));

        // Synchronize stream to ensure completion
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ [Thread-safe preprocess] Exception: " << e.what() << std::endl;
        return -1;
    }
}

// GPU do normalization, BGR2RGB, NHWC to NCHW
void cuda_pure_preprocess(uint8_t *src, float *dst, int dst_width, int dst_height) {
    // 🔥 Changed to use thread-safe version
    ThreadPreprocessContext* ctx = ThreadSafePreprocessManager::get_context();
    if (!ctx) {
        std::cerr << "❌ Get thread context failed" << std::endl;
        return;
    }
    
    int img_size = dst_width * dst_height * 3;
    if (!ctx->ensure_buffer(img_size)) {
        std::cerr << "❌ Allocate buffer failed" << std::endl;
        return;
    }

    size_t output_bytes = dst_width * dst_height * 3 * sizeof(float);
    if (!ctx->ensure_staging(output_bytes)) {
        std::cerr << "❌ Allocate staging buffer failed" << std::endl;
        return;
    }
    
    CUDA_CHECK(cudaMemcpyAsync(ctx->device_buffer, src, img_size, 
                              cudaMemcpyHostToDevice, ctx->stream));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = (jobs + threads - 1) / threads;

    preprocess_kernel<<<blocks, threads, 0, ctx->stream>>>(
        ctx->device_buffer, ctx->staging_buffer, dst_width, dst_height, jobs);
    
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(dst, ctx->staging_buffer, output_bytes,
                              cudaMemcpyDeviceToDevice, ctx->stream));

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

// 🔥 Changed: use thread-safe version
void cuda_preprocess(uint8_t *src, int src_width, int src_height,
                    float *dst, int dst_width, int dst_height) {
    int result = cuda_preprocess_thread_safe(src, src_width, src_height, 
                                           dst, dst_width, dst_height);
    if (result != 0) {
        throw std::runtime_error("Thread-safe preprocess failed");
    }
}

// External interface function
extern "C" {

// 🔥 Global initialization function
void cuda_preprocess_init(int max_image_size) {
    // For backward compatibility, still set flag
    cuda_initialized = true;
}

// 🔥 Global destruction function
void cuda_preprocess_destroy() {
    ThreadSafePreprocessManager::cleanup_all();
    cuda_initialized = false;
}

// Use GPU to do all preprocess steps
void process_input_gpu(cv::Mat& src, float* input_device_buffer) {
    try {
        int result = cuda_preprocess_thread_safe(src.ptr(), src.cols, src.rows, 
                                                input_device_buffer, kInputW, kInputH);
        if (result != 0) {
            throw std::runtime_error("GPU preprocess failed");
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ [process_input_gpu] Exception: " << e.what() << std::endl;
        throw;
    }
}

// Use CPU to do letterbox, GPU to do normalization, BGR2RGB, NHWC to NCHW
void process_input_cv_affine(cv::Mat& src, float* input_device_buffer) {
    try {
        auto warp_dst = letterbox_cpu(src);
        cuda_pure_preprocess(warp_dst.ptr(), input_device_buffer, kInputW, kInputH);
    } catch (const std::exception& e) {
        std::cerr << "❌ [process_input_cv_affine] Exception: " << e.what() << std::endl;
        throw;
    }
}

// Use CPU to do letterbox, normalization, BGR2RGB, NHWC to NCHW
void process_input_cpu(cv::Mat& src, float* input_device_buffer) {
    auto warp_dst = letterbox_cpu(src);
    warp_dst.convertTo(warp_dst, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(warp_dst, warp_dst, cv::COLOR_BGR2RGB);
    
    std::vector<cv::Mat> warp_dst_nchw_channels;
    cv::split(warp_dst, warp_dst_nchw_channels);

    for (auto &img : warp_dst_nchw_channels) {
        img = img.reshape(1, 1);
    }
    
    cv::Mat warp_dst_nchw;
    cv::hconcat(warp_dst_nchw_channels, warp_dst_nchw);
    
    CUDA_CHECK(cudaMemcpy(input_device_buffer, warp_dst_nchw.ptr(), 
                         kInputH * kInputW * 3 * sizeof(float), cudaMemcpyHostToDevice));
}


void cuda_preprocess_roi_safe(uint8_t* src, int src_width, int src_height,
                             float* dst, int dst_width, int dst_height,
                             void* temp_buffer, cudaStream_t stream) {
    int img_size = src_width * src_height * 3;
    
    CUDA_CHECK(cudaMemcpyAsync(temp_buffer, src, img_size, 
                              cudaMemcpyHostToDevice, stream));
    
    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = (jobs + threads - 1) / threads;

    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t*>(temp_buffer), src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);
    
    CUDA_CHECK(cudaGetLastError());
}

//simplified ROI preprocess interface
void process_roi_gpu(const cv::Mat& roi_in, float* output, void*, cudaStream_t) {
    // Use generic thread-safe preprocess function
    cv::Mat roi;
    if (roi_in.isContinuous()) {
        roi = roi_in;
    } else {
        roi = roi_in.clone();
    }

    int ret = cuda_preprocess_thread_safe(roi.ptr(), roi.cols, roi.rows,
                                         output, kInputW, kInputH);
    if (ret != 0) {
        throw std::runtime_error("ROI preprocess failed");
    }
}

// 🔥 Support GPU preprocess with specified stream (for multi-stream parallel)
void process_input_gpu_stream(const cv::Mat& src, float* output, void* temp_buffer, cudaStream_t stream) {
    try {
        // Ensure image is continuous
        cv::Mat image;
        if (src.isContinuous()) {
            image = src;
        } else {
            image = src.clone();
        }
        
        // Use cuda_preprocess_roi_safe, it supports stream parameter
        cuda_preprocess_roi_safe(image.ptr(), image.cols, image.rows,
                                output, kInputW, kInputH,
                                temp_buffer, stream);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ [process_input_gpu_stream] Exception: " << e.what() << std::endl;
        throw;
    }
}

// thread cleanup function
void cleanup_current_thread() {
    ThreadSafePreprocessManager::cleanup_thread();
}

// } // extern "C" (commented 0)
} // extern "C"