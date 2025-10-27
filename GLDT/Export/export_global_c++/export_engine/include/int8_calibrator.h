#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstdint>
#include <opencv2/opencv.hpp>

// ==================== Preprocessing Function Declarations ====================
// These functions are used for image preprocessing during INT8 calibration

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height);
void cuda_batch_preprocess(std::vector<cv::Mat>& img_batch,
                           float* dst, int dst_width, int dst_height);
void process_input_gpu(cv::Mat& input_img, float* input_device_buffer);
void process_input_cv_affine(cv::Mat& src, float* input_device_buffer);
void process_input_cpu(cv::Mat& src, float* input_device_buffer);

// ==================== INT8 Calibrator ====================

/**
 * @brief INT8 calibrator class for TensorRT INT8 quantization
 * 
 * This class implements the IInt8EntropyCalibrator2 interface of TensorRT,
 * providing calibration data during model quantization
 */
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    /**
     * @brief Constructor
     * @param calibDataDir Calibration data directory path
     * @param calibListFile Calibration data list file path
     * @param batchSize Batch size
     * @param inputW Input image width
     * @param inputH Input image height
     * @param cacheFile Calibration cache file path
     * @param useLetterbox Whether to use letterbox preprocessing
     * @param useBgr2Rgb Whether to convert BGR to RGB
     * @param calibLimit Calibration image count limit (0 means no limit)
     */
    Int8Calibrator(const std::string& calibDataDir,
                   const std::string& calibListFile,
                   int batchSize,
                   int inputW,
                   int inputH,
                   const std::string& cacheFile = "calibration.cache",
                   bool useLetterbox = true,
                   bool useBgr2Rgb = true,
                   int calibLimit = 0);

    /**
     * @brief Destructor
     */
    virtual ~Int8Calibrator();

    /**
     * @brief Get batch size
     * @return Batch size
     */
    int getBatchSize() const noexcept override;

    /**
     * @brief Get a batch of calibration data
     * @param names Input tensor name array
     * @param nbBindings Number of binding points
     * @return Pointer array to device memory, or nullptr if no more data
     */
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    /**
     * @brief Read calibration cache
     * @param length Length of cache data
     * @return Pointer to cache data, or nullptr if no cache exists
     */
    const void* readCalibrationCache(size_t& length) noexcept override;

    /**
     * @brief Write calibration cache
     * @param cache Cache data pointer
     * @param length Length of cache data
     */
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    /**
     * @brief Preprocess a single image
     * @param imagePath Image file path
     * @param deviceBuffer Device memory buffer
     * @return Whether preprocessing was successful
     */
    bool preprocessImage(const std::string& imagePath, float* deviceBuffer);

    /**
     * @brief Apply letterbox transformation
     * @param src Source image
     * @return Transformed image
     */
    cv::Mat applyLetterbox(const cv::Mat& src);

    /**
     * @brief Initialize CUDA memory
     */
    void initializeCudaMemory();

    /**
     * @brief Free CUDA memory
     */
    void freeCudaMemory();

private:
    std::string mCalibDataDir;          ///< Calibration data directory
    std::string mCalibListFile;         ///< Calibration data list file
    std::string mCacheFile;             ///< Calibration cache file
    int mBatchSize;                     ///< Batch size
    int mInputW;                        ///< Input width
    int mInputH;                        ///< Input height
    bool mUseLetterbox;                 ///< Whether to use letterbox
    bool mUseBgr2Rgb;                   ///< Whether to convert BGR to RGB
    int mCalibLimit;                    ///< Calibration image count limit
    
    std::vector<std::string> mImageFiles; ///< Image file list
    int mCurrentBatch;                    ///< Current batch index
    
    void* mDeviceInput;                   ///< Device input memory (current_frame)
    void* mDeviceInput2;                  ///< Device input memory (previous_frame)
    std::vector<char> mCalibrationCache;  ///< Calibration cache data
    
    size_t mInputSize;                    ///< Size of single input (bytes)
    bool mCudaMemoryInitialized;          ///< Whether CUDA memory is initialized
};
