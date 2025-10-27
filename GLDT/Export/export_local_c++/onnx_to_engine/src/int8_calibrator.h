#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const std::string& calibListFile, 
                         const std::string& calibDataPath,
                         int batchSize, 
                         int inputH, 
                         int inputW, 
                         int inputC = 3,
                         const std::string& calibTableName = "calibration.table");

    ~Int8EntropyCalibrator();

    // Override base class methods
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    std::vector<std::string> imageFiles_;
    std::string calibDataPath_;
    std::string calibTableName_;
    
    int batchSize_;
    int inputH_;
    int inputW_;
    int inputC_;
    int imageIndex_;
    
    size_t inputSize_;
    void* deviceInput_;
    std::vector<float> hostInput_;
    
    std::vector<char> calibrationCache_;
    
    // Preprocess image
    void preprocessImage(const cv::Mat& img, float* buffer, int idx);
    
    // Load calibration image list
    bool loadCalibrationList(const std::string& calibListFile);
};

#endif // INT8_CALIBRATOR_H 