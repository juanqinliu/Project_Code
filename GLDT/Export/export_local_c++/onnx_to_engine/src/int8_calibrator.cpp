#include "int8_calibrator.h"
#include <cassert>
#include <stdexcept>
#include <string>

Int8EntropyCalibrator::Int8EntropyCalibrator(const std::string& calibListFile, 
                                           const std::string& calibDataPath,
                                           int batchSize, 
                                           int inputH, 
                                           int inputW, 
                                           int inputC,
                                           const std::string& calibTableName)
    : calibDataPath_(calibDataPath)
    , calibTableName_(calibTableName)
    , batchSize_(batchSize)
    , inputH_(inputH)
    , inputW_(inputW)
    , inputC_(inputC)
    , imageIndex_(0)
    , deviceInput_(nullptr) {
    
    // First check if calibration cache file exists
    std::ifstream cacheFile(calibTableName_);
    if (cacheFile.good()) {
        std::cout << "Found existing calibration cache file: " << calibTableName_ << std::endl;
        std::cout << "Skipping image loading, will use cache file directly" << std::endl;
        cacheFile.close();
        
        // Calculate input size (still needed for GPU memory allocation)
        inputSize_ = batchSize_ * inputC_ * inputH_ * inputW_ * sizeof(float);
        
        // Allocate GPU memory
        cudaError_t status = cudaMalloc(&deviceInput_, inputSize_);
        if (status != cudaSuccess) {
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(status) << std::endl;
            return;
        }
        
        // Allocate host memory
        hostInput_.resize(batchSize_ * inputC_ * inputH_ * inputW_);
        
        std::cout << "INT8 calibrator initialization complete (using cache mode)" << std::endl;
        std::cout << "Batch size: " << batchSize_ << std::endl;
        std::cout << "Input dimensions: " << inputC_ << "x" << inputH_ << "x" << inputW_ << std::endl;
        return;
    }
    
    std::cout << "No calibration cache file found, will perform new calibration" << std::endl;
    
    // Load calibration image list
    if (!loadCalibrationList(calibListFile)) {
        std::cerr << "Failed to load calibration list: " << calibListFile << std::endl;
        return;
    }
    
    std::cout << "Loaded " << imageFiles_.size() << " calibration images" << std::endl;
    
    // Calculate input size
    inputSize_ = batchSize_ * inputC_ * inputH_ * inputW_ * sizeof(float);
    
    // Allocate GPU memory
    cudaError_t status = cudaMalloc(&deviceInput_, inputSize_);
    if (status != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(status) << std::endl;
        return;
    }
    
    // Allocate host memory
    hostInput_.resize(batchSize_ * inputC_ * inputH_ * inputW_);
    
    std::cout << "INT8 calibrator initialization complete" << std::endl;
    std::cout << "Batch size: " << batchSize_ << std::endl;
    std::cout << "Input dimensions: " << inputC_ << "x" << inputH_ << "x" << inputW_ << std::endl;
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    if (deviceInput_) {
        cudaFree(deviceInput_);
    }
}

bool Int8EntropyCalibrator::loadCalibrationList(const std::string& calibListFile) {
    std::ifstream file(calibListFile);
    if (!file.is_open()) {
        std::cerr << "Unable to open calibration list file: " << calibListFile << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            imageFiles_.push_back(line);
        }
    }
    
    file.close();
    
    // Randomly shuffle image order for better calibration results
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(imageFiles_.begin(), imageFiles_.end(), g);
    
    return !imageFiles_.empty();
}

int Int8EntropyCalibrator::getBatchSize() const noexcept {
    return batchSize_;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    // If no image file list, using cache mode, return false directly
    if (imageFiles_.empty()) {
        std::cout << "Using calibration cache mode, skipping batch processing" << std::endl;
        return false;
    }
    
    if (imageIndex_ + batchSize_ > static_cast<int>(imageFiles_.size())) {
        return false; // No more data
    }
    
    // Clear host buffer
    std::fill(hostInput_.begin(), hostInput_.end(), 0.0f);
    
    // Load and preprocess a batch of images
    int validImages = 0;
    for (int i = 0; i < batchSize_; ++i) {
        if (imageIndex_ + i >= static_cast<int>(imageFiles_.size())) {
            break;
        }
        
        std::string imagePath = calibDataPath_ + "/" + imageFiles_[imageIndex_ + i];
        cv::Mat img = cv::imread(imagePath);
        
        if (img.empty()) {
            std::cerr << "Unable to load image: " << imagePath << std::endl;
            continue;
        }
        
        // Validate image dimensions
        if (img.rows == 0 || img.cols == 0 || img.channels() != 3) {
            std::cerr << "Invalid image format: " << imagePath << std::endl;
            continue;
        }
        
        try {
            preprocessImage(img, hostInput_.data(), validImages);
            validImages++;
        } catch (const std::exception& e) {
            std::cerr << "Image preprocessing failed: " << imagePath << ", error: " << e.what() << std::endl;
            continue;
        }
    }
    
    if (validImages == 0) {
        std::cerr << "No valid images in batch" << std::endl;
        return false;
    }
    
    // Copy data to GPU with error checking
    cudaError_t status = cudaMemcpy(deviceInput_, hostInput_.data(), inputSize_, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Synchronize CUDA operations
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << "CUDA sync failed: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    bindings[0] = deviceInput_;
    imageIndex_ += batchSize_;
    
    std::cout << "Calibration progress: " << imageIndex_ << "/" << imageFiles_.size() << " images (valid: " << validImages << ")" << std::endl;
    
    return true;
}

void Int8EntropyCalibrator::preprocessImage(const cv::Mat& img, float* buffer, int idx) {
    // Validate input parameters
    if (img.empty() || buffer == nullptr || idx < 0 || idx >= batchSize_) {
        throw std::runtime_error("Invalid input parameters for preprocessImage");
    }
    
    // Resize image
    cv::Mat resized;
    try {
        cv::resize(img, resized, cv::Size(inputW_, inputH_));
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to resize image: " + std::string(e.what()));
    }
    
    // Convert to RGB (OpenCV uses BGR)
    cv::Mat rgb;
    try {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to convert color: " + std::string(e.what()));
    }
    
    // Convert to float and normalize to [0, 1]
    cv::Mat normalized;
    try {
        rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to normalize image: " + std::string(e.what()));
    }
    
    // Rearrange dimensions: HWC -> CHW
    std::vector<cv::Mat> channels(inputC_);
    try {
        cv::split(normalized, channels);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to split channels: " + std::string(e.what()));
    }
    
    // Validate channel count
    if (channels.size() != static_cast<size_t>(inputC_)) {
        throw std::runtime_error("Channel count mismatch: expected " + std::to_string(inputC_) + 
                                ", got " + std::to_string(channels.size()));
    }
    
    int channelSize = inputH_ * inputW_;
    int imageOffset = idx * inputC_ * channelSize;
    
    // Boundary check
    int totalSize = batchSize_ * inputC_ * channelSize;
    if (imageOffset + inputC_ * channelSize > totalSize) {
        throw std::runtime_error("Buffer overflow detected in preprocessImage");
    }
    
    for (int c = 0; c < inputC_; ++c) {
        if (channels[c].empty() || channels[c].type() != CV_32F) {
            throw std::runtime_error("Invalid channel data at index " + std::to_string(c));
        }
        
        float* channelData = channels[c].ptr<float>();
        if (channelData == nullptr) {
            throw std::runtime_error("Null channel data at index " + std::to_string(c));
        }
        
        // Safe memory copy
        std::memcpy(buffer + imageOffset + c * channelSize, channelData, channelSize * sizeof(float));
    }
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    std::ifstream input(calibTableName_, std::ios::binary);
    if (!input) {
        length = 0;
        return nullptr;
    }
    
    input.seekg(0, std::ios::end);
    length = input.tellg();
    input.seekg(0, std::ios::beg);
    
    calibrationCache_.resize(length);
    input.read(calibrationCache_.data(), length);
    input.close();
    
    std::cout << "Read calibration cache: " << calibTableName_ << " (" << length << " bytes)" << std::endl;
    
    return calibrationCache_.data();
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream output(calibTableName_, std::ios::binary);
    if (!output) {
        std::cerr << "Unable to write calibration cache: " << calibTableName_ << std::endl;
        return;
    }
    
    output.write(static_cast<const char*>(cache), length);
    output.close();
    
    std::cout << "Write calibration cache: " << calibTableName_ << " (" << length << " bytes)" << std::endl;
} 