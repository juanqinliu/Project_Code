#include "int8_calibrator.h"
#include "config.h"
#include <iostream>
#include <algorithm>
#include <cstring>

Int8Calibrator::Int8Calibrator(const std::string& calibDataDir,
                               const std::string& calibListFile,
                               int batchSize,
                               int inputW,
                               int inputH,
                               const std::string& cacheFile,
                               bool useLetterbox,
                               bool useBgr2Rgb,
                               int calibLimit)
    : mCalibDataDir(calibDataDir)
    , mCalibListFile(calibListFile)
    , mCacheFile(cacheFile)
    , mBatchSize(batchSize)
    , mInputW(inputW)
    , mInputH(inputH)
    , mUseLetterbox(useLetterbox)
    , mUseBgr2Rgb(useBgr2Rgb)
    , mCalibLimit(calibLimit)
    , mCurrentBatch(0)
    , mDeviceInput(nullptr)
    , mDeviceInput2(nullptr)
    , mInputSize(0)
    , mCudaMemoryInitialized(false)
{
    // Read calibration data file list
    std::ifstream file(mCalibListFile);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open calibration list file: " << mCalibListFile << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            mImageFiles.push_back(line);
            // Check if limit is reached
            if (mCalibLimit > 0 && static_cast<int>(mImageFiles.size()) >= mCalibLimit) {
                break;
            }
        }
    }
    file.close();

    std::cout << "Calibrator initialized: " << mImageFiles.size() << " images" << std::endl;
    std::cout << "Batch size: " << mBatchSize << std::endl;
    std::cout << "Input size: " << mInputW << "x" << mInputH << std::endl;
    std::cout << "Use letterbox: " << (mUseLetterbox ? "Yes" : "No") << std::endl;
    std::cout << "BGR to RGB: " << (mUseBgr2Rgb ? "Yes" : "No") << std::endl;

    // Initialize CUDA memory
    initializeCudaMemory();
}

Int8Calibrator::~Int8Calibrator()
{
    freeCudaMemory();
}

void Int8Calibrator::initializeCudaMemory()
{
    if (mCudaMemoryInitialized) {
        return;
    }

    // Calculate size of single input (bytes)
    mInputSize = mBatchSize * 3 * mInputW * mInputH * sizeof(float);
    
    // Allocate device memory (for both inputs)
    CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputSize));
    CUDA_CHECK(cudaMalloc(&mDeviceInput2, mInputSize));
    
    mCudaMemoryInitialized = true;
    std::cout << "CUDA memory initialized, allocated size: " << (mInputSize * 2 / (1024.0 * 1024.0)) << " MB (dual inputs)" << std::endl;
}

void Int8Calibrator::freeCudaMemory()
{
    if (mCudaMemoryInitialized) {
        if (mDeviceInput) {
            CUDA_CHECK(cudaFree(mDeviceInput));
            mDeviceInput = nullptr;
        }
        if (mDeviceInput2) {
            CUDA_CHECK(cudaFree(mDeviceInput2));
            mDeviceInput2 = nullptr;
        }
        mCudaMemoryInitialized = false;
    }
}

int Int8Calibrator::getBatchSize() const noexcept
{
    return mBatchSize;
}

bool Int8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    // Check if there is more data
    if (mCurrentBatch >= static_cast<int>(mImageFiles.size())) {
        std::cout << "Calibration complete, processed " << mCurrentBatch << " images" << std::endl;
        return false;
    }

    try {
        // Process current batch of images
        std::vector<cv::Mat> batchImages;
        for (int i = 0; i < mBatchSize && (mCurrentBatch + i) < static_cast<int>(mImageFiles.size()); ++i) {
            std::string imagePath = mCalibDataDir + "/" + mImageFiles[mCurrentBatch + i];
            
            // Read image
            cv::Mat img = cv::imread(imagePath);
            if (img.empty()) {
                std::cerr << "Warning: Cannot read image: " << imagePath << std::endl;
                continue;
            }
            
            batchImages.push_back(img);
        }

        if (batchImages.empty()) {
            std::cerr << "Error: No valid images in current batch" << std::endl;
            return false;
        }

        // Preprocess batch images
        std::vector<float> hostData(mBatchSize * 3 * mInputW * mInputH, 0.0f);
        
        for (size_t i = 0; i < batchImages.size(); ++i) {
            cv::Mat processedImg = batchImages[i];
            
            // Apply letterbox transformation (if enabled)
            if (mUseLetterbox) {
                processedImg = applyLetterbox(processedImg);
            } else {
                // Direct resize
                cv::resize(processedImg, processedImg, cv::Size(mInputW, mInputH));
            }
            
            // Convert to float and normalize
            processedImg.convertTo(processedImg, CV_32FC3, 1.0 / 255.0);
            
            // BGR to RGB (if enabled)
            if (mUseBgr2Rgb) {
                cv::cvtColor(processedImg, processedImg, cv::COLOR_BGR2RGB);
            }
            
            // Convert to CHW format
            std::vector<cv::Mat> channels;
            cv::split(processedImg, channels);
            
            size_t channelSize = mInputW * mInputH;
            size_t imageOffset = i * 3 * channelSize;
            
            for (int c = 0; c < 3; ++c) {
                std::memcpy(&hostData[imageOffset + c * channelSize], 
                           channels[c].ptr<float>(), 
                           channelSize * sizeof(float));
            }
        }

        // Copy data to device memory (both inputs use the same data)
        CUDA_CHECK(cudaMemcpy(mDeviceInput, hostData.data(), mInputSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mDeviceInput2, hostData.data(), mInputSize, cudaMemcpyHostToDevice));
        
        // Set binding points (dual inputs: current_frame and previous_frame)
        bindings[0] = mDeviceInput;
        bindings[1] = mDeviceInput2;
        
        mCurrentBatch += mBatchSize;
        
        if (mCurrentBatch % 100 == 0) {
            std::cout << "Calibration progress: " << mCurrentBatch << "/" << mImageFiles.size() << std::endl;
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Calibration batch processing error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat Int8Calibrator::applyLetterbox(const cv::Mat& src)
{
    float scale = std::min(static_cast<float>(mInputH) / src.rows, 
                          static_cast<float>(mInputW) / src.cols);
    
    int newWidth = static_cast<int>(src.cols * scale);
    int newHeight = static_cast<int>(src.rows * scale);
    
    // Resize image
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newWidth, newHeight));
    
    // Create target image and fill
    cv::Mat result = cv::Mat::zeros(mInputH, mInputW, src.type());
    
    int offsetX = (mInputW - newWidth) / 2;
    int offsetY = (mInputH - newHeight) / 2;
    
    // Copy resized image to center of target image
    resized.copyTo(result(cv::Rect(offsetX, offsetY, newWidth, newHeight)));
    
    return result;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept
{
    mCalibrationCache.clear();
    
    std::ifstream input(mCacheFile, std::ios::binary);
    if (!input.is_open()) {
        std::cout << "Calibration cache file not found: " << mCacheFile << std::endl;
        length = 0;
        return nullptr;
    }
    
    input.seekg(0, std::ios::end);
    length = input.tellg();
    input.seekg(0, std::ios::beg);
    
    mCalibrationCache.resize(length);
    input.read(mCalibrationCache.data(), length);
    input.close();
    
    std::cout << "Read calibration cache: " << mCacheFile << " (size: " << length << " bytes)" << std::endl;
    return mCalibrationCache.data();
}

void Int8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::ofstream output(mCacheFile, std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Error: Cannot write calibration cache file: " << mCacheFile << std::endl;
        return;
    }
    
    output.write(static_cast<const char*>(cache), length);
    output.close();
    
    std::cout << "Write calibration cache: " << mCacheFile << " (size: " << length << " bytes)" << std::endl;
}
