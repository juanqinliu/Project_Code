#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "config.h"
#include "int8_calibrator.h"

// Extract filename from file path (without extension)
std::string getBaseName(const std::string& filepath) {
    // Find last path separator
    size_t lastSlash = filepath.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? filepath : filepath.substr(lastSlash + 1);
    
    // Find last dot, remove extension
    size_t lastDot = filename.find_last_of(".");
    return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}

// Display help information
void showHelp() {
    std::cout << "TensorRT Engine Export Tool" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  ./export_engine <onnx_file> --precision <precision> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Required Arguments:" << std::endl;
    std::cout << "  <onnx_file>             ONNX model file path" << std::endl;
    std::cout << "  --precision <type>      Precision type: fp32, fp16, int8" << std::endl;
    std::cout << std::endl;
    std::cout << "General Options:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  --workspace-size <MB>   Workspace size in MB (default: 1024)" << std::endl;
    std::cout << std::endl;
    std::cout << "INT8 Calibration Options (only valid with --precision int8):" << std::endl;
    std::cout << "  --calib-data <path>     Calibration data directory (default: ./calib_data)" << std::endl;
    std::cout << "  --calib-list <path>     Calibration data list file (default: ./calib_list.txt)" << std::endl;
    std::cout << "  --batch-size <n>        Batch size (default: 1)" << std::endl;
    std::cout << "  --calib-limit <n>       Calibration image count limit (default: 0=unlimited)" << std::endl;
    std::cout << "  --no-letterbox          Disable letterbox preprocessing" << std::endl;
    std::cout << "  --no-bgr2rgb           Disable BGR to RGB conversion" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./export_engine model.onnx --precision fp32" << std::endl;
    std::cout << "  ./export_engine model.onnx --precision fp16" << std::endl;
    std::cout << "  ./export_engine model.onnx --precision int8 --calib-data ./calib_data --calib-limit 1000" << std::endl;
}

// Parse command line arguments
struct Config {
    std::string onnxPath;
    std::string precision = "";  // Required parameter, no default value
    std::string calibDataDir = "./calib_data";
    std::string calibListFile = "./calib_list.txt";
    int batchSize = 1;
    int calibLimit = 0;
    int workspaceSize = 1024;  // MB
    bool useLetterbox = true;
    bool useBgr2Rgb = true;
    bool showHelpFlag = false;
};

Config parseArgs(int argc, char** argv) {
    Config config;
    
    if (argc < 2) {
        config.showHelpFlag = true;
        return config;
    }
    
    // First parameter is ONNX file path
    config.onnxPath = argv[1];
    
    // Parse remaining arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            config.showHelpFlag = true;
            return config;
        }
        else if (arg == "--precision" && i + 1 < argc) {
            std::string precision = argv[++i];
            if (precision == "fp32" || precision == "fp16" || precision == "int8") {
                config.precision = precision;
            } else {
                std::cerr << "Error: Unsupported precision type: " << precision << std::endl;
                std::cerr << "Supported precision types: fp32, fp16, int8" << std::endl;
                config.showHelpFlag = true;
                return config;
            }
        }
        else if (arg == "--workspace-size" && i + 1 < argc) {
            config.workspaceSize = std::atoi(argv[++i]);
        }
        else if (arg == "--calib-data" && i + 1 < argc) {
            config.calibDataDir = argv[++i];
        }
        else if (arg == "--calib-list" && i + 1 < argc) {
            config.calibListFile = argv[++i];
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            config.batchSize = std::atoi(argv[++i]);
        }
        else if (arg == "--calib-limit" && i + 1 < argc) {
            config.calibLimit = std::atoi(argv[++i]);
        }
        else if (arg == "--no-letterbox") {
            config.useLetterbox = false;
        }
        else if (arg == "--no-bgr2rgb") {
            config.useBgr2Rgb = false;
        }
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            config.showHelpFlag = true;
            return config;
        }
    }
    
    // Check required parameters
    if (config.precision.empty()) {
        std::cerr << "Error: Must specify --precision argument" << std::endl;
        config.showHelpFlag = true;
    }
    
    return config;
}

// Main function
int main(int argc, char **argv)
{
    // Parse command line arguments
    Config config = parseArgs(argc, argv);
    
    if (config.showHelpFlag) {
        showHelp();
        return 0;
    }
    
    // Check if ONNX file exists
    std::ifstream onnxFile(config.onnxPath);
    if (!onnxFile.good()) {
        std::cerr << "Error: ONNX file does not exist: " << config.onnxPath << std::endl;
        return -1;
    }
    onnxFile.close();
    
    // If using INT8, check calibration data
    if (config.precision == "int8") {
        std::ifstream calibList(config.calibListFile);
        if (!calibList.good()) {
            std::cerr << "Error: Calibration list file does not exist: " << config.calibListFile << std::endl;
            return -1;
        }
        calibList.close();
    }
    
    std::cout << "\n=== Configuration Information ===" << std::endl;
    std::cout << "  ONNX file: " << config.onnxPath << std::endl;
    std::cout << "  Precision mode: " << config.precision << std::endl;
    std::cout << "  Workspace size: " << config.workspaceSize << " MB" << std::endl;
    
    if (config.precision == "int8") {
        std::cout << "  Calibration data directory: " << config.calibDataDir << std::endl;
        std::cout << "  Calibration list file: " << config.calibListFile << std::endl;
        std::cout << "  Batch size: " << config.batchSize << std::endl;
        std::cout << "  Calibration image limit: " << (config.calibLimit > 0 ? std::to_string(config.calibLimit) : "Unlimited") << std::endl;
        std::cout << "  Use letterbox: " << (config.useLetterbox ? "Yes" : "No") << std::endl;
        std::cout << "  BGR to RGB: " << (config.useBgr2Rgb ? "Yes" : "No") << std::endl;
    }

    // Create weights directory
    int mkdir_result = system("mkdir -p model_engine");
    if (mkdir_result != 0) {
        std::cerr << "Failed to create model_engine directory" << std::endl;
    }

    // Extract model name from ONNX file path
    std::string model_basename = getBaseName(config.onnxPath);
    std::string engine_file = "./model_engine/" + model_basename + "_" + config.precision + ".engine";
    
    std::cout << "Model name: " << model_basename << std::endl;
    std::cout << "Engine file: " << engine_file << std::endl;

    // =========== 1. Create builder ===========
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;
        return -1;
    }

    // ========== 2. Create network: builder--->network ==========
    // Explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // Call builder's createNetworkV2 method to create network
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cout << "Failed to create network" << std::endl;
        return -1;
    }

    // Create onnxparser for parsing onnx file
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    // Call onnxparser's parseFromFile method to parse onnx file
    auto parsed = parser->parseFromFile(config.onnxPath.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cout << "Failed to parse onnx file" << std::endl;
        return -1;
    }

    // Print network information
    std::cout << "\n=== Network Information ===" << std::endl;
    std::cout << "Number of inputs: " << network->getNbInputs() << std::endl;
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        std::cout << "Input " << i << ": " << input->getName() << ", Dimensions: ";
        auto dims = input->getDimensions();
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? "x" : "");
        }
        std::cout << std::endl;
    }

    std::cout << "Number of outputs: " << network->getNbOutputs() << std::endl;
    for (int i = 0; i < network->getNbOutputs(); i++) {
        auto output = network->getOutput(i);
        std::cout << "Output " << i << ": " << output->getName() << ", Dimensions: ";
        auto dims = output->getDimensions();
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? "x" : "");
        }
        std::cout << std::endl;
    }

    // Configure network parameters - support two inputs (current frame and previous frame)
    auto profile = builder->createOptimizationProfile();                                                           
    
    // Current frame input configuration
    auto current_frame = network->getInput(0);
    profile->setDimensions(current_frame->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, kInputH, kInputW}); 
    profile->setDimensions(current_frame->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, kInputH, kInputW}); 
    profile->setDimensions(current_frame->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, kInputH, kInputW}); 

    // Previous frame input configuration
    if (network->getNbInputs() > 1) {
        auto previous_frame = network->getInput(1);
        profile->setDimensions(previous_frame->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, kInputH, kInputW}); 
        profile->setDimensions(previous_frame->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, kInputH, kInputW}); 
        profile->setDimensions(previous_frame->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, kInputH, kInputW}); 
    }

    // ========== 3. Create config: builder--->config ==========
    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builderConfig)
    {
        std::cout << "Failed to create builderConfig" << std::endl;
        return -1;
    }
    
    // Use addOptimizationProfile method to add profile
    builderConfig->addOptimizationProfile(profile);

    // Set maximum workspace
    size_t workspaceBytes = static_cast<size_t>(config.workspaceSize) * 1024 * 1024; // MB to bytes
    builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes);

    // Reduce optimization level to reduce memory usage during tactic search
    // Reduce timing iterations to further reduce build time memory and time
    builderConfig->setMinTimingIterations(1);
    builderConfig->setAvgTimingIterations(1);

    // Limit tactic sources to avoid algorithms that consume too much memory (adjustable as needed)
    builderConfig->setTacticSources(1U << static_cast<int>(nvinfer1::TacticSource::kCUBLAS));

    // If DLA is available on Jetson, prioritize DLA and allow GPU fallback to further relieve GPU memory pressure
    if (builder->getNbDLACores() > 0) {
        builderConfig->setDLACore(0);
        builderConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }

    // Create stream
    cudaStream_t profileStream = samplesCommon::makeCudaStream();
    if (profileStream == nullptr)
    {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return -1;
    }
    builderConfig->setProfileStream(profileStream);

    // Create INT8 calibrator (if needed)
    std::unique_ptr<Int8Calibrator> calibrator;
    
    // Set precision
    if (config.precision == "int8")
    {
        if (!builder->platformHasFastInt8())
        {
            std::cout << "Warning: Device does not support INT8, falling back to FP16" << std::endl;
            config.precision = "fp16";
        }
        else
        {
            std::cout << "Enable INT8 precision" << std::endl;
            builderConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
            
            // Also enable FP16 as fallback
            if (builder->platformHasFastFp16())
            {
                builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            
            // Create calibrator
            calibrator = std::make_unique<Int8Calibrator>(
                config.calibDataDir,
                config.calibListFile,
                config.batchSize,
                kInputW,
                kInputH,
                "./model_engine/" + model_basename + "_calib.cache",
                config.useLetterbox,
                config.useBgr2Rgb,
                config.calibLimit
            );
            
            builderConfig->setInt8Calibrator(calibrator.get());
        }
    }
    
    if (config.precision == "fp16")
    {
        if (builder->platformHasFastFp16())
        {
            std::cout << "Enable FP16 precision" << std::endl;
            builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else
        {
            std::cout << "Device does not support FP16, falling back to FP32" << std::endl;
            config.precision = "fp32";
        }
    }
    
    if (config.precision == "fp32")
    {
        std::cout << "Use FP32 precision" << std::endl;
    }
        
        // ========== 4. Create engine: builder--->engine(*network, *builderConfig) ==========
    std::cout << "Start building engine..." << std::endl;
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *builderConfig));
        if (!plan)
        {
        std::cout << "Failed to create engine" << std::endl;
            return -1;
        }

        // ========== 5. Serialize and save engine ==========
    std::ofstream engine_file_stream(engine_file, std::ios::binary);
    if (!engine_file_stream)
    {
        std::cout << "Failed to open engine file: " << engine_file << std::endl;
        return -1;
    }
    
    engine_file_stream.write((char *)plan->data(), plan->size());
    if (engine_file_stream.fail())
    {
        std::cout << "Failed to write engine file: " << engine_file << std::endl;
        return -1;
    }
    
    engine_file_stream.close();
        
    std::cout << "Engine built successfully! Saved as " << engine_file << std::endl;
    
    // Print some useful information
    std::cout << "\n=== TensorRT Engine Information ===" << std::endl;
    std::cout << "Serialized engine size: " << (plan->size() / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Create an engine to get other information
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        std::cout << "Failed to create runtime" << std::endl;
        return -1;
    }
    
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine)
    {
        std::cout << "Failed to create engine" << std::endl;
        return -1;
    }
    
    std::cout << "Engine input/output binding points:" << std::endl;
    int numBindings = engine->getNbBindings();
    for (int i = 0; i < numBindings; ++i)
    {
        std::cout << "Binding " << i << ": ";
        std::cout << engine->getBindingName(i) << " (";
        std::cout << (engine->bindingIsInput(i) ? "Input" : "Output") << "), ";
        auto dims = engine->getBindingDimensions(i);
        std::cout << "Dimensions: ";
        for (int d = 0; d < dims.nbDims; d++) {
            std::cout << dims.d[d] << (d < dims.nbDims - 1 ? "x" : "");
        }
        std::cout << std::endl;
    }

    // Release CUDA stream
    cudaStreamDestroy(profileStream);

    std::cout << "\nEngine build and save complete." << std::endl;
    std::cout << "To use this engine, you need to provide two input tensors:" << std::endl;
    std::cout << "1. Current frame (current_frame)" << std::endl;
    std::cout << "2. Previous frame (previous_frame)" << std::endl;

    return 0;
} 