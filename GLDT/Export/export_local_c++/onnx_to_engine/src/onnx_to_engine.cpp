#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <sstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"
#include "int8_calibrator.h"

using namespace nvinfer1;

// Helper function: print dimension information
std::string printDims(const nvinfer1::Dims& dims) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < dims.nbDims; i++) {
        if (i > 0) ss << ", ";
        ss << dims.d[i];
    }
    ss << "]";
    return ss.str();
}

// Precision Mode Enum
enum class PrecisionMode {
    FP32,
    FP16,
    INT8
};

// Error Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Add more log levels
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[F] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[E] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "[W] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[I] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[V] " << msg << std::endl;
                break;
            default:
                std::cout << msg << std::endl;
                break;
        }
    }
} gLogger;

// Helper function for resource release
struct TRTDestroy {
    template <class T>
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

bool build_engine(const std::string& onnx_file, const std::string& engine_file,
                 const std::string& calib_data_dir, const std::string& calib_list,
                 int min_batch = 1, int opt_batch = 16, int max_batch = 32,
                 int imgsz = 640, size_t workspace_mb = 2048,
                 PrecisionMode precision = PrecisionMode::INT8) {
    
    std::cout << "\n=== Building TensorRT Engine ===" << std::endl;
    std::cout << "ONNX file: " << onnx_file << std::endl;
    std::cout << "Engine file: " << engine_file << std::endl;
    std::cout << "Batch size range: " << min_batch << " - " << max_batch 
              << " (optimal: " << opt_batch << ")" << std::endl;
    std::cout << "Input size: " << imgsz << "x" << imgsz << std::endl;
    std::cout << "Workspace size: " << workspace_mb << " MB" << std::endl;
    std::cout << "Precision mode: ";
    if (precision == PrecisionMode::INT8) std::cout << "INT8";
    else if (precision == PrecisionMode::FP16) std::cout << "FP16";
    else std::cout << "FP32";
    std::cout << std::endl;
    
    // Create builder
    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        std::cerr << "Create builder failed" << std::endl;
        return false;
    }
    
    // Create network definition - ensure explicit batch is used
    const auto explicit_batch = 
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cerr << "Create network definition failed" << std::endl;
        return false;
    }
    
    // Create ONNX parser
    TRTUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        std::cerr << "Create ONNX parser failed" << std::endl;
        return false;
    }
    
    // Parse ONNX file
    std::cout << "\n=== Parsing ONNX Model ===" << std::endl;
    bool parsed = parser->parseFromFile(onnx_file.c_str(),
                                      static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) {
        std::cerr << "ONNX parsing failed" << std::endl;
        return false;
    }
    
    // Print network information
    std::cout << "\n=== Network Information ===" << std::endl;
    std::cout << "Number of inputs: " << network->getNbInputs() << std::endl;
    std::cout << "Number of outputs: " << network->getNbOutputs() << std::endl;
    
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "\nInput " << i << ": " << input->getName() << std::endl;
        std::cout << "Original dimensions: " << printDims(dims) << std::endl;
        std::cout << "Data type: " << static_cast<int>(input->getType()) << std::endl;
    }
    
    // Create build configuration
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Create build configuration failed" << std::endl;
        return false;
    }
    
    // Set maximum workspace size
    size_t workspace_bytes = static_cast<size_t>(workspace_mb) * 1024 * 1024;
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
    
    // Create optimization profile
    std::cout << "\n=== Create optimization profile ===" << std::endl;
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Create optimization profile failed" << std::endl;
        return false;
    }
    
    // Set dynamic range of input tensor
    bool all_dynamic_set = true;
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        
        std::cout << "\nSet dynamic range of input " << input->getName() << ":" << std::endl;
        std::cout << "Original dimension: " << printDims(dims) << std::endl;
        
        // Create new dimension object
        nvinfer1::Dims minDims = dims;
        nvinfer1::Dims optDims = dims;
        nvinfer1::Dims maxDims = dims;
        
        // Ensure dimension number is correct
        if (dims.nbDims < 1) {
            std::cerr << "Error: Input dimension number must be greater than 0" << std::endl;
            return false;
        }
        
        // Set batch dimension
        minDims.d[0] = std::max(1, min_batch);  // Ensure minimum value is at least 1
        optDims.d[0] = std::max(min_batch, std::min(opt_batch, max_batch));  // Ensure opt is between min and max
        maxDims.d[0] = std::max(opt_batch, max_batch);  // Ensure max is not less than opt
        
        // Set other dimensions
        for (int d = 1; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) {
                // For dynamic dimension, set reasonable range
                minDims.d[d] = 1;  // Set minimum value to 1
                optDims.d[d] = imgsz;  // Set optimal value to target size
                maxDims.d[d] = std::max(imgsz, 1024);  // Set maximum value
            } else {
                // For static dimension, keep unchanged
                minDims.d[d] = dims.d[d];
                optDims.d[d] = dims.d[d];
                maxDims.d[d] = dims.d[d];
            }
        }
        
        std::cout << "Minimum dimension: " << printDims(minDims) << std::endl;
        std::cout << "Optimal dimension: " << printDims(optDims) << std::endl;
        std::cout << "Maximum dimension: " << printDims(maxDims) << std::endl;
        
        // Validate dimension values
        auto validate_dims = [](const nvinfer1::Dims& d, const char* name) {
            for (int i = 0; i < d.nbDims; i++) {
                if (d.d[i] <= 0) {
                    std::cerr << "Error: " << name << " dimension must be greater than 0, current " 
                             << i << " dimension is " << d.d[i] << std::endl;
                    return false;
                }
            }
            return true;
        };
        
        if (!validate_dims(minDims, "Minimum") || 
            !validate_dims(optDims, "Optimal") || 
            !validate_dims(maxDims, "Maximum")) {
            all_dynamic_set = false;
            continue;
        }
        
        // Set dimensions for the optimization profile
        bool success = profile->setDimensions(input->getName(), 
                                            nvinfer1::OptProfileSelector::kMIN, minDims) &&
                      profile->setDimensions(input->getName(), 
                                            nvinfer1::OptProfileSelector::kOPT, optDims) &&
                      profile->setDimensions(input->getName(), 
                                            nvinfer1::OptProfileSelector::kMAX, maxDims);
                                            
        if (!success) {
            std::cerr << "Failed to set dynamic range for input " << input->getName() << std::endl;
            all_dynamic_set = false;
            continue;
        }
        
        std::cout << "✅ Dynamic range set successfully" << std::endl;
    }
    
    if (!all_dynamic_set) {
        std::cerr << "Failed to set dynamic range" << std::endl;
        return false;
    }
    
    // Validate the optimization profile
    if (!profile->isValid()) {
        std::cerr << "Optimization profile is invalid" << std::endl;
        return false;
    }
    
    // Add the optimization profile to the configuration
    config->addOptimizationProfile(profile);
    
    // Set precision flags
    Int8EntropyCalibrator* calibrator = nullptr;
    
    if (precision == PrecisionMode::FP16) {
        std::cout << "\n=== Configuring FP16 Precision ===" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        
        // Check FP16 support
        if (!builder->platformHasFastFp16()) {
            std::cout << "Warning: Platform does not support fast FP16, performance may be limited" << std::endl;
        } else {
            std::cout << "✅ Platform supports fast FP16" << std::endl;
        }
        
        // Prefer FP16 over FP32 when possible
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        std::cout << "✅ FP16 precision constraints enabled" << std::endl;
    } else if (precision == PrecisionMode::INT8) {
        std::cout << "\n=== Configuring INT8 Quantization ===" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kINT8);  // Enable INT8
        
        if (!builder->platformHasFastInt8()) {
            std::cerr << "Warning: Platform does not support fast INT8" << std::endl;
        }
        
        // Create INT8 calibrator
        std::cout << "Creating INT8 calibrator..." << std::endl;
        try {
            calibrator = new Int8EntropyCalibrator(
                calib_list,           // Calibration list file
                calib_data_dir,       // Calibration data directory
                min_batch,            // Batch size
                imgsz,                // Input height
                imgsz,                // Input width
                3,                    // Input channels
                "calibration.table"   // Calibration cache file name
            );
            config->setInt8Calibrator(calibrator);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create calibrator: " << e.what() << std::endl;
            return false;
        }
    } else if (precision == PrecisionMode::FP32) {
        std::cout << "\n=== Using FP32 Precision ===" << std::endl;
    }
    
    // Build engine
    std::cout << "\n=== Building Engine ===" << std::endl;
    std::cout << "This may take several minutes..." << std::endl;
    
    TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(
        builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        std::cerr << "Engine build failed" << std::endl;
        delete calibrator;  // Safe deletion, even if nullptr
        return false;
    }
    
    // Save engine file
    std::cout << "\n=== Saving Engine ===" << std::endl;
    std::ofstream engineFile(engine_file, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Unable to create engine file" << std::endl;
        delete calibrator;  // Safe deletion
        return false;
    }
    engineFile.write(static_cast<const char*>(serializedEngine->data()),
                    serializedEngine->size());
    
    // Validate engine
    std::cout << "\n=== Validating Engine ===" << std::endl;
    
    // Create runtime engine for validation
    TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        delete calibrator;  // Safe deletion
        return false;
    }
    
    TRTUniquePtr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serializedEngine->data(), 
                                     serializedEngine->size()));
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        delete calibrator;  // Safe deletion
        return false;
    }
    
    // Validate bindings and dynamic dimensions
    std::cout << "Number of bindings: " << engine->getNbBindings() << std::endl;
    bool has_dynamic_shapes = false;
    
    for (int i = 0; i < engine->getNbBindings(); i++) {
        std::cout << "\nBinding " << i << ": " << engine->getBindingName(i)
                  << (engine->bindingIsInput(i) ? " (input)" : " (output)")
                  << std::endl;
        
        // Get binding dimensions
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        std::cout << "Dimensions: " << printDims(dims) << std::endl;
        
        // Check if dynamic shapes are supported
        bool binding_is_dynamic = false;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) {
                binding_is_dynamic = true;
                has_dynamic_shapes = true;
                break;
            }
        }
        std::cout << "Is dynamic: " << (binding_is_dynamic ? "Yes" : "No") << std::endl;
        
        if (engine->bindingIsInput(i) && !binding_is_dynamic) {
            std::cout << "Warning: Input " << engine->getBindingName(i) 
                      << " does not support dynamic batching" << std::endl;
        }
    }
    
    // Clean up resources
    delete calibrator;  // Safe deletion, even if nullptr
    
    // Print final status
    std::cout << "\n=== Build Complete ===" << std::endl;
    std::cout << "Engine size: " << serializedEngine->size() / (1024.0 * 1024.0) 
              << " MB" << std::endl;
    std::cout << "Precision mode: ";
    if (precision == PrecisionMode::INT8) std::cout << "INT8";
    else if (precision == PrecisionMode::FP16) std::cout << "FP16";
    else std::cout << "FP32";
    std::cout << std::endl;
    std::cout << "Dynamic shapes support: " << (has_dynamic_shapes ? "Yes ✓" : "No ✗") << std::endl;
    
    if (!has_dynamic_shapes) {
        std::cout << "\n⚠️  WARNING: Engine does not support dynamic batching!" << std::endl;
        std::cout << "This will prevent batch inference optimization." << std::endl;
        std::cout << "\nTroubleshooting:" << std::endl;
        std::cout << "1. Verify ONNX has dynamic batch: first dim should be 'batch_size'" << std::endl;
        std::cout << "2. Check optimization profile configuration" << std::endl;
        std::cout << "3. Ensure min/opt/max batch values are valid" << std::endl;
    } else {
        std::cout << "\n✅ Dynamic batch support verified!" << std::endl;
        std::cout << "Engine supports batch sizes from " << min_batch 
                  << " to " << max_batch << std::endl;
        std::cout << "Optimized for batch size: " << opt_batch << std::endl;
    }
    
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " --input <onnx_file> --output <engine_file> "
                  << "--calib-dir <calib_data_dir> --calib-list <calib_list> "
                  << "[--batch-min <min>] [--batch-opt <opt>] [--batch-max <max>] "
                  << "[--imgsz <size>] [--workspace <mb>] [--precision <fp32|fp16|int8>]" << std::endl;
        return 1;
    }

    // Default parameter values
    std::string onnx_file;
    std::string engine_file;
    std::string calib_data_dir;
    std::string calib_list;
    int min_batch = 1;
    int opt_batch = 16;
    int max_batch = 32;
    int imgsz = 640;
    size_t workspace_mb = 2048;
    PrecisionMode precision = PrecisionMode::INT8;  // Default to INT8

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            onnx_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            engine_file = argv[++i];
        } else if (arg == "--calib-dir" && i + 1 < argc) {
            calib_data_dir = argv[++i];
        } else if (arg == "--calib-list" && i + 1 < argc) {
            calib_list = argv[++i];
        } else if (arg == "--batch-min" && i + 1 < argc) {
            min_batch = std::stoi(argv[++i]);
        } else if (arg == "--batch-opt" && i + 1 < argc) {
            opt_batch = std::stoi(argv[++i]);
        } else if (arg == "--batch-max" && i + 1 < argc) {
            max_batch = std::stoi(argv[++i]);
        } else if (arg == "--imgsz" && i + 1 < argc) {
            imgsz = std::stoi(argv[++i]);
        } else if (arg == "--workspace" && i + 1 < argc) {
            workspace_mb = std::stoul(argv[++i]);
        } else if (arg == "--precision" && i + 1 < argc) {
            std::string prec = argv[++i];
            if (prec == "fp16") {
                precision = PrecisionMode::FP16;
            } else if (prec == "int8") {
                precision = PrecisionMode::INT8;
            } else if (prec == "fp32") {
                precision = PrecisionMode::FP32;
            } else {
                std::cerr << "Error: Invalid precision mode, must be fp32, fp16, or int8" << std::endl;
                return 1;
            }
        }
    }

    // Validate required parameters
    if (onnx_file.empty() || engine_file.empty()) {
        std::cerr << "Error: Missing required parameters" << std::endl;
        return 1;
    }

    // Calibration data is required for INT8 mode
    if (precision == PrecisionMode::INT8 && (calib_data_dir.empty() || calib_list.empty())) {
        std::cerr << "Error: INT8 mode requires calibration data and list" << std::endl;
        return 1;
    }

    // Validate parameter values
    if (min_batch < 1 || opt_batch < min_batch || max_batch < opt_batch) {
        std::cerr << "Error: Invalid batch size range" << std::endl;
        return 1;
    }

    if (imgsz <= 0 || (imgsz % 32) != 0) {
        std::cerr << "Error: Image size must be a multiple of 32" << std::endl;
        return 1;
    }

    if (workspace_mb < 1024) {
        std::cerr << "Warning: Workspace size less than 1024MB may affect performance" << std::endl;
    }

    // Check if files exist
    std::ifstream onnx_check(onnx_file);
    if (!onnx_check.good()) {
        std::cerr << "Error: ONNX file not found: " << onnx_file << std::endl;
        return 1;
    }

    // Check calibration files only in INT8 mode
    if (precision == PrecisionMode::INT8) {
        std::ifstream calib_dir_check(calib_data_dir);
        if (!calib_dir_check.good()) {
            std::cerr << "Error: Calibration data directory not found: " << calib_data_dir << std::endl;
            return 1;
        }

        std::ifstream calib_list_check(calib_list);
        if (!calib_list_check.good()) {
            std::cerr << "Error: Calibration list file not found: " << calib_list << std::endl;
            return 1;
        }
    }

    if (!build_engine(onnx_file, engine_file, calib_data_dir, calib_list,
                     min_batch, opt_batch, max_batch, imgsz, workspace_mb, precision)) {
        std::cerr << "Engine build failed" << std::endl;
        return 1;
    }

    return 0;
} 