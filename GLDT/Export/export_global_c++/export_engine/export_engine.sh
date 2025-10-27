#!/bin/bash


set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本目录和构建配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BIN_NAME="export_engine"

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Show help information
show_help() {
    echo "Unified TensorRT Engine Export Script"
    echo "====================================="
    echo ""
    echo "Usage:"
    echo "  $0 [options] [ONNX_FILE_PATH] [--precision <precision_type>]"
    echo ""
    echo "Precision Types:"
    echo "  fp32     Use FP32 precision (default)"
    echo "  fp16     Use FP16 precision"
    echo "  int8     Use INT8 precision (requires calibration data)"
    echo ""
    echo "Basic Options:"
    echo "  -h, --help             Show this help message"
    echo "  -i, --input PATH       Specify input ONNX file (equivalent to positional argument)"
    echo "  -o, --output PATH      Specify output engine file path"
    echo "  -p, --precision TYPE   Specify precision type (fp32/fp16/int8)"
    echo "  -d, --device DEVICE    Specify GPU device ID"
    echo "  -v, --verbose          Verbose output"
    echo ""
    echo "Project Management Options:"
    echo "  -b, --build            Build project"
    echo "  -c, --clean            Clean build directory"
    echo "  -r, --rebuild          Rebuild project"
    echo "  --workspace-size N     Workspace size in MB (default: 1024)"
    echo ""
    echo "Batch Processing Options:"
    echo "  --list-models          List available ONNX models"
    echo "  --auto-convert         Auto convert all ONNX models"
    echo ""
    echo "INT8 Calibration Options:"
    echo "  --calib-data PATH      Calibration data directory (default: ./calib_data)"
    echo "  --calib-list PATH      Calibration list file (default: ./calib_list.txt)"
    echo "  --batch-size N         Batch size (default: 1)"
    echo "  --calib-limit N        Calibration image limit (default: 0=unlimited)"
    echo "  --no-letterbox         Disable letterbox preprocessing"
    echo "  --no-bgr2rgb          Disable BGR to RGB conversion"
    echo ""
    echo "Examples:"
    echo "  $0 -b                                    # Build project"
    echo "  $0 ./weights/model.onnx                  # Use default FP32 precision"
    echo "  $0 ./weights/model.onnx -p fp16          # Use FP16 precision"
    echo "  $0 ./weights/model.onnx -p int8 --calib-limit 1000"
    echo "  $0 --list-models                         # List all models"
    echo "  $0 --auto-convert -p fp16                # Batch convert to FP16"
    echo ""
    echo "Environment Variables:"
    echo "  CUDA_VISIBLE_DEVICES   Specify GPU device"
    echo "  TENSORRT_HOME         TensorRT installation path"
    echo "  OPENCV_DIR            OpenCV installation path"
}

# Check environment
check_environment() {
    print_info "Checking environment..."
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        print_warning "nvcc not found, please ensure CUDA is properly installed"
    else
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_success "CUDA version: $cuda_version"
    fi
    
    # Check TensorRT
    if [ -z "$TENSORRT_HOME" ]; then
        print_warning "TENSORRT_HOME not set, will auto-search TensorRT"
    else
        print_success "TensorRT path: $TENSORRT_HOME"
    fi
    
    # Check OpenCV
    if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
        print_warning "OpenCV pkg-config not found, please ensure OpenCV is properly installed"
    else
        local opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)
        print_success "OpenCV version: $opencv_version"
    fi
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "cmake not found, please install CMake 3.10+"
        exit 1
    fi
    
    # Check make
    if ! command -v make &> /dev/null; then
        print_error "make not found, please ensure make is properly installed"
        exit 1
    fi
    
    # Check compiler
    if ! command -v g++ &> /dev/null; then
        print_error "g++ compiler not found, please install C++ compiler"
        exit 1
    fi
    
    print_success "Environment check completed"
}

# Build C++ program
build_cpp_program() {
    print_step "Building TensorRT engine export tool..."
    
    # Check build directory
    if [ ! -d "$BUILD_DIR" ]; then
        print_info "Creating build directory: $BUILD_DIR"
        mkdir -p "$BUILD_DIR"
    fi
    
    cd "$BUILD_DIR"
    
    # Configure CMake
    print_info "Configuring CMake..."
    if [ "$VERBOSE" = "true" ]; then
        cmake .. -DCMAKE_BUILD_TYPE=Release
    else
        cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    fi
    
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed"
        exit 1
    fi
    
    # Compile
    print_info "Compiling project..."
    local jobs=$(nproc)
    if [ "$VERBOSE" = "true" ]; then
        make -j$jobs
    else
        make -j$jobs > /dev/null 2>&1
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Compilation failed"
        exit 1
    fi
    
    # Check executable file
    if [ ! -f "$BUILD_DIR/$BIN_NAME" ]; then
        print_error "Executable file not generated: $BUILD_DIR/$BIN_NAME"
        exit 1
    fi
    
    print_success "Build completed! Executable: $BUILD_DIR/$BIN_NAME"
    cd "$SCRIPT_DIR"
}

# Check if C++ program exists
check_cpp_program() {
    if [ ! -f "$BUILD_DIR/$BIN_NAME" ]; then
        print_info "C++ program does not exist, starting build..."
        build_cpp_program
    elif [ "$FORCE_BUILD" = true ]; then
        print_info "Force rebuilding C++ program..."
        build_cpp_program
    fi
}

# Clean build files
clean_build() {
    print_step "Cleaning build directory..."
    
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    else
        print_info "Build directory does not exist, nothing to clean"
    fi
}

# Check ONNX file
check_onnx_file() {
    local onnx_file="$1"
    
    if [ ! -f "$onnx_file" ]; then
        print_error "ONNX file does not exist: $onnx_file"
        return 1
    fi
    
    if [[ ! "$onnx_file" =~ \.onnx$ ]]; then
        print_error "Unsupported file format, please use .onnx files"
        return 1
    fi
    
    # Check file size
    local file_size=$(stat -c%s "$onnx_file" 2>/dev/null || stat -f%z "$onnx_file" 2>/dev/null)
    if [ "$file_size" -lt 1024 ]; then
        print_warning "ONNX file is too small, may not be a valid model file"
    fi
    
    print_success "ONNX file check passed: $onnx_file"
    return 0
}

# List available ONNX models
list_models() {
    print_info "Searching for ONNX model files..."
    
    local found_models=()
    
    # Search current directory and subdirectories
    while IFS= read -r -d '' file; do
        found_models+=("$file")
    done < <(find . -name "*.onnx" -type f -print0 2>/dev/null)
    
    # Search parent directory
    if [ -d ".." ]; then
        while IFS= read -r -d '' file; do
            found_models+=("$file")
        done < <(find .. -name "*.onnx" -type f -print0 2>/dev/null)
    fi
    
    if [ ${#found_models[@]} -eq 0 ]; then
        print_warning "No ONNX model files found"
        return 1
    fi
    
    print_success "Found ${#found_models[@]} ONNX model files:"
    for model in "${found_models[@]}"; do
        local size=$(du -h "$model" | cut -f1)
        echo "  - $model ($size)"
    done
    
    return 0
}

# Convert ONNX to TensorRT engine
convert_model() {
    local onnx_file="$1"
    local precision="$2"
    local output_path="$3"
    
    print_step "Converting ONNX model to TensorRT engine..."
    
    # Convert to absolute path
    if [[ ! "$onnx_file" = /* ]]; then
        onnx_file="$(cd "$(dirname "$onnx_file")" && pwd)/$(basename "$onnx_file")"
    fi
    
    # Check executable file
    if [ ! -f "$BUILD_DIR/$BIN_NAME" ]; then
        print_error "Executable file does not exist, please build project first"
        print_info "Run: $0 -b"
        exit 1
    fi
    
    # Set environment variables
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        print_info "Using GPU device: $CUDA_VISIBLE_DEVICES"
    fi
    
    # Set TensorRT memory optimization environment variables
    export TENSORRT_MAX_WORKSPACE_SIZE=2147483648  # 2GB workspace
    export TENSORRT_MAX_GPU_MEMORY=4294967296      # 4GB GPU memory limit
    export TENSORRT_USE_CUDNN_ALGO_0=1             # Use cuDNN algorithm 0 (memory friendly)
    export TENSORRT_USE_CUBLAS_ALGO_0=1            # Use cuBLAS algorithm 0 (memory friendly)
    
    print_info "Setting TensorRT memory optimization:"
    print_info "  Workspace size: 2GB"
    print_info "  GPU memory limit: 4GB"
    print_info "  Using memory-friendly algorithms"
    
    # Build command
    local cmd="$BUILD_DIR/$BIN_NAME \"$onnx_file\""
    cmd="$cmd --precision $precision"
    
    # Set output path to be the same directory as the ONNX file
    local onnx_dir="$(dirname "$onnx_file")"
    local model_name="$(basename "$onnx_file" .onnx)"
    local engine_output="$onnx_dir/${model_name}_${precision}.engine"
    cmd="$cmd --output \"$engine_output\""
    
    # Add workspace size
    if [ -n "$WORKSPACE_SIZE" ]; then
        cmd="$cmd --workspace-size $WORKSPACE_SIZE"
    fi
    
    # Add INT8 specific parameters
    if [ "$precision" = "int8" ]; then
        if [ -n "$CALIB_DATA_DIR" ]; then
            cmd="$cmd --calib-data \"$CALIB_DATA_DIR\""
        fi
        
        if [ -n "$CALIB_LIST_FILE" ]; then
            cmd="$cmd --calib-list \"$CALIB_LIST_FILE\""
        fi
        
        if [ -n "$BATCH_SIZE" ]; then
            cmd="$cmd --batch-size $BATCH_SIZE"
        fi
        
        if [ -n "$CALIB_LIMIT" ] && [ "$CALIB_LIMIT" -gt 0 ]; then
            cmd="$cmd --calib-limit $CALIB_LIMIT"
        fi
        
        if [ "$USE_LETTERBOX" = false ]; then
            cmd="$cmd --no-letterbox"
        fi
        
        if [ "$USE_BGR2RGB" = false ]; then
            cmd="$cmd --no-bgr2rgb"
        fi
    fi
    
    print_info "Executing command: $cmd"
    
    # Execute conversion
    if eval $cmd; then
        print_success "TensorRT engine conversion completed!"
        
        # Check generated engine file
        if [ -f "$engine_output" ]; then
            local size=$(du -h "$engine_output" | cut -f1)
            print_info "Generated engine file: $engine_output ($size)"
        else
            print_warning "Engine file not found at expected location: $engine_output"
        fi
        
        return 0
    else
        print_error "TensorRT engine conversion failed!"
        return 1
    fi
}

# Auto convert all models
auto_convert() {
    print_step "Auto converting all ONNX models..."
    
    # List models
    local models=()
    while IFS= read -r -d '' file; do
        models+=("$file")
    done < <(find . -name "*.onnx" -type f -print0 2>/dev/null)
    
    if [ ${#models[@]} -eq 0 ]; then
        print_warning "No ONNX model files found in current directory"
        return 1
    fi
    
    print_info "Found ${#models[@]} model files, starting conversion..."
    
    local success_count=0
    local fail_count=0
    
    for model in "${models[@]}"; do
        print_info "Converting: $model"
        
        if convert_model "$model" "$PRECISION" ""; then
            ((success_count++))
        else
            ((fail_count++))
        fi
        
        echo ""
    done
    
    print_success "Auto conversion completed! Success: $success_count, Failed: $fail_count"
}

# Main function
main() {
    # Default parameters
    local build_flag="false"
    local clean_flag="false"
    local rebuild_flag="false"
    local list_flag="false"
    local auto_convert_flag="false"
    ONNX_FILE=""
    PRECISION="fp32"
    WORKSPACE_SIZE=""
    CALIB_DATA_DIR=""
    CALIB_LIST_FILE=""
    BATCH_SIZE=""
    CALIB_LIMIT=""
    USE_LETTERBOX=true
    USE_BGR2RGB=true
    FORCE_BUILD=false
    VERBOSE="false"
    local output_path=""
    local device=""
    
    # Parse command line parameters
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -b|--build)
                build_flag="true"
                shift
                ;;
            -c|--clean)
                clean_flag="true"
                shift
                ;;
            -r|--rebuild)
                rebuild_flag="true"
                shift
                ;;
            -i|--input)
                ONNX_FILE="$2"
                shift 2
                ;;
            -o|--output)
                output_path="$2"
                shift 2
                ;;
            -p|--precision)
                PRECISION="$2"
                shift 2
                ;;
            -d|--device)
                device="$2"
                export CUDA_VISIBLE_DEVICES="$device"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            --list-models)
                list_flag="true"
                shift
                ;;
            --auto-convert)
                auto_convert_flag="true"
                shift
                ;;
            --workspace-size)
                WORKSPACE_SIZE="$2"
                shift 2
                ;;
            --calib-data)
                CALIB_DATA_DIR="$2"
                shift 2
                ;;
            --calib-list)
                CALIB_LIST_FILE="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --calib-limit)
                CALIB_LIMIT="$2"
                shift 2
                ;;
            --no-letterbox)
                USE_LETTERBOX=false
                shift
                ;;
            --no-bgr2rgb)
                USE_BGR2RGB=false
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                if [ -z "$ONNX_FILE" ]; then
                    ONNX_FILE="$1"
                else
                    print_error "Command has too many parameters: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # If output path not provided, auto-generate based on input
    if [ -z "$output_path" ] && [ -n "$ONNX_FILE" ]; then
        local base_dir="$(dirname "$ONNX_FILE")"
        local base_name="$(basename "$ONNX_FILE" .onnx)"
        output_path="$base_dir/${base_name}.engine"
        if [ "$VERBOSE" = "true" ]; then
            print_info "Output path not specified, automatically set to: $output_path"
        fi
    fi
    
    # Check environment
    check_environment
    
    # Handle clean operation
    if [ "$clean_flag" = "true" ]; then
        clean_build
        exit 0
    fi
    
    # Handle rebuild operation
    if [ "$rebuild_flag" = "true" ]; then
        clean_build
        build_flag="true"
    fi
    
    # Handle build operation
    if [ "$build_flag" = "true" ]; then
        build_cpp_program
        if [ -z "$ONNX_FILE" ]; then
            exit 0
        fi
    fi
    
    # Handle list models operation
    if [ "$list_flag" = "true" ]; then
        list_models
        exit 0
    fi
    
    # Handle auto convert operation
    if [ "$auto_convert_flag" = "true" ]; then
        # Ensure project is built
        if [ ! -f "$BUILD_DIR/$BIN_NAME" ]; then
            print_info "Project not built, building..."
            build_cpp_program
        fi
        auto_convert
        exit 0
    fi
    
    # Handle single file conversion
    if [ -n "$ONNX_FILE" ]; then
        # Check ONNX file
        if ! check_onnx_file "$ONNX_FILE"; then
            exit 1
        fi
        
        # Validate precision type
        if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp16" && "$PRECISION" != "int8" ]]; then
            print_error "Unsupported precision type: $PRECISION"
            print_error "Supported precision types: fp32, fp16, int8"
            exit 1
        fi
        
        # Ensure project is built
        if [ ! -f "$BUILD_DIR/$BIN_NAME" ]; then
            print_info "Project not built, building..."
            build_cpp_program
        fi
        
        # Convert model
        convert_model "$ONNX_FILE" "$PRECISION" "$output_path"
        exit $?
    fi
    
    # If no operation specified, show help
    print_warning "No operation specified"
    show_help
    exit 1
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
