#!/bin/bash

# Set color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
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

# Error handling function
error_exit() {
    print_error "$1"
    exit 1
}

# Get the script directory and original directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_DIR="$(pwd)"

# Add path check in the script
if [ -f "$SCRIPT_DIR/build/onnx_to_engine" ]; then
    print_info "Found existing executable file: $SCRIPT_DIR/build/onnx_to_engine"
    print_info "File information:"
    ls -lh "$SCRIPT_DIR/build/onnx_to_engine" | awk '{print "  - " $9 ": " $5 " (Modification time: " $6 " " $7 " " $8 ")"}'
    print_info "Proceeding with recompilation..."
fi

# Check compile environment
print_info "Checking compile environment..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    print_warning "nvcc not found, please ensure CUDA is correctly installed"
    if [ -z "$CUDA_HOME" ]; then
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME="/usr/local/cuda"
            print_info "Automatically set CUDA_HOME to: $CUDA_HOME"
        else
            print_error "CUDA installation not found, please set CUDA_HOME environment variable"
            exit 1
        fi
    fi
else
    print_success "CUDA environment check passed"
fi

# Check TensorRT
if [ -z "$TENSORRT_ROOT" ]; then
    if [ -d "/usr/include/x86_64-linux-gnu" ] && [ -f "/usr/include/x86_64-linux-gnu/NvInfer.h" ]; then
        export TENSORRT_ROOT="/usr/lib/x86_64-linux-gnu"
        print_info "Automatically set TENSORRT_ROOT to: $TENSORRT_ROOT"
    else
        print_warning "TensorRT not found, please set TENSORRT_ROOT environment variable"
    fi
else
    print_success "TensorRT environment check passed"
fi

# Check OpenCV
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    print_warning "OpenCV not found, please ensure OpenCV is correctly installed"
else
    print_success "OpenCV environment check passed"
fi

# Create and enter build directory
print_info "Preparing build directory..."
mkdir -p "$SCRIPT_DIR/build" || error_exit "Unable to create build directory"
cd "$SCRIPT_DIR/build" || error_exit "Unable to enter build directory"

    # Clean old build files (optional)
if [ "$1" = "--clean" ]; then
    print_info "Cleaning old build files..."
    rm -rf *
fi

# Run CMake configuration
print_info "Running CMake configuration..."
cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release || error_exit "CMake configuration failed"

# Compile project
print_info "Compiling project..."
make -j$(nproc) || error_exit "Compilation failed"

# Check compilation results
if [ -f "onnx_to_engine" ]; then
    print_success "Build Success!"
    print_info "Generated executable files:"
    print_info "  - $(pwd)/onnx_to_engine (ONNX to TensorRT converter)"
    
    # Show file information
    echo ""
    print_info "File information:"
    ls -lh onnx_to_engine | awk '{print "  - " $9 ": " $5 " (Modification time: " $6 " " $7 " " $8 ")"}'
    
    # Set executable permissions
    chmod +x onnx_to_engine
    
else
    error_exit "Executable file not found"
fi


cd "$ORIGINAL_DIR"

print_success "Build completed!"
print_info "Usage:"
print_info "  1. Convert ONNX to TensorRT engine:"
print_info "     $SCRIPT_DIR/build/onnx_to_engine --help"
print_info "  2. Example command:"
print_info "     $SCRIPT_DIR/build/onnx_to_engine --input <onnx_file> --output <engine_file> --precision fp16"
print_info ""