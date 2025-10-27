#!/bin/bash

# Color output
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

# Default parameters
INPUT_ONNX="local.onnx"
OUTPUT_ENGINE="local.engine"
MIN_BATCH=1
OPT_BATCH=4
MAX_BATCH=32
WORKSPACE=2048
IMGSZ=640
CALIB_DIR="calib_data"
CALIB_LIST="calib_list.txt"
PRECISION="int8"  # Default to INT8 precision

# Help information
show_help() {
    echo "ONNX to TensorRT Engine Export Tool"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input <file>        Input ONNX file (default: local.onnx)"
    echo "  -o, --output <file>       Output Engine file (default: local.engine)"
    echo "  --batch-min <size>        Minimum batch size (default: 1)"
    echo "  --batch-opt <size>        Optimal batch size (default: 4)"
    echo "  --batch-max <size>        Maximum batch size (default: 32)"
    echo "  --workspace <mb>          Workspace size in MB (default: 2048)"
    echo "  --imgsz <size>           Input image size (default: 640)"
    echo "  --calib-dir <dir>         Calibration data directory (default: calib_data)"
    echo "  --calib-list <file>       Calibration list file (default: calib_list.txt)"
    echo "  --precision <mode>        Precision mode: int8, fp16 or fp32 (default: int8)"
    echo "  --auto-calib-list         Auto-generate calibration list from calib_dir"
    echo "  --calib-num <n>           Maximum number of samples for auto-generation (default: 1000)"
    echo "  --skip-build              Skip C++ program compilation"
    echo "  -h, --help                Show help information"
}

# Parse command line arguments
SKIP_BUILD=false
AUTO_CALIB=false
CALIB_NUM=1000

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_ONNX="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_ENGINE="$2"
            shift 2
            ;;
        --batch-min)
            MIN_BATCH="$2"
            shift 2
            ;;
        --batch-opt)
            OPT_BATCH="$2"
            shift 2
            ;;
        --batch-max)
            MAX_BATCH="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --calib-dir)
            CALIB_DIR="$2"
            shift 2
            ;;
        --calib-list)
            CALIB_LIST="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            if [[ "$PRECISION" != "int8" && "$PRECISION" != "fp16" && "$PRECISION" != "fp32" ]]; then
                print_error "Precision mode must be int8, fp16 or fp32"
                exit 1
            fi
            shift 2
            ;;
        --auto-calib-list)
            AUTO_CALIB=true
            shift
            ;;
        --calib-num)
            CALIB_NUM="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
done

print_info "=== ONNX to TensorRT Engine Export Tool ==="
print_info "Input ONNX file: $INPUT_ONNX"
print_info "Output Engine file: $OUTPUT_ENGINE"
print_info "Batch size range: $MIN_BATCH - $MAX_BATCH (optimal: $OPT_BATCH)"
print_info "Image size: ${IMGSZ}x${IMGSZ}"
print_info "Workspace: ${WORKSPACE}MB"
print_info "Precision mode: $PRECISION"
if [[ "$PRECISION" == "int8" ]]; then
    print_info "Calibration data directory: $CALIB_DIR"
    print_info "Calibration list file: $CALIB_LIST"
fi
echo ""

# Parameter validation
# Convert relative paths to absolute paths
if [[ "$INPUT_ONNX" != /* ]]; then
    INPUT_ONNX="$(pwd)/$INPUT_ONNX"
fi

# Auto-generate output filename based on input ONNX file if not specified
if [[ "$OUTPUT_ENGINE" == "local.engine" ]]; then
    # Extract directory and filename without extension from input ONNX file
    ONNX_DIR=$(dirname "$INPUT_ONNX")
    ONNX_BASENAME=$(basename "$INPUT_ONNX" .onnx)
    
    # Generate output filename with precision and batch info
    OUTPUT_ENGINE="${ONNX_DIR}/${ONNX_BASENAME}_${PRECISION}_b${MAX_BATCH}_${IMGSZ}.engine"
    print_info "Auto-generated output filename: $OUTPUT_ENGINE"
fi

if [[ "$OUTPUT_ENGINE" != /* ]]; then
    OUTPUT_ENGINE="$(pwd)/$OUTPUT_ENGINE"
fi

if [ ! -f "$INPUT_ONNX" ]; then
    print_error "Input file $INPUT_ONNX does not exist"
    exit 1
fi

# Only INT8 mode requires calibration data
if [[ "$PRECISION" == "int8" ]]; then
    if [ ! -d "$CALIB_DIR" ]; then
        print_error "Calibration data directory $CALIB_DIR does not exist"
        exit 1
    fi

    # Auto-generate if needed or if list doesn't exist
    if [[ "$AUTO_CALIB" == true || ! -f "$CALIB_LIST" ]]; then
        print_info "Generating calibration list: $CALIB_LIST (max $CALIB_NUM entries)"
        # Generate list with limited count (support common image formats)
        find "$CALIB_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) \
            | head -n "$CALIB_NUM" \
            | sed "s#^$CALIB_DIR/##" > "$CALIB_LIST"
        if [ ! -s "$CALIB_LIST" ]; then
            print_error "No usable images found in $CALIB_DIR to generate calibration list"
            exit 1
        fi
        print_success "Calibration list generated: $(wc -l < "$CALIB_LIST") entries"
    fi

    if [ ! -f "$CALIB_LIST" ]; then
        print_error "Calibration list file $CALIB_LIST does not exist"
        exit 1
    fi
fi

if [ $MIN_BATCH -lt 1 ] || [ $OPT_BATCH -lt $MIN_BATCH ] || [ $MAX_BATCH -lt $OPT_BATCH ]; then
    print_error "Invalid batch size range"
    exit 1
fi

if [ $WORKSPACE -lt 1024 ]; then
    print_warning "Workspace less than 1024MB may affect performance"
fi

# Compile C++ program
if [ "$SKIP_BUILD" = false ]; then
    print_info "Compiling C++ conversion program..."
    
    # Get script directory and original working directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ORIGINAL_DIR="$(pwd)"
    
    # Create build directory
    mkdir -p "$SCRIPT_DIR/build"
    cd "$SCRIPT_DIR/build"
    
    # Check CUDA environment
    if [ -z "$CUDA_HOME" ]; then
        if [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME="/usr/local/cuda"
            print_success "Automatically set CUDA_HOME to: $CUDA_HOME"
        else
            print_error "CUDA installation not found, please set CUDA_HOME environment variable"
            exit 1
        fi
    fi
    
    # Run cmake - ensure pointing to correct CMakeLists.txt directory
    cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed, please check TensorRT and CUDA installation"
        cd "$ORIGINAL_DIR"
        exit 1
    fi
    
    # Compile
    make -j$(nproc)
    if [ $? -eq 0 ]; then
        print_success "C++ program compilation completed"
        cd "$ORIGINAL_DIR"
    else
        print_error "C++ program compilation failed"
        cd "$ORIGINAL_DIR"
        exit 1
    fi
else
    print_warning "Skipping C++ program compilation"
    if [ ! -f "$SCRIPT_DIR/build/onnx_to_engine" ]; then
        print_error "Executable file $SCRIPT_DIR/build/onnx_to_engine does not exist"
        exit 1
    fi
fi

# ONNX -> TensorRT Engine conversion
print_info "Starting ONNX -> TensorRT Engine conversion..."

# Prepare command arguments
CMD_ARGS=(
    "--input" "$INPUT_ONNX"
    "--output" "$OUTPUT_ENGINE"
    "--batch-min" "$MIN_BATCH"
    "--batch-opt" "$OPT_BATCH"
    "--batch-max" "$MAX_BATCH"
    "--imgsz" "$IMGSZ"
    "--workspace" "$WORKSPACE"
    "--precision" "$PRECISION"
)

# Only INT8 mode requires calibration data parameters
if [[ "$PRECISION" == "int8" ]]; then
    CMD_ARGS+=(
        "--calib-dir" "$CALIB_DIR"
        "--calib-list" "$CALIB_LIST"
    )
fi

# Debug information: show actual parameters being passed
print_info "Executing command: $SCRIPT_DIR/build/onnx_to_engine ${CMD_ARGS[*]}"

# Execute conversion command
"$SCRIPT_DIR/build/onnx_to_engine" "${CMD_ARGS[@]}"

if [ $? -eq 0 ] && [ -f "$OUTPUT_ENGINE" ]; then
    print_success "ONNX -> TensorRT Engine conversion completed"
    print_info "Engine file size: $(du -h "$OUTPUT_ENGINE" | cut -f1)"
else
    print_error "ONNX -> TensorRT Engine conversion failed"
    exit 1
fi

# Completion
echo ""
print_success "============== Conversion Complete =============="
print_info "Input file: $INPUT_ONNX ($(du -h "$INPUT_ONNX" | cut -f1))"
print_info "Engine file: $OUTPUT_ENGINE ($(du -h "$OUTPUT_ENGINE" | cut -f1))"
print_info "Features: ${PRECISION^^} precision + dynamic batching ($MIN_BATCH-$MAX_BATCH)"
print_info "Recommended batch size $OPT_BATCH for optimal performance" 