#!/bin/bash
#
# TensorRT Engine Builder for FP16 Dynamic Batch Inference
#
# Optimized for variable ROI counts (1-N)
# Target use case: Local detection with 1-4 ROIs per frame
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Default parameters optimized for dynamic ROI inference
INPUT_ONNX=""
OUTPUT_ENGINE=""
MIN_BATCH=1          
OPT_BATCH=4          
MAX_BATCH=8         
IMGSZ=640
WORKSPACE=4096       
PRECISION="fp16"

show_help() {
    cat << EOF
TensorRT Engine Builder (FP16 Dynamic Batch Optimized)

Usage: $0 [options]

Options:
  -i, --input <file>         Input ONNX model (.onnx) [REQUIRED]
  -o, --output <file>        Output engine file (.engine, default: auto)
  --batch-min <size>         Minimum batch size (default: 1)
  --batch-opt <size>         Optimal batch size (default: 3)
  --batch-max <size>         Maximum batch size (default: 16)
  --imgsz <size>             Input image size (default: 640)
  --workspace <mb>           Workspace size in MB (default: 4096)
  -h, --help                 Show this help

Batch Configuration:
  The engine will be optimized for batch=${OPT_BATCH} while supporting ${MIN_BATCH}-${MAX_BATCH}
  For best performance, set --batch-opt to your typical ROI count

Features:
  • FP16 precision (2x faster than FP32, better accuracy than INT8)
  • Dynamic batch support (handles variable ROI counts)
  • Optimized kernel selection for typical batch size
  • Large workspace for batch kernel fusion

Examples:
  $0 -i model.onnx
  $0 -i model.onnx -o model_fp16.engine
  $0 -i model.onnx --batch-opt 4 --batch-max 8

EOF
}

# Parse arguments
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
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_ONNX" ]; then
    error "Input ONNX file is required"
    show_help
    exit 1
fi

if [ ! -f "$INPUT_ONNX" ]; then
    error "Input file not found: $INPUT_ONNX"
    exit 1
fi

# Auto-generate output filename
if [ -z "$OUTPUT_ENGINE" ]; then
    INPUT_DIR="$(dirname "$INPUT_ONNX")"
    INPUT_BASE="$(basename "$INPUT_ONNX" .onnx)"
    OUTPUT_ENGINE="${INPUT_DIR}/${INPUT_BASE}_fp16_b${MIN_BATCH}-${MAX_BATCH}.engine"
fi

# Validate batch configuration
if [ $MIN_BATCH -lt 1 ] || [ $OPT_BATCH -lt $MIN_BATCH ] || [ $MAX_BATCH -lt $OPT_BATCH ]; then
    error "Invalid batch configuration: min=$MIN_BATCH, opt=$OPT_BATCH, max=$MAX_BATCH"
    error "Required: 1 <= min <= opt <= max"
    exit 1
fi

# Display configuration
echo
info "TensorRT Engine Builder (FP16 Dynamic Batch)"
info "Input:     $INPUT_ONNX"
info "Output:    $OUTPUT_ENGINE"
info "Precision: FP16"
info "Batch:     min=$MIN_BATCH, opt=$OPT_BATCH, max=$MAX_BATCH"
info "Size:      ${IMGSZ}x${IMGSZ}"
info "Workspace: ${WORKSPACE}MB"
echo

# Locate script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="$SCRIPT_DIR/build/onnx_to_engine"

# Check if binary exists, build if needed
if [ ! -f "$BUILD_BIN" ]; then
    warn "Binary not found, building..."
    if [ -f "$SCRIPT_DIR/build.sh" ]; then
        cd "$SCRIPT_DIR" && ./build.sh
        if [ $? -ne 0 ]; then
            error "Build failed"
            exit 1
        fi
    else
        error "build.sh not found"
        exit 1
    fi
fi

if [ ! -f "$BUILD_BIN" ]; then
    error "Binary still not found after build: $BUILD_BIN"
    exit 1
fi

# Build engine
info "Building TensorRT engine (this may take a few minutes)..."
echo

"$BUILD_BIN" \
    --input "$INPUT_ONNX" \
    --output "$OUTPUT_ENGINE" \
    --batch-min "$MIN_BATCH" \
    --batch-opt "$OPT_BATCH" \
    --batch-max "$MAX_BATCH" \
    --imgsz "$IMGSZ" \
    --workspace "$WORKSPACE" \
    --precision "$PRECISION"

# Check result
if [ $? -eq 0 ] && [ -f "$OUTPUT_ENGINE" ]; then
    echo
    success "Engine build successful!"
    info "Output: $OUTPUT_ENGINE ($(du -h "$OUTPUT_ENGINE" | cut -f1))"
    echo
    
    # Verify dynamic batch support
    info "Verifying dynamic batch support..."
    if command -v trtexec &> /dev/null; then
        echo
        # Test with different batch sizes
        for batch in $MIN_BATCH $OPT_BATCH $MAX_BATCH; do
            info "Testing batch=$batch..."
            trtexec --loadEngine="$OUTPUT_ENGINE" \
                    --shapes=images:${batch}x3x${IMGSZ}x${IMGSZ} \
                    --noDataTransfers \
                    --verbose 2>&1 | grep -i "batch\|throughput" | head -5
        done
        echo
        success "Dynamic batch verification complete"
    else
        warn "trtexec not found, skipping verification"
        info "Install with: apt-get install tensorrt"
    fi
    
    echo
    info "Next steps:"
    info "  1. Copy engine to inference workspace"
    info "  2. Update model path in configuration"
    info "  3. Test with varying ROI counts (1-${MAX_BATCH})"
    info ""
    info "Expected performance:"
    info "  • batch=1: ~25ms"
    info "  • batch=3: ~30-35ms (not ~60ms!)"
    info "  • batch=$MAX_BATCH: ~80-100ms"
    echo
else
    echo
    error "Engine build failed"
    exit 1
fi

