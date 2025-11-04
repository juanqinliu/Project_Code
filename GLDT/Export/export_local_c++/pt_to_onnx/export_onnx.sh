#!/bin/bash
#
# PyTorch to ONNX Export Script
# 
# Wrapper script for pt_to_onnx.py with automatic dependency management
# and sensible defaults for YOLO models.
#

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging helpers
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Default parameters
INPUT_PT="local.pt"
OUTPUT_ONNX=""
OUTPUT_SET=0
IMGSZ=640
OPSET=11
NO_SIMPLIFY=0

# Show usage
show_help() {
    cat << EOF
PyTorch to ONNX Exporter (Batch Optimized)

Usage: $0 [options]

Options:
  -i, --input <file>         Input PyTorch model (.pt)
  -o, --output <file>        Output ONNX model (.onnx, default: auto)
  --imgsz <size>             Image size (default: 640, must be multiple of 32)
  --opset <version>          ONNX opset version (default: 11)
  --no-simplify              Skip ONNX graph simplification
  -h, --help                 Show this help

Features:
  • Automatic intermediate output removal (improves TensorRT batch performance)
  • Dynamic batch dimension support
  • ONNX graph simplification
  
Examples:
  $0 -i model.pt
  $0 -i weights/yolo.pt -o weights/yolo.onnx
  $0 -i model.pt --imgsz 1024

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_ONNX="$2"
            OUTPUT_SET=1
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --opset)
            OPSET="$2"
            shift 2
            ;;
        --no-simplify)
            NO_SIMPLIFY=1
            shift
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

# Auto-generate output filename if not specified
if [ $OUTPUT_SET -eq 0 ]; then
    INPUT_DIR="$(dirname "$INPUT_PT")"
    INPUT_BASE="$(basename "$INPUT_PT" .pt)"
    OUTPUT_ONNX="${INPUT_DIR}/${INPUT_BASE}.onnx"
fi

# Display configuration
info "PyTorch → ONNX Export"
info "Input:  $INPUT_PT"
info "Output: $OUTPUT_ONNX"
info "Size:   ${IMGSZ}x${IMGSZ}"
info "Opset:  $OPSET"
info "Simplify: $([ $NO_SIMPLIFY -eq 0 ] && echo 'yes' || echo 'no')"
echo

# Validate input file
if [ ! -f "$INPUT_PT" ]; then
    error "Input file not found: $INPUT_PT"
    exit 1
fi

# Validate image size (must be multiple of 32)
if [ $IMGSZ -le 0 ] || [ $((IMGSZ % 32)) -ne 0 ]; then
    error "Image size must be positive and multiple of 32"
    exit 1
fi

# Check Python dependencies
info "Checking dependencies..."
if ! python3 -c "import torch, onnx, onnxsim" 2>/dev/null; then
    warn "Missing dependencies, installing..."
    pip install torch onnx onnx-simplifier --quiet
    if [ $? -eq 0 ]; then
        success "Dependencies installed"
    else
        error "Failed to install dependencies"
        exit 1
    fi
else
    success "Dependencies OK"
fi

# Locate Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pt_to_onnx.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    error "Conversion script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build Python arguments
ARGS=(
    -i "$INPUT_PT"
    -o "$OUTPUT_ONNX"
    --imgsz "$IMGSZ"
    --opset "$OPSET"
)

[ $NO_SIMPLIFY -eq 1 ] && ARGS+=(--no-simplify)

# Run conversion
info "Starting conversion..."
echo
python3 "$PYTHON_SCRIPT" "${ARGS[@]}"

# Check result
if [ $? -eq 0 ] && [ -f "$OUTPUT_ONNX" ]; then
    echo
    success "Conversion successful"
    info "Output: $OUTPUT_ONNX ($(du -h "$OUTPUT_ONNX" | cut -f1))"
else
    echo
    error "Conversion failed"
    exit 1
fi
