#!/bin/bash

# Colored output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print helpers
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

# Default args
INPUT_PT="local_p2.pt"
OUTPUT_ONNX="local_p2.onnx"
OUTPUT_SET=0
IMGSZ=640
OPSET=11
NO_SIMPLIFY=0
KEEP_INTERMEDIATE=0
VERBOSE=0

# Help
show_help() {
    echo "PT to ONNX exporter"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input <file>         Input PT file (default: ${INPUT_PT})"
    echo "  -o, --output <file>        Output ONNX file (default: ${OUTPUT_ONNX})"
    echo "  --imgsz <size>             Input image size (default: ${IMGSZ})"
    echo "  --opset <int>              ONNX opset version (default: ${OPSET})"
    echo "  --no-simplify              Disable ONNX simplification"
    echo "  --keep-intermediate        Keep intermediate outputs"
    echo "  --verbose                  Verbose export logs"
    echo "  -h, --help                 Show this help message"
}

# Parse CLI options
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
            shift 1
            ;;
        --keep-intermediate)
            KEEP_INTERMEDIATE=1
            shift 1
            ;;
        --verbose)
            VERBOSE=1
            shift 1
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# If output not explicitly set, derive it from input: same dir/name with .onnx suffix
if [ $OUTPUT_SET -eq 0 ]; then
    INPUT_DIR="$(dirname -- "$INPUT_PT")"
    INPUT_BASE="$(basename -- "$INPUT_PT")"
    # remove only the last .pt suffix if present
    if [[ "$INPUT_BASE" == *.pt ]]; then
        BASE_NO_EXT="${INPUT_BASE%.pt}"
    else
        BASE_NO_EXT="$INPUT_BASE"
    fi
    # preserve relative path semantics; if INPUT_DIR is '.', keep './' for consistency with examples
    if [ "$INPUT_DIR" = "." ]; then
        OUTPUT_ONNX="./${BASE_NO_EXT}.onnx"
    else
        OUTPUT_ONNX="${INPUT_DIR}/${BASE_NO_EXT}.onnx"
        # If input path is relative and doesn't start with './', add './' for consistent display
        if [[ "$INPUT_PT" != /* && "$INPUT_PT" != ./* ]]; then
            OUTPUT_ONNX="./${OUTPUT_ONNX}"
        fi
    fi
fi

print_info "=== PT to ONNX Exporter ==="
print_info "Input PT: $INPUT_PT"
print_info "Output ONNX: $OUTPUT_ONNX"
print_info "Image size: ${IMGSZ}x${IMGSZ}"
print_info "Opset: ${OPSET} | Simplify: $([ $NO_SIMPLIFY -eq 0 ] && echo on || echo off) | Keep intermediate: $([ $KEEP_INTERMEDIATE -eq 1 ] && echo yes || echo no) | Verbose: $([ $VERBOSE -eq 1 ] && echo yes || echo no)"
echo ""

# Validate params
if [ ! -f "$INPUT_PT" ]; then
    print_error "Input file not found: $INPUT_PT"
    exit 1
fi

if [ $IMGSZ -le 0 ] || [ $(($IMGSZ % 32)) -ne 0 ]; then
    print_error "Image size must be a positive multiple of 32"
    exit 1
fi

# Check Python deps
print_info "Checking Python dependencies..."
if ! python3 -c "import torch, onnx, onnxsim" 2>/dev/null; then
    print_warning "Missing dependencies, installing: torch torchvision onnx onnxsim ..."
    pip install torch torchvision onnx onnxsim
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
else
    print_success "Python dependencies OK"
fi

# PT -> ONNX
print_info "Starting PT -> ONNX conversion..."

# Locate python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pt_to_onnx.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build python args
PY_ARGS=(
    --input "$INPUT_PT"
    --output "$OUTPUT_ONNX"
    --imgsz "$IMGSZ"
    --opset "$OPSET"
)

if [ $NO_SIMPLIFY -eq 1 ]; then
    PY_ARGS+=(--no-simplify)
fi
if [ $KEEP_INTERMEDIATE -eq 1 ]; then
    PY_ARGS+=(--keep-intermediate)
fi
if [ $VERBOSE -eq 1 ]; then
    PY_ARGS+=(--verbose)
fi

python3 "$PYTHON_SCRIPT" "${PY_ARGS[@]}"

if [ $? -eq 0 ] && [ -f "$OUTPUT_ONNX" ]; then
    print_success "PT -> ONNX conversion completed"
    print_info "ONNX size: $(du -h "$OUTPUT_ONNX" | cut -f1)"
else
    print_error "PT -> ONNX conversion failed"
    exit 1
fi

# Done
echo ""
print_success "============== Finished =============="
print_info "Input: $INPUT_PT ($(du -h "$INPUT_PT" | cut -f1))"
print_info "ONNX: $OUTPUT_ONNX ($(du -h "$OUTPUT_ONNX" | cut -f1))"
print_info "Image size: ${IMGSZ}x${IMGSZ}"