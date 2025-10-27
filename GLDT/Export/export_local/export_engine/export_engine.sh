#!/bin/bash

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
CALIB_DIR="$SCRIPT_DIR/calib_data"
CALIB_LIST="$SCRIPT_DIR/calib_list.txt"
PRECISION="int8"  # Default to INT8 precision

# Options compatible with export_engine.sh
SKIP_BUILD=false          # kept for compatibility; no build in Python flow
AUTO_CALIB=false
CALIB_NUM=1000
CALIB_NUM_SET=false       # Track if user explicitly set --calib-num

show_help() {
    echo "ONNX to TensorRT Engine Export Tool (Python)"
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
    echo "  --imgsz <size>            Input image size (default: 640)"
    echo "  --calib-dir <dir>         Calibration data directory (default: ./calib_data)"
    echo "  --calib-list <file>       Calibration list file (default: ./calib_list.txt)"
    echo "  --precision <mode>        Precision mode: int8, fp16 or fp32 (default: int8)"
    echo "  --auto-calib-list         Force regenerate calibration list from calib_dir"
    echo "  --calib-num <n>           Max calibration samples (default: 1000)"
    echo "                            Regenerates list if count differs from existing"
    echo "  --skip-build              Kept for compatibility (no-op in Python)"
    echo "  -h, --help                Show help information"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_ONNX="$2"; shift 2 ;;
        -o|--output)
            OUTPUT_ENGINE="$2"; shift 2 ;;
        --batch-min)
            MIN_BATCH="$2"; shift 2 ;;
        --batch-opt)
            OPT_BATCH="$2"; shift 2 ;;
        --batch-max)
            MAX_BATCH="$2"; shift 2 ;;
        --workspace)
            WORKSPACE="$2"; shift 2 ;;
        --imgsz)
            IMGSZ="$2"; shift 2 ;;
        --calib-dir)
            CALIB_DIR="$2"; shift 2 ;;
        --calib-list)
            CALIB_LIST="$2"; shift 2 ;;
        --precision)
            PRECISION="$2"; shift 2 ;;
        --auto-calib-list)
            AUTO_CALIB=true; shift ;;
        --calib-num)
            CALIB_NUM="$2"
            CALIB_NUM_SET=true
            shift 2 ;;
        --skip-build)
            SKIP_BUILD=true; shift ;;
        -h|--help)
            show_help; exit 0 ;;
        *)
            print_error "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
done

print_info "=== ONNX to TensorRT Engine Export Tool (Python) ==="
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

# Validate parameters and normalize paths
if [[ "$INPUT_ONNX" != /* ]]; then INPUT_ONNX="$(pwd)/$INPUT_ONNX"; fi

# Auto-generate output filename with precision suffix if using default name
if [[ "$OUTPUT_ENGINE" == "local.engine" ]]; then
    ONNX_DIR=$(dirname "$INPUT_ONNX")
    ONNX_BASENAME=$(basename "$INPUT_ONNX" .onnx)
    OUTPUT_ENGINE="${ONNX_DIR}/${ONNX_BASENAME}_${PRECISION}.engine"
    print_info "Auto-generated output filename: $OUTPUT_ENGINE"
else
    # If user specified output file, check if it needs precision suffix
    if [[ "$OUTPUT_ENGINE" != /* ]]; then 
        OUTPUT_ENGINE="$(pwd)/$OUTPUT_ENGINE"
    fi
    
    # Add precision suffix if not already present and not explicitly disabled
    ENGINE_BASE="${OUTPUT_ENGINE%.engine}"
    if [[ ! "$ENGINE_BASE" =~ _(fp32|fp16|int8)$ ]]; then
        OUTPUT_ENGINE="${ENGINE_BASE}_${PRECISION}.engine"
        print_info "Added precision suffix to output: $OUTPUT_ENGINE"
    fi
fi

if [ ! -f "$INPUT_ONNX" ]; then
    print_error "Input file $INPUT_ONNX does not exist"; exit 1
fi

if [ $MIN_BATCH -lt 1 ] || [ $OPT_BATCH -lt $MIN_BATCH ] || [ $MAX_BATCH -lt $OPT_BATCH ]; then
    print_error "Invalid batch size range"; exit 1
fi

if [ $WORKSPACE -lt 1024 ]; then
    print_warning "Workspace less than 1024MB may affect performance"
fi

# INT8 calibration handling
if [[ "$PRECISION" == "int8" ]]; then
    if [ ! -d "$CALIB_DIR" ]; then
        print_error "Calibration data directory $CALIB_DIR does not exist"; exit 1
    fi

    # Generate calib list based on user intent
    SHOULD_REGENERATE=false
    
    # Determine if we should regenerate the list
    if [[ "$AUTO_CALIB" == true ]]; then
        # Force regenerate when --auto-calib-list is specified
        SHOULD_REGENERATE=true
        print_info "Force regenerating calibration list (--auto-calib-list): $CALIB_LIST"
    elif [[ "$CALIB_NUM_SET" == true ]]; then
        # If user specified --calib-num, check if we need to regenerate
        if [ -f "$CALIB_LIST" ]; then
            EXISTING_COUNT=$(wc -l < "$CALIB_LIST")
            if [ "$EXISTING_COUNT" -ne "$CALIB_NUM" ]; then
                SHOULD_REGENERATE=true
                print_info "Regenerating calibration list (count mismatch: $EXISTING_COUNT → $CALIB_NUM)"
            else
                print_info "Using existing calibration list: $CALIB_LIST ($EXISTING_COUNT entries)"
            fi
        else
            SHOULD_REGENERATE=true
            print_info "Generating calibration list (file not found): $CALIB_LIST"
        fi
    elif [ ! -f "$CALIB_LIST" ]; then
        # Auto-generate if file doesn't exist
        SHOULD_REGENERATE=true
        print_info "Calibration list not found, auto-generating: $CALIB_LIST (max $CALIB_NUM entries)"
    else
        # Use existing file
        EXISTING_COUNT=$(wc -l < "$CALIB_LIST")
        print_info "Using existing calibration list: $CALIB_LIST ($EXISTING_COUNT entries)"
        print_info "To regenerate with different count, specify --calib-num <n> or use --auto-calib-list"
    fi
    
    # Generate list if needed
    if [[ "$SHOULD_REGENERATE" == true ]]; then
        (cd "$CALIB_DIR" && find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) \
            | sed 's#^\./##' \
            | shuf \
            | head -n "$CALIB_NUM") > "$CALIB_LIST"
        if [ ! -s "$CALIB_LIST" ]; then
            print_error "No usable images found in $CALIB_DIR to generate calibration list"; exit 1
        fi
        FINAL_COUNT=$(wc -l < "$CALIB_LIST")
        print_success "Calibration list generated: $FINAL_COUNT entries"
    fi

    if [ ! -f "$CALIB_LIST" ]; then
        print_error "Calibration list file $CALIB_LIST does not exist"; exit 1
    fi
fi

# Skip build (compatibility only)
if [ "$SKIP_BUILD" = true ]; then
    print_warning "Skipping C++ program compilation (Python flow does not require build)"
fi

# Execute Python conversion
PY_MAIN="$SCRIPT_DIR/onnx_to_engine.py"

if [ ! -f "$PY_MAIN" ]; then
    print_error "Python converter not found: $PY_MAIN"; exit 1
fi

print_info "Starting ONNX -> TensorRT Engine conversion (Python)..."

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

if [[ "$PRECISION" == "int8" ]]; then
    CMD_ARGS+=("--calib-dir" "$CALIB_DIR" "--calib-list" "$CALIB_LIST")
fi

print_info "Executing command: python3 $PY_MAIN ${CMD_ARGS[*]}"

python3 "$PY_MAIN" "${CMD_ARGS[@]}"
ret=$?

if [ $ret -eq 0 ] && [ -f "$OUTPUT_ENGINE" ]; then
    print_success "ONNX -> TensorRT Engine conversion completed"
    print_info "Engine file size: $(du -h "$OUTPUT_ENGINE" | cut -f1)"
else
    print_error "ONNX -> TensorRT Engine conversion failed"
    exit 1
fi

echo ""
print_success "============== Conversion Complete =============="
print_info "Input file: $INPUT_ONNX ($(du -h "$INPUT_ONNX" | cut -f1))"
print_info "Engine file: $OUTPUT_ENGINE ($(du -h "$OUTPUT_ENGINE" | cut -f1))"
print_info "Features: ${PRECISION^^} precision + dynamic batching ($MIN_BATCH-$MAX_BATCH)"
print_info "Recommended batch size $OPT_BATCH for optimal performance"


