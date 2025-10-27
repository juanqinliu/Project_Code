#!/bin/bash

# Global Detection PT Model to ONNX Export Script
# Usage: ./export_global_onnx.sh [options]

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Show help information
show_help() {
    echo "Global Detection PT Model to ONNX Export Script"
    echo ""
    echo "Usage:"
    echo "  $0 -i <model_path> [options]"
    echo ""
    echo "Required Options:"
    echo "  -i, --input PATH      Input PT model file path"
    echo ""
    echo "Optional Parameters:"
    echo "  -o, --output PATH     Output ONNX file path"
    echo "  -s, --static          Generate static ONNX model (recommended for TensorRT)"
    echo "  -d, --dynamic         Generate dynamic ONNX model (default)"
    echo "  --imgsz SIZE          Input image size (default: 640)"
    echo "  -b, --batch-size SIZE Batch size (default: 1)"
    echo "  --opset VERSION       ONNX opset version (default: 12)"
    echo "  --simplify            Simplify ONNX model (default: enabled)"
    echo "  --no-simplify         Do not simplify ONNX model"
    echo "  --half                Use FP16 precision (only effective on GPU)"
    echo "  --fp32                Force FP32 precision"
    echo "  --device DEVICE       Device to use (default: cpu)"
    echo "  --gpu                 Use GPU device (equivalent to --device cuda)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i weights/global.pt --static --simplify"
    echo "  $0 --input weights/global.pt --gpu --half --imgsz 640"
    echo "  $0 -i weights/global.pt --output model_onnx/global.onnx --static"
}

# Check Python environment
check_python_env() {
    print_info "Checking Python environment..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please ensure Python 3.7+ is installed"
        exit 1
    fi
    
    # Check required Python packages
    local missing_packages=()
    
    python3 -c "import torch" 2>/dev/null || missing_packages+=("torch")
    python3 -c "import onnx" 2>/dev/null || missing_packages+=("onnx")
    python3 -c "import ultralytics" 2>/dev/null || missing_packages+=("ultralytics")
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_error "Missing required Python packages: ${missing_packages[*]}"
        print_info "Please install using:"
        echo "pip3 install torch onnx ultralytics"
        exit 1
    fi
    
    print_success "Python environment check passed"
}

# Check model file
check_model_file() {
    local model_path="$1"
    
    if [ ! -f "$model_path" ]; then
        print_error "Model file does not exist: $model_path"
        exit 1
    fi
    
    if [[ ! "$model_path" =~ \.(pt|pth)$ ]]; then
        print_error "Unsupported model file format. Please use .pt or .pth files"
        exit 1
    fi
    
    print_success "Model file check passed: $model_path"
}

# Main conversion function
convert_model() {
    local model_path="$1"
    local output_path="$2"
    local static_flag="$3"
    local imgsz="$4"
    local batch_size="$5"
    local opset="$6"
    local simplify_flag="$7"
    local half_flag="$8"
    local device="$9"
    
    print_info "Starting model conversion..."
    
    # Get script directory
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Build Python command using array for safety
    local python_cmd=(
        "python3"
        "$script_dir/pt_to_onnx.py"
        "--model" "$model_path"
    )
    
    if [ -n "$output_path" ]; then
        python_cmd+=("--output" "$output_path")
    fi
    
    if [ "$static_flag" = "true" ]; then
        python_cmd+=("--static")
    else
        python_cmd+=("--dynamic")
    fi
    
    python_cmd+=("--imgsz" "$imgsz")
    python_cmd+=("--batch-size" "$batch_size")
    python_cmd+=("--opset" "$opset")
    
    if [ "$simplify_flag" = "true" ]; then
        python_cmd+=("--simplify")
    else
        python_cmd+=("--no-simplify")
    fi
    
    if [ "$half_flag" = "true" ]; then
        python_cmd+=("--half")
    fi
    
    python_cmd+=("--device" "$device")
    
    print_info "Executing command: ${python_cmd[*]}"
    
    # Execute conversion
    if "${python_cmd[@]}"; then
        print_success "Model conversion completed!"
        return 0
    else
        print_error "Model conversion failed!"
        return 1
    fi
}

# Main function
main() {
    # Default parameters
    local model_path=""
    local output_path=""
    local static_flag="false"
    local imgsz=640
    local batch_size=1
    local opset=12
    local simplify_flag="true"
    local half_flag="false"
    local device="0"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--input)
                model_path="$2"
                shift 2
                ;;
            -o|--output)
                output_path="$2"
                shift 2
                ;;
            -s|--static)
                static_flag="true"
                shift
                ;;
            -d|--dynamic)
                static_flag="false"
                shift
                ;;
            --imgsz)
                imgsz="$2"
                shift 2
                ;;
            -b|--batch-size)
                batch_size="$2"
                shift 2
                ;;
            --opset)
                opset="$2"
                shift 2
                ;;
            --simplify)
                simplify_flag="true"
                shift
                ;;
            --no-simplify)
                simplify_flag="false"
                shift
                ;;
            --half)
                half_flag="true"
                shift
                ;;
            --fp32)
                half_flag="false"
                shift
                ;;
            --device)
                device="$2"
                shift 2
                ;;
            --gpu)
                device="cuda"
                shift
                ;;
            -* )
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            * )
                print_error "Unexpected positional argument: $1"
                print_error "Please use -i/--input flag to specify the model path"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check required parameters
    if [ -z "$model_path" ]; then
        print_error "Please specify model file path using -i/--input flag"
        show_help
        exit 1
    fi

    # Calculate default output path if not specified
    if [ -z "$output_path" ]; then
        local model_dir
        model_dir="$(dirname "$model_path")"
        local base_name
        base_name="$(basename "$model_path")"
        base_name="${base_name%.*}"
        output_path="$model_dir/${base_name}.onnx"
        print_info "Output path not specified, automatically set to: $output_path"
    fi
    
    # Check Python environment
    check_python_env
    
    # Check model file
    check_model_file "$model_path"
    
    # Execute conversion
    convert_model "$model_path" "$output_path" "$static_flag" "$imgsz" "$batch_size" "$opset" "$simplify_flag" "$half_flag" "$device"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi