## Overview

`export_global.sh` is a one-step model export script that can directly convert PyTorch (.pt) models to TensorRT Engine format using high-performance C++ TensorRT engine conversion.

The script integrates the following two steps:
1. **PT → ONNX**: Using `export_onnx/export_onnx.sh` 
2. **ONNX → Engine**: Using `export_engine/export_engine.sh`

## Directory Structure

```
export_global_c++/
├── export_global.sh             # Main unified export script (one-step PT to Engine)
├── README.md                    # Project documentation and usage guide
├── weights/                     # Model files directory
├── export_onnx/                 # PT to ONNX conversion module
│   ├── export_onnx.sh           # Shell script for ONNX export
│   └── pt_to_onnx.py            # Python script for PT to ONNX conversion
└── export_engine/               # ONNX to Engine conversion module (C++ implementation)
    ├── export_engine.sh         # Shell script for Engine export
    ├── build/                   # C++ build directory and compiled binaries
    ├── src/                     # C++ source code
    ├── include/                 # C++ header files
    ├── calib_data/              # Calibration images directory (1348 images)
    ├── calib_list.txt           # Calibration image list file
    ├── CMakeLists.txt           # CMake build configuration
    └── build.sh                 # Build script
```


## Quick Start

### Basic Usage

```bash
# Simplest usage (FP32 precision)
./export_global.sh -i weights/global_fixed_best_train63.pt

# FP16 precision conversion
./export_global.sh -i weights/global_fixed_best_train63.pt --precision fp16

# INT8 precision conversion (requires calibration data)
./export_global.sh -i weights/global_fixed_best_train63.pt --precision int8 --calib-dir export_engine/calib_data/ --calib-num 200
```



### Using Scripts Individually

```bash
# Export ONNX model
./export_onnx/export_onnx.sh -i weights/model.pt

# Export engine model with FP16 precision
./export_engine/export_engine.sh -i weights/model.onnx --precision fp16

# Export engine model with INT8 precision
./export_engine/export_engine.sh -i weights/model.onnx --precision int8 --calib-dir export_engine/calib_data/ --calib-num 200

```

## Complete Parameter Reference

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `-i, --input PATH` | Input PT model file path |

### Output Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-o, --output PATH` | Output Engine file path | Auto-generated (based on input filename and precision) |
| `--onnx-output PATH` | Intermediate ONNX file path | Auto-generated (same name as input file with .onnx extension) |


### ONNX Export Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-s, --static` | Generate static ONNX model (recommended for TensorRT) | - |
| `-d, --dynamic` | Generate dynamic ONNX model | Default |
| `--imgsz SIZE` | Input image size | 640 |
| `-b, --batch-size SIZE` | Batch size | 1 |
| `--opset VERSION` | ONNX opset version | 12 |
| `--simplify` | Simplify ONNX model | Default enabled |
| `--no-simplify` | Do not simplify ONNX model | - |
| `--device DEVICE` | Device for ONNX export | 0 |

### TensorRT Engine Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--precision TYPE` | Precision type (fp32/fp16/int8) | fp32 |
| `--workspace SIZE` | Workspace size (GB) | 2.0 |

### INT8 Calibration Configuration

| Parameter | Description | Default | Internal Mapping |
|-----------|-------------|---------|------------------|
| `--calib-list PATH` | Calibration image list file path | - | `--calib-list` |
| `--calib-dir PATH` | Calibration image directory path | - | `--calib-data` |
| `--calib-num NUM` | Maximum calibration image count | 50 | `--calib-limit` |


