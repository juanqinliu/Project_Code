# Local Detection Model Export Tool

This tool provides a command to convert PyTorch models (.pt) directly to TensorRT engines (.engine), combining both PT→ONNX and ONNX→Engine conversion steps.

## Quick Start

```bash
# Basic usage - convert model.pt to model.engine with INT8 precision
./export_local.sh -i model.pt

# Convert with FP16 precision
./export_local.sh -i model.pt --precision fp16

# Convert with int8 precision
./export_local.sh -i model.pt --precision int8 --calib_data onnx_to_engine/calib_data

```

## Command Line Options

### Required
- `-i, --input <file>` - Input PT file
- `-o, --output <file>` - Output engine file

### Optional
- `--precision <fp32|fp16|int8>` - Precision of the engine to be exported (default: fp32)
- `--calib_data <dir>` - Directory containing calibration data for INT8 precision (default: None)
- `--batch_size <int>` - Batch size for engine inference (default: 1)
- `--max_workspace_size <int>` - Maximum workspace size for TensorRT engine (default: 1GB)
- `--verbose` - Enable verbose output


### PT to ONNX
```bash
./pt_to_onnx/export_onnx.sh -i weights/local.pt 

```

### ONNX to Engine
```bash
# fp16
./onnx_to_engine/export_engine.sh -i weights/local.onnx --precision fp16
# int8
./onnx_to_engine/export_engine.sh -i weights/local.onnx --precision int8 --calib_data onnx_to_engine/calib_data
```


