# Local Detection Model Export Tool

This tool provides a command to convert PyTorch models (.pt) directly to TensorRT engines (.engine), combining both PT→ONNX and ONNX→Engine conversion steps.

## Quick Start

```bash
# Basic usage - convert model.pt to model.engine with INT8 precision
./export_local.sh -i model.pt

# Convert with FP16 precision
./export_local.sh -i model.pt --precision fp16

# Keep intermediate ONNX file
./export_local.sh -i model.pt --keep-onnx

```

## Command Line Options

### Required
- `-i, --input <file>` - Input PT file


### PT to ONNX
```bash
./pt_to_onnx/export_onnx.sh -i weights/local.pt 

```

### ONNX to Engine
```bash
./onnx_to_engine/export_engine.sh -o weights/local.onnx --precision fp16
```


