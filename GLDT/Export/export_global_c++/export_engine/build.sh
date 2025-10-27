#!/bin/bash

# Create build directory
mkdir -p build && cd build

# Configure CMake
cmake ..

# Compile
make -j$(nproc)

# Print
if [ $? -eq 0 ]; then
    echo -e "\n\033[32mBuild Success!\033[0m"
    echo -e "Usage:"
    echo -e "  ./export_engine <onnx_file> --precision <type> [options]"
    echo -e "Example:"
    echo -e "  ./export_engine ../../global_model_640.onnx --precision fp32"
    echo -e "  ./export_engine ../../global_model_640.onnx --precision fp16"
    echo -e "  ./export_engine ../../global_model_640.onnx --precision int8 --calib-data ../calib_data"
    echo -e ""
    echo -e "Get full help information:"
    echo -e "  ./export_engine --help"
else
    echo -e "\n\033[31mBuild Failed! Please check the error information.\033[0m"
fi 