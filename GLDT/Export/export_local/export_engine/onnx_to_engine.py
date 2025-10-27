#!/usr/bin/env python3
"""
ONNX to TensorRT Engine Converter - Python Implementation

This Python implementation provides equivalent functionality to the C++ version
for converting ONNX models to TensorRT engines with support for:
- FP32, FP16, and INT8 precision modes
- Dynamic batching with configurable batch sizes
- INT8 calibration with custom calibrator
- Comprehensive error handling and logging

Author: AI Assistant
Date: 2024
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
except ImportError as e:
    print(f"Error: TensorRT or PyCUDA not found. Please install: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TensorRTLogger(trt.ILogger):
    """Custom TensorRT logger to handle TensorRT logging messages"""
    
    def __init__(self):
        trt.ILogger.__init__(self)
    
    def log(self, severity, msg):
        """Log TensorRT messages with appropriate Python logging levels"""
        if severity == trt.ILogger.Severity.INTERNAL_ERROR:
            logger.error(f"[TRT-INTERNAL_ERROR] {msg}")
        elif severity == trt.ILogger.Severity.ERROR:
            logger.error(f"[TRT-ERROR] {msg}")
        elif severity == trt.ILogger.Severity.WARNING:
            logger.warning(f"[TRT-WARNING] {msg}")
        elif severity == trt.ILogger.Severity.INFO:
            logger.info(f"[TRT-INFO] {msg}")
        elif severity == trt.ILogger.Severity.VERBOSE:
            logger.debug(f"[TRT-VERBOSE] {msg}")

class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator for TensorRT
    
    This class implements the INT8 calibration interface required by TensorRT
    for quantizing models to INT8 precision. It loads calibration images,
    preprocesses them, and provides batches to TensorRT during calibration.
    """
    
    def __init__(self, 
                 calib_list_file: str,
                 calib_data_path: str,
                 batch_size: int,
                 input_height: int,
                 input_width: int,
                 input_channels: int = 3,
                 calib_table_name: str = "calibration.table"):
        """
        Initialize the INT8 calibrator
        
        Args:
            calib_list_file: Path to file containing list of calibration images
            calib_data_path: Directory containing calibration images
            batch_size: Batch size for calibration
            input_height: Input image height
            input_width: Input image width
            input_channels: Number of input channels (default: 3 for RGB)
            calib_table_name: Name of the calibration cache file
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.calib_data_path = Path(calib_data_path)
        self.calib_table_name = calib_table_name
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.image_index = 0
        
        # Calculate input size
        self.input_size = batch_size * input_channels * input_height * input_width
        
        # Load calibration images
        self.image_files = self._load_calibration_list(calib_list_file)
        
        # Check for existing calibration cache
        if os.path.exists(calib_table_name):
            logger.info(f"Found existing calibration cache: {calib_table_name}")
            logger.info("Skipping image loading, will use cache file directly")
        else:
            logger.info("No calibration cache found, will perform new calibration")
            logger.info(f"Loaded {len(self.image_files)} calibration images")
        
        # Allocate GPU memory
        self.device_input = cuda.mem_alloc(self.input_size * 4)  # 4 bytes per float32
        self.host_input = np.zeros((batch_size, input_channels, input_height, input_width), 
                                 dtype=np.float32)
        
        logger.info(f"INT8 calibrator initialization complete")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Input dimensions: {input_channels}x{input_height}x{input_width}")
    
    def _load_calibration_list(self, calib_list_file: str) -> List[str]:
        """Load the list of calibration images from file"""
        try:
            with open(calib_list_file, 'r') as f:
                image_files = [line.strip() for line in f if line.strip()]
            
            # Shuffle for better calibration results
            np.random.shuffle(image_files)
            
            logger.info(f"Loaded {len(image_files)} calibration images from {calib_list_file}")
            return image_files
            
        except Exception as e:
            logger.error(f"Failed to load calibration list: {e}")
            return []
    
    def get_batch_size(self) -> int:
        """Return the batch size for calibration"""
        return self.batch_size
    
    def get_batch(self, names: List[str]) -> List[int]:
        """
        Get the next batch of calibration data
        
        Args:
            names: List of input tensor names
            
        Returns:
            List of GPU memory pointers for the batch data
        """
        # If no image files (cache mode), return empty
        if not self.image_files:
            logger.info("Using calibration cache mode, skipping batch processing")
            return []
        
        if self.image_index + self.batch_size > len(self.image_files):
            return []  # No more data
        
        # Clear host buffer
        self.host_input.fill(0.0)
        
        # Load and preprocess a batch of images
        valid_images = 0
        for i in range(self.batch_size):
            if self.image_index + i >= len(self.image_files):
                break
            
            image_path = self.calib_data_path / self.image_files[self.image_index + i]
            
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.warning(f"Unable to load image: {image_path}")
                    continue
                
                # Validate image
                if img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] != 3:
                    logger.warning(f"Invalid image format: {image_path}")
                    continue
                
                # Preprocess image
                self._preprocess_image(img, valid_images)
                valid_images += 1
                
            except Exception as e:
                logger.warning(f"Image preprocessing failed: {image_path}, error: {e}")
                continue
        
        if valid_images == 0:
            logger.error("No valid images in batch")
            return []
        
        # Copy data to GPU
        cuda.memcpy_htod(self.device_input, self.host_input)
        
        self.image_index += self.batch_size
        
        logger.info(f"Calibration progress: {self.image_index}/{len(self.image_files)} "
                   f"images (valid: {valid_images})")
        
        return [int(self.device_input)]
    
    def _preprocess_image(self, img: np.ndarray, batch_idx: int):
        """
        Preprocess a single image for calibration
        
        Args:
            img: Input image as numpy array
            batch_idx: Index in the batch
        """
        # Resize image
        resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        chw = np.transpose(normalized, (2, 0, 1))
        
        # Store in host buffer
        self.host_input[batch_idx] = chw
    
    def read_calibration_cache(self) -> bytes:
        """Read calibration cache from file"""
        if not os.path.exists(self.calib_table_name):
            return b""
        
        with open(self.calib_table_name, 'rb') as f:
            cache = f.read()
        
        logger.info(f"Read calibration cache: {self.calib_table_name} ({len(cache)} bytes)")
        return cache
    
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache to file"""
        with open(self.calib_table_name, 'wb') as f:
            f.write(cache)
        
        logger.info(f"Write calibration cache: {self.calib_table_name} ({len(cache)} bytes)")

class ONNXToTensorRTConverter:
    """
    Main converter class for ONNX to TensorRT engine conversion
    
    This class handles the complete conversion pipeline including:
    - ONNX model parsing
    - TensorRT engine building
    - Dynamic shape configuration
    - Precision mode setting
    - Engine serialization
    """
    
    def __init__(self, logger: TensorRTLogger):
        """
        Initialize the converter
        
        Args:
            logger: TensorRT logger instance
        """
        self.logger = logger
        self.builder = None
        self.network = None
        self.config = None
        self.parser = None
    
    def build_engine(self,
                    onnx_file: str,
                    engine_file: str,
                    calib_data_dir: str = "",
                    calib_list: str = "",
                    min_batch: int = 1,
                    opt_batch: int = 16,
                    max_batch: int = 32,
                    imgsz: int = 640,
                    workspace_mb: int = 2048,
                    precision: str = "int8") -> bool:
        """
        Build TensorRT engine from ONNX model
        
        Args:
            onnx_file: Path to input ONNX file
            engine_file: Path to output engine file
            calib_data_dir: Directory containing calibration images (for INT8)
            calib_list: File containing list of calibration images (for INT8)
            min_batch: Minimum batch size
            opt_batch: Optimal batch size
            max_batch: Maximum batch size
            imgsz: Input image size
            workspace_mb: Workspace size in MB
            precision: Precision mode ("fp32", "fp16", "int8")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("\n=== Building TensorRT Engine ===")
            logger.info(f"ONNX file: {onnx_file}")
            logger.info(f"Engine file: {engine_file}")
            logger.info(f"Batch size range: {min_batch} - {max_batch} (optimal: {opt_batch})")
            logger.info(f"Input size: {imgsz}x{imgsz}")
            logger.info(f"Workspace size: {workspace_mb} MB")
            logger.info(f"Precision mode: {precision.upper()}")
            
            # Create TensorRT builder
            self.builder = trt.Builder(self.logger)
            if self.builder is None:
                logger.error("Failed to create TensorRT builder")
                return False
            
            # Log TRT version and try creating network with best-effort compatibility
            try:
                logger.info(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
            except Exception:
                pass

            self.network = None
            # Prefer explicit batch; try multiple API signatures for compatibility across TRT versions
            try:
                # Preferred: use enum flags directly (no shifting), support EXPLICIT_PRECISION when available
                flags = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_PRECISION'):
                    try:
                        flags |= int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
                        logger.info("Including EXPLICIT_PRECISION flag")
                    except Exception:
                        pass
                self.network = self.builder.create_network(flags)
                logger.info("Created network with explicit batch (enum flags API)")
            except Exception as e1:
                logger.warning(f"Explicit batch (enum flags) create_network failed: {e1}")
                try:
                    # Old-style bitflag usage (TRT 7/8)
                    explicit_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                    self.network = self.builder.create_network(explicit_flag)
                    logger.info("Created network with explicit batch (bitflag API)")
                except Exception as e2:
                    logger.warning(f"Explicit batch (bitflag) create_network failed: {e2}")
                    try:
                        # Fallback to implicit batch (no dynamic shapes); still usable for static-batch models
                        self.network = self.builder.create_network(0)
                        logger.warning("Fallback to implicit-batch network (dynamic shapes disabled)")
                    except Exception as e3:
                        logger.error(f"All create_network attempts failed: {e3}")
                        self.network = None

            if self.network is None:
                logger.error("Failed to create network definition (check TensorRT Python/C++ version match)")
                return False
            
            # Create ONNX parser
            self.parser = trt.OnnxParser(self.network, self.logger)
            if self.parser is None:
                logger.error("Failed to create ONNX parser")
                return False
            
            # Parse ONNX file
            logger.info("\n=== Parsing ONNX Model ===")
            with open(onnx_file, 'rb') as model:
                if not self.parser.parse(model.read()):
                    logger.error("ONNX parsing failed")
                    for error in range(self.parser.num_errors):
                        logger.error(f"Parser error {error}: {self.parser.get_error(error)}")
                    return False
            
            # Print network information
            self._print_network_info()
            
            # Create build configuration
            self.config = self.builder.create_builder_config()
            if self.config is None:
                logger.error("Failed to create build configuration")
                return False
            
            # Set workspace size
            workspace_bytes = workspace_mb * 1024 * 1024
            self.config.max_workspace_size = workspace_bytes
            
            # Create optimization profiles
            logger.info("\n=== Creating Optimization Profile ===")

            if precision == "int8":
                # Profile 0: calibration-only profile (min=opt=max=batch_min)
                calib_profile = self.builder.create_optimization_profile()
                if calib_profile is None:
                    logger.error("Failed to create calibration optimization profile")
                    return False
                if not self._set_dynamic_shapes(calib_profile, min_batch, min_batch, min_batch, imgsz):
                    logger.error("Failed to set calibration profile shapes")
                    return False
                self.config.add_optimization_profile(calib_profile)

                # Profile 1: runtime dynamic profile (min/opt/max per user)
                runtime_profile = self.builder.create_optimization_profile()
                if runtime_profile is None:
                    logger.error("Failed to create runtime optimization profile")
                    return False
                if not self._set_dynamic_shapes(runtime_profile, min_batch, opt_batch, max_batch, imgsz):
                    logger.error("Failed to set runtime profile shapes")
                    return False
                self.config.add_optimization_profile(runtime_profile)
            else:
                # Single dynamic profile for FP16/FP32
                profile = self.builder.create_optimization_profile()
                if profile is None:
                    logger.error("Failed to create optimization profile")
                    return False
                if not self._set_dynamic_shapes(profile, min_batch, opt_batch, max_batch, imgsz):
                    logger.error("Failed to set dynamic shapes")
                    return False
                self.config.add_optimization_profile(profile)
            
            # Set precision flags
            calibrator = None
            if precision == "fp16":
                logger.info("\n=== Using FP16 Precision ===")
                self.config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                logger.info("\n=== Configuring INT8 Quantization ===")
                self.config.set_flag(trt.BuilderFlag.INT8)
                
                if not self.builder.platform_has_fast_int8:
                    logger.warning("Platform does not support fast INT8")
                
                # Create INT8 calibrator
                calibrator = Int8EntropyCalibrator(
                    calib_list, calib_data_dir, min_batch, imgsz, imgsz, 3, "calibration.table"
                )
                self.config.int8_calibrator = calibrator
            else:
                logger.info("\n=== Using FP32 Precision ===")
            
            # Build engine
            logger.info("\n=== Building Engine ===")
            logger.info("This may take several minutes...")
            
            serialized_engine = self.builder.build_serialized_network(self.network, self.config)
            if serialized_engine is None:
                logger.error("Engine build failed")
                return False
            
            # Save engine file (handle IHostMemory vs bytes)
            logger.info("\n=== Saving Engine ===")
            try:
                if isinstance(serialized_engine, (bytes, bytearray)):
                    engine_bytes = bytes(serialized_engine)
                else:
                    # TensorRT IHostMemory supports buffer protocol
                    engine_bytes = memoryview(serialized_engine).tobytes()
            except Exception:
                # Fallback: try .tobytes() or .data/.size
                try:
                    engine_bytes = serialized_engine.tobytes()  # type: ignore[attr-defined]
                except Exception:
                    # Last resort: read via pointer+size not exposed in Python; rethrow
                    raise

            with open(engine_file, 'wb') as f:
                f.write(engine_bytes)

            # Validate engine (use bytes for deserialization)
            self._validate_engine(engine_bytes)

            logger.info("\n=== Build Complete ===")
            try:
                if hasattr(serialized_engine, 'size'):
                    sz = serialized_engine.size  # type: ignore[attr-defined]
                    try:
                        # Some versions expose size as method
                        sz = serialized_engine.size()  # type: ignore[call-arg]
                    except Exception:
                        pass
                else:
                    sz = len(engine_bytes)
                logger.info(f"Engine size: {float(sz) / (1024.0 * 1024.0):.2f} MB")
            except Exception:
                logger.info(f"Engine saved: {engine_file}")
            logger.info(f"Precision mode: {precision.upper()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Engine build failed with exception: {e}")
            return False
    
    def _print_network_info(self):
        """Print network information"""
        logger.info("\n=== Network Information ===")
        logger.info(f"Number of inputs: {self.network.num_inputs}")
        logger.info(f"Number of outputs: {self.network.num_outputs}")
        
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            dims = input_tensor.shape
            logger.info(f"\nInput {i}: {input_tensor.name}")
            logger.info(f"Shape: {dims}")
            logger.info(f"Data type: {input_tensor.dtype}")
    
    def _set_dynamic_shapes(self, profile, min_batch, opt_batch, max_batch, imgsz):
        """Set dynamic shapes for input tensors"""
        all_dynamic_set = True
        
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            original_shape = input_tensor.shape
            
            logger.info(f"\nSetting dynamic range for input {input_tensor.name}:")
            logger.info(f"Original shape: {original_shape}")
            
            # Create dynamic shapes
            min_shape = list(original_shape)
            opt_shape = list(original_shape)
            max_shape = list(original_shape)
            
            # Set batch dimension
            min_shape[0] = max(1, min_batch)
            opt_shape[0] = max(min_batch, min(opt_batch, max_batch))
            max_shape[0] = max(opt_batch, max_batch)
            
            # Set other dimensions
            for d in range(1, len(original_shape)):
                if original_shape[d] == -1:  # Dynamic dimension
                    min_shape[d] = 1
                    opt_shape[d] = imgsz
                    max_shape[d] = max(imgsz, 1024)
                else:  # Static dimension
                    min_shape[d] = original_shape[d]
                    opt_shape[d] = original_shape[d]
                    max_shape[d] = original_shape[d]
            
            logger.info(f"Minimum shape: {min_shape}")
            logger.info(f"Optimal shape: {opt_shape}")
            logger.info(f"Maximum shape: {max_shape}")
            
            # Validate shapes
            for shape, name in [(min_shape, "Minimum"), (opt_shape, "Optimal"), (max_shape, "Maximum")]:
                for j, dim in enumerate(shape):
                    if dim <= 0:
                        logger.error(f"Error: {name} dimension must be > 0, dimension {j} is {dim}")
                        all_dynamic_set = False
                        break
            
            if not all_dynamic_set:
                continue
            
            # Set shapes in profile
            try:
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                logger.info("✅ Dynamic range set successfully")
            except Exception as e:
                logger.error(f"Failed to set dynamic range: {e}")
                all_dynamic_set = False
        
        if not all_dynamic_set:
            logger.error("Failed to set dynamic ranges")
            return False
        
        # Validate profile (TensorRT 8.6 Python API may not expose is_valid)
        try:
            has_is_valid = hasattr(profile, 'is_valid')
            if has_is_valid:
                if not profile.is_valid():
                    logger.error("Optimization profile is invalid")
                    return False
            else:
                logger.info("Optimization profile 'is_valid' not available; assuming valid after set_shape")
        except Exception as e:
            logger.warning(f"Optimization profile validation skipped due to exception: {e}")
        
        return True
    
    def _validate_engine(self, serialized_engine):
        """Validate the built engine"""
        logger.info("\n=== Validating Engine ===")
        
        try:
            # Create runtime and deserialize engine
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            if not engine:
                logger.error("Failed to deserialize engine")
                return
            
            logger.info(f"Number of bindings: {engine.num_bindings}")
            
            has_dynamic_shapes = False
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                is_input = engine.binding_is_input(i)
                shape = engine.get_binding_shape(i)
                
                logger.info(f"\nBinding {i}: {name} ({'input' if is_input else 'output'})")
                logger.info(f"Shape: {shape}")
                
                # Check for dynamic shapes
                is_dynamic = any(dim == -1 for dim in shape)
                has_dynamic_shapes = has_dynamic_shapes or is_dynamic
                logger.info(f"Is dynamic: {'Yes' if is_dynamic else 'No'}")
            
            logger.info(f"Dynamic shapes support: {'Yes' if has_dynamic_shapes else 'No'}")
            
            if not has_dynamic_shapes:
                logger.warning("\nWarning: Engine does not support dynamic batching!")
                logger.warning("Suggestions:")
                logger.warning("1. Check if ONNX model correctly sets dynamic batch dimensions")
                logger.warning("2. Ensure TensorRT build configuration correctly sets dynamic ranges")
                logger.warning("3. Consider using a newer version of TensorRT")
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="ONNX to TensorRT Engine Converter")
    
    parser.add_argument("--input", "-i", required=True, help="Input ONNX file")
    parser.add_argument("--output", "-o", required=True, help="Output engine file")
    parser.add_argument("--calib-dir", help="Calibration data directory (for INT8)")
    parser.add_argument("--calib-list", help="Calibration list file (for INT8)")
    parser.add_argument("--batch-min", type=int, default=1, help="Minimum batch size")
    parser.add_argument("--batch-opt", type=int, default=16, help="Optimal batch size")
    parser.add_argument("--batch-max", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--workspace", type=int, default=2048, help="Workspace size in MB")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="int8",
                       help="Precision mode")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    if args.precision == "int8":
        if not args.calib_dir or not args.calib_list:
            logger.error("INT8 mode requires calibration data directory and list")
            return 1
        
        if not os.path.exists(args.calib_dir):
            logger.error(f"Calibration directory not found: {args.calib_dir}")
            return 1
        
        if not os.path.exists(args.calib_list):
            logger.error(f"Calibration list file not found: {args.calib_list}")
            return 1
    
    if args.batch_min < 1 or args.batch_opt < args.batch_min or args.batch_max < args.batch_opt:
        logger.error("Invalid batch size range")
        return 1
    
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        logger.error("Image size must be a multiple of 32")
        return 1
    
    if args.workspace < 1024:
        logger.warning("Workspace size less than 1024MB may affect performance")
    
    # Create converter and build engine
    logger.info("=== ONNX to TensorRT Engine Converter ===")
    
    trt_logger = TensorRTLogger()
    converter = ONNXToTensorRTConverter(trt_logger)
    
    success = converter.build_engine(
        onnx_file=args.input,
        engine_file=args.output,
        calib_data_dir=args.calib_dir or "",
        calib_list=args.calib_list or "",
        min_batch=args.batch_min,
        opt_batch=args.batch_opt,
        max_batch=args.batch_max,
        imgsz=args.imgsz,
        workspace_mb=args.workspace,
        precision=args.precision
    )
    
    if success:
        logger.info("✅ Conversion completed successfully!")
        return 0
    else:
        logger.error("❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
