#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ONNX to TensorRT Engine conversion tool
Supports FP32, FP16, INT8 precision with comprehensive error handling
"""

import os
import sys
import argparse
import logging
import glob
import random
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import time

import numpy as np
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    print(f"TensorRT version: {trt.__version__}")
    
    if not hasattr(trt, 'Builder') or not hasattr(trt, 'Logger'):
        raise ImportError("TensorRT APIs unavailable")
        
except ImportError as e:
    print(f"Error: TensorRT import failed - {e}")
    print("Please check TensorRT installation and CUDA compatibility")
    sys.exit(1)


class EngineConfig:
    """Engine build configuration"""
    
    def __init__(self):
        self.input_shape = (1, 3, 640, 640)  # NCHW
        self.max_workspace_size = 2 * 1024 * 1024 * 1024  # 2GB
        self.max_batch_size = 1
        
        self.use_fp16 = False
        self.use_int8 = False
        
        self.calib_batch_size = 1
        self.calib_max_images = 50
        self.calib_cache_file = None
        
        self.enable_dla = False
        self.dla_core = 0
        self.gpu_fallback = True
        
        self.verbose = False
        self.timing_cache = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'input_shape': self.input_shape,
            'max_workspace_size': self.max_workspace_size,
            'max_batch_size': self.max_batch_size,
            'use_fp16': self.use_fp16,
            'use_int8': self.use_int8,
            'calib_batch_size': self.calib_batch_size,
            'calib_max_images': self.calib_max_images,
            'enable_dla': self.enable_dla,
            'dla_core': self.dla_core,
            'gpu_fallback': self.gpu_fallback,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EngineConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class TensorRTLogger(trt.ILogger):
    """TensorRT logger"""
    
    def __init__(self, severity: trt.Logger.Severity = trt.Logger.INFO):
        trt.ILogger.__init__(self)
        self.severity = severity
        self.level_map = {
            trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
            trt.Logger.ERROR: logging.ERROR,
            trt.Logger.WARNING: logging.WARNING,
            trt.Logger.INFO: logging.INFO,
            trt.Logger.VERBOSE: logging.DEBUG
        }
        
    def log(self, severity: trt.Logger.Severity, msg: str):
        """Log message"""
        if severity <= self.severity:
            level = self.level_map.get(severity, logging.INFO)
            logging.log(level, f"[TensorRT] {msg}")


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 quantization calibrator"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 input_names: List[str],
                 input_shape: Tuple[int, int, int, int],
                 batch_size: int = 1,
                 cache_file: Optional[str] = None):
        super().__init__()
        
        self.image_paths = image_paths
        self.input_names = input_names
        self.input_shape = input_shape  # (N, C, H, W)
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0
        
        # Allocate device memory
        self.device_buffers = {}
        _, c, h, w = input_shape
        buffer_size = c * h * w * np.float32().nbytes
        
        for name in input_names:
            self.device_buffers[name] = cuda.mem_alloc(buffer_size)
            
        logging.info(f"INT8 calibrator initialized with {len(image_paths)} images")
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image"""
        try:
            _, c, h, w = self.input_shape
            
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Cannot read image: {image_path}")
                return np.zeros((c, h, w), dtype=np.float32)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = np.ascontiguousarray(image)
            
            return image
            
        except Exception as e:
            logging.error(f"Image preprocessing failed {image_path}: {e}")
            return np.zeros((c, h, w), dtype=np.float32)
    
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """Get calibration batch"""
        if self.current_index >= len(self.image_paths):
            logging.info("Calibration data exhausted")
            return None
        
        try:
            image_path = self.image_paths[self.current_index]
            processed_image = self.preprocess_image(image_path)
            
            if not processed_image.flags['C_CONTIGUOUS']:
                processed_image = np.ascontiguousarray(processed_image, dtype=np.float32)
            
            expected_shape = self.input_shape[1:]  # Remove batch dimension
            if processed_image.shape != expected_shape:
                logging.error(f"Shape mismatch: expected {expected_shape}, got {processed_image.shape}")
                return None
            
            device_ptrs = []
            for name in names:
                if name in self.device_buffers:
                    cuda.memcpy_htod(self.device_buffers[name], processed_image)
                    device_ptrs.append(int(self.device_buffers[name]))
                else:
                    logging.error(f"Input buffer not found: {name}")
                    return None
            
            self.current_index += 1
            
            if self.current_index % 50 == 0:
                logging.info(f"Calibration progress: {self.current_index}/{len(self.image_paths)}")
            
            return device_ptrs
            
        except Exception as e:
            logging.error(f"Failed to get calibration batch: {e}")
            import traceback
            logging.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read calibration cache"""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = f.read()
                logging.info(f"Read calibration cache: {self.cache_file}")
                return cache_data
            except Exception as e:
                logging.warning(f"Failed to read calibration cache: {e}")
        return None
    
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache"""
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)
                logging.info(f"Saved calibration cache: {self.cache_file}")
            except Exception as e:
                logging.error(f"Failed to save calibration cache: {e}")


class EngineBuilder:
    """TensorRT engine builder"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = TensorRTLogger(
            trt.Logger.VERBOSE if config.verbose else trt.Logger.INFO
        )
        
        trt.init_libnvinfer_plugins(self.logger, "")
        
        logging.info(f"TensorRT version: {trt.__version__}")
        logging.info(f"CUDA devices available: {cuda.Device.count()}")
    
    def collect_calibration_images(self, 
                                 calib_list: Optional[str] = None,
                                 calib_dir: Optional[str] = None) -> List[str]:
        """Collect calibration images"""
        image_paths = []
        
        if calib_list and os.path.isfile(calib_list):
            try:
                with open(calib_list, 'r', encoding='utf-8') as f:
                    paths = [line.strip() for line in f if line.strip()]
                    for path in paths:
                        if os.path.isfile(path):
                            image_paths.append(path)
                        else:
                            logging.warning(f"Calibration image not found: {path}")
                logging.info(f"Loaded {len(image_paths)} images from list file")
            except Exception as e:
                logging.error(f"Failed to read calibration list: {e}")
        
        if calib_dir and os.path.isdir(calib_dir):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            for ext in extensions:
                pattern = os.path.join(calib_dir, '**', ext)
                image_paths.extend(glob.glob(pattern, recursive=True))
            logging.info(f"Found {len(image_paths)} images in directory")
        
        if image_paths:
            random.shuffle(image_paths)
            if len(image_paths) > self.config.calib_max_images:
                image_paths = image_paths[:self.config.calib_max_images]
                logging.info(f"Limited calibration images to {self.config.calib_max_images}")
        
        return image_paths
    
    def create_network_from_onnx(self, onnx_path: str):
        """Create network from ONNX file"""
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)
        
        logging.info(f"Parsing ONNX model: {onnx_path}")
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                error_msgs = []
                for error_idx in range(parser.num_errors):
                    error_msgs.append(str(parser.get_error(error_idx)))
                raise RuntimeError(f"ONNX parsing failed:\n" + "\n".join(error_msgs))
        
        self._print_network_info(network)
        return builder, network
    
    def _print_network_info(self, network):
        """Print network information"""
        logging.info("=== Network Information ===")
        logging.info(f"Inputs: {network.num_inputs}")
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            logging.info(f"  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, type: {input_tensor.dtype}")
        
        logging.info(f"Outputs: {network.num_outputs}")
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            logging.info(f"  Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, type: {output_tensor.dtype}")
    
    def create_builder_config(self, 
                            builder, 
                            network,
                            calibrator: Optional[Int8Calibrator] = None):
        """Create builder configuration"""
        config = builder.create_builder_config()
        
        config.max_workspace_size = self.config.max_workspace_size
        logging.info(f"Workspace size: {self.config.max_workspace_size / (1024**3):.1f} GB")
        
        if self.config.use_fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logging.info("Enabled FP16 precision")
            else:
                logging.warning("Device does not support FP16, fallback to FP32")
                self.config.use_fp16 = False
        
        if self.config.use_int8:
            if not calibrator:
                raise ValueError("INT8 mode requires calibrator")
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            logging.info("Enabled INT8 precision")
        
        if self.config.enable_dla and builder.num_DLA_cores > 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.config.dla_core
            if self.config.gpu_fallback:
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            logging.info(f"Enabled DLA core {self.config.dla_core}")
        
        profile = builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = list(input_tensor.shape)
            
            if input_shape[0] == -1:
                input_shape[0] = self.config.max_batch_size
            
            shape_tuple = tuple(input_shape)
            profile.set_shape(input_tensor.name, shape_tuple, shape_tuple, shape_tuple)
            logging.info(f"Set input {input_tensor.name} shape: {shape_tuple}")
        
        config.add_optimization_profile(profile)
        return config
    
    def build_engine(self, 
                    onnx_path: str, 
                    engine_path: str,
                    calib_list: Optional[str] = None,
                    calib_dir: Optional[str] = None) -> bool:
        """Build TensorRT engine"""
        try:
            start_time = time.time()
            
            builder, network = self.create_network_from_onnx(onnx_path)
            
            calibrator = None
            if self.config.use_int8:
                calib_images = self.collect_calibration_images(calib_list, calib_dir)
                if not calib_images:
                    raise ValueError("INT8 mode requires calibration images")
                
                input_names = [network.get_input(i).name for i in range(network.num_inputs)]
                
                if not self.config.calib_cache_file:
                    cache_dir = os.path.dirname(engine_path)
                    model_name = Path(onnx_path).stem
                    self.config.calib_cache_file = os.path.join(cache_dir, f"{model_name}_calib.cache")
                
                calibrator = Int8Calibrator(
                    calib_images, 
                    input_names, 
                    self.config.input_shape,
                    self.config.calib_batch_size,
                    self.config.calib_cache_file
                )
            
            config = self.create_builder_config(builder, network, calibrator)
            
            logging.info("Starting TensorRT engine build...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if not serialized_engine:
                raise RuntimeError("Engine build failed")
            
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            build_time = time.time() - start_time
            
            self._validate_engine(serialized_engine, engine_path)
            
            logging.info(f"✅ Engine build successful!")
            logging.info(f"   Build time: {build_time:.1f} seconds")
            logging.info(f"   Engine file: {engine_path}")
            logging.info(f"   File size: {os.path.getsize(engine_path) / (1024**2):.1f} MB")
            
            return True
            
        except Exception as e:
            logging.error(f"Engine build failed: {e}")
            return False
    
    def _validate_engine(self, serialized_engine: bytes, engine_path: str):
        """Validate engine"""
        try:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            
            if not engine:
                raise RuntimeError("Engine deserialization failed")
            
            logging.info("=== Engine Validation ===")
            logging.info(f"Bindings: {engine.num_bindings}")
            
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                is_input = engine.binding_is_input(i)
                shape = engine.get_binding_shape(i)
                dtype = engine.get_binding_dtype(i)
                logging.info(f"  Binding {i}: {name} ({'input' if is_input else 'output'}), shape: {shape}, type: {dtype}")
            
        except Exception as e:
            logging.warning(f"Engine validation failed: {e}")


def setup_logging(verbose: bool = False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def generate_default_engine_path(onnx_path: str, config: EngineConfig) -> str:
    """Generate default engine path"""
    onnx_path_obj = Path(onnx_path)
    base_name = onnx_path_obj.stem
    
    precision = "int8" if config.use_int8 else "fp16" if config.use_fp16 else "fp32"
    engine_name = f"{base_name}_{precision}.engine"
    engine_dir = onnx_path_obj.parent / "model_engine"
    
    return str(engine_dir / engine_name)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Unified ONNX to TensorRT Engine conversion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (FP32)
  python onnx_to_engine.py --onnx model.onnx
  
  # FP16 precision
  python onnx_to_engine.py --onnx model.onnx --fp16
  
  # INT8 precision (requires calibration data)
  python onnx_to_engine.py --onnx model.onnx --int8 --calib-dir ./images
  
  # Custom configuration
  python onnx_to_engine.py --onnx model.onnx --fp16 --workspace 4 --output custom.engine
        """
    )
    
    parser.add_argument('--onnx', type=str, required=True, help='ONNX model file path')
    parser.add_argument('--output', type=str, default=None, help='Output engine file path (auto-generated if not specified)')
    
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    precision_group.add_argument('--int8', action='store_true', help='Use INT8 precision')
    
    parser.add_argument('--calib-list', type=str, default=None, help='Calibration image list file path')
    parser.add_argument('--calib-dir', type=str, default=None, help='Calibration image directory path')
    parser.add_argument('--calib-num', type=int, default=200, help='Maximum calibration images')
    parser.add_argument('--calib-cache', type=str, default=None, help='Calibration cache file path')
    
    parser.add_argument('--workspace', type=float, default=8.0, help='Workspace size (GB)')
    parser.add_argument('--batch-size', type=int, default=1, help='Maximum batch size')
    parser.add_argument('--input-shape', type=str, default="1,3,640,640", help='Input shape (format: N,C,H,W)')
    
    parser.add_argument('--enable-dla', action='store_true', help='Enable DLA (Jetson only)')
    parser.add_argument('--dla-core', type=int, default=0, help='DLA core number')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--config-file', type=str, default=None, help='Configuration file path (JSON format)')
    parser.add_argument('--save-config', type=str, default=None, help='Save configuration to file')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if not os.path.isfile(args.onnx):
        logging.error(f"ONNX file not found: {args.onnx}")
        return 1
    
    config = EngineConfig()
    
    if args.config_file and os.path.isfile(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                config_dict = json.load(f)
            config = EngineConfig.from_dict(config_dict)
            logging.info(f"Loaded config from: {args.config_file}")
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
            return 1
    
    config.use_fp16 = args.fp16
    config.use_int8 = args.int8
    config.max_workspace_size = int(args.workspace * 1024 * 1024 * 1024)
    config.max_batch_size = args.batch_size
    config.calib_max_images = args.calib_num
    config.calib_cache_file = args.calib_cache
    config.enable_dla = args.enable_dla
    config.dla_core = args.dla_core
    config.verbose = args.verbose
    
    try:
        shape_parts = [int(x.strip()) for x in args.input_shape.split(',')]
        if len(shape_parts) == 4:
            config.input_shape = tuple(shape_parts)
        else:
            raise ValueError("Input shape must have 4 dimensions")
    except Exception as e:
        logging.error(f"Invalid input shape format: {e}")
        return 1
    
    if not args.output:
        args.output = generate_default_engine_path(args.onnx, config)
        logging.info(f"Auto-generated output path: {args.output}")
    
    if args.save_config:
        try:
            os.makedirs(os.path.dirname(args.save_config), exist_ok=True)
            with open(args.save_config, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            logging.info(f"Configuration saved to: {args.save_config}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    if config.use_int8:
        if not args.calib_list and not args.calib_dir:
            logging.error("INT8 mode requires calibration data (--calib-list or --calib-dir)")
            return 1
    
    logging.info("=== Starting TensorRT Engine Build ===")
    logging.info(f"ONNX file: {args.onnx}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Precision: {'INT8' if config.use_int8 else 'FP16' if config.use_fp16 else 'FP32'}")
    logging.info(f"Input shape: {config.input_shape}")
    logging.info(f"Workspace: {config.max_workspace_size / (1024**3):.1f} GB")
    
    builder = EngineBuilder(config)
    success = builder.build_engine(
        args.onnx, 
        args.output,
        args.calib_list,
        args.calib_dir
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())