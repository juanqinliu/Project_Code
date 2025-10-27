#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import warnings

import torch

# Add project root to Python path to allow for ultralytics imports
def setup_ultralytics_import():
    """Setup ultralytics import path and return required modules"""
    
    # Method 1: Try direct import (pip installed)
    try:
        from ultralytics.utils.torch_utils import select_device
        from ultralytics.models.yolo.model import DetectionModel
        print("INFO: Successfully imported ultralytics from system installation.")
        return select_device, DetectionModel
    except ImportError:
        pass
    
    # Method 2: Try adding project root to path
    try:
        ROOT = Path(__file__).resolve().parents[3]
        ULTRALYTICS_PATH = ROOT / "ultralytics"
        
        if ULTRALYTICS_PATH.exists() and str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
            print(f"INFO: Added project root to Python path: {ROOT}")
        
        from ultralytics.utils.torch_utils import select_device
        from ultralytics.models.yolo.model import DetectionModel
        print(f"INFO: Successfully imported ultralytics from local path: {ULTRALYTICS_PATH}")
        return select_device, DetectionModel
        
    except ImportError as e:
        print(f"ERROR: Failed to import 'ultralytics'. Ensure it's installed properly.")
        print(f"ERROR: {e}")
        print("\nSuggested solutions:")
        print("  1. Install via pip: pip3 install ultralytics")
        print("  2. Install in editable mode: cd ultralytics && pip3 install -e .")
        sys.exit(1)

# Import ultralytics modules
select_device, DetectionModel = setup_ultralytics_import()

# --- Custom Model Definition for Compatibility ---
# This allows loading .pt files that were saved using this custom class structure.
class MotionDetectionModel(DetectionModel):
    """Custom MotionDetectionModel to enable loading of models saved with this class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_motion = True

    def _predict_biformer_fusion(self, x, prev_x=None):
        """Placeholder for dual-frame forward pass. The actual logic resides in the main forward method."""
        return self.forward(x, prev_x)


class ModelWrapper(torch.nn.Module):
    """
    A wrapper for PyTorch models to intelligently handle single or dual frame inputs for ONNX export.
    It determines the number of inputs based on the model's architecture.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_dual_input = self._check_dual_input_support()
        print(f"INFO: Model identified as {'dual-input' if self.is_dual_input else 'single-input'}.")

    def _check_dual_input_support(self):
        """Checks if the model appears to support dual-frame input."""
        # Heuristics to detect motion-aware models
        if hasattr(self.model, '_predict_biformer_fusion'):
            return True
        if hasattr(self.model, 'has_motion') and self.model.has_motion:
            return True
        return False

    def forward(self, *args):
        """Forward pass that adapts to the number of inputs."""
        if self.is_dual_input:
            # For dual-input models, expect two tensors: current_frame and previous_frame
            current_frame, previous_frame = args
            # Models might expect a tuple or separate arguments
            try:
                return self.model((current_frame, previous_frame))
            except Exception:
                return self.model(current_frame, prev_x=previous_frame)
        else:
            # For single-input models, expect one tensor: images
            return self.model(args[0])


def load_model(model_path, device):
    """Loads a PyTorch model from the given path."""
    print(f"INFO: Loading PyTorch model from: {model_path}")
    try:
        ckpt = torch.load(model_path, map_location=device)
        model = ckpt.get('model', ckpt)
        if isinstance(model, dict):
             print("ERROR: Loaded checkpoint does not contain a valid model object.")
             return None
        model.eval()
        print("INFO: Model loaded successfully.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return None


def export_and_verify_onnx(wrapped_model, dummy_input, output_path, opset, dynamic, simplify):
    """Exports the model to ONNX, simplifies it, and verifies the output."""
    if wrapped_model.is_dual_input:
        input_names = ["current_frame", "previous_frame"]
        output_names = ["output0"]
        dynamic_axes = {"current_frame": {0: "batch"}, "previous_frame": {0: "batch"}, "output0": {0: "batch"}}
    else:
        input_names = ["images"]
        output_names = ["output0"]
        dynamic_axes = {"images": {0: "batch"}, "output0": {0: "batch"}}

    if not dynamic:
        dynamic_axes = None # Static export

    print(f"INFO: Starting ONNX export...\n  - Path: {output_path}\n  - Opset: {opset}\n  - Dynamic: {dynamic}")
    
    # Suppress tracer warnings during export
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        try:
            torch.onnx.export(
                wrapped_model,
                dummy_input,
                output_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
            )
            print("INFO: ONNX export completed.")
        except Exception as e:
            print(f"ERROR: ONNX export failed: {e}")
            return False

    # Simplify the model
    if simplify:
        try:
            import onnx
            import onnxsim
            print("INFO: Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simplified, output_path)
                print("INFO: ONNX model simplified successfully.")
            else:
                print("WARNING: Failed to simplify ONNX model. Using the original export.")
        except Exception as e:
            print(f"WARNING: ONNX simplification failed: {e}")

    # Verify the model
    try:
        import onnx
        print("INFO: Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("INFO: ONNX model verification successful.")
        # Print model info
        print("\n--- ONNX Model Details ---")
        for i, node in enumerate(onnx_model.graph.input):
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in node.type.tensor_type.shape.dim]
            print(f"  Input  {i}: {node.name}, Shape: {shape}")
        for i, node in enumerate(onnx_model.graph.output):
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in node.type.tensor_type.shape.dim]
            print(f"  Output {i}: {node.name}, Shape: {shape}")
        print("--------------------------\n")
        return True
    except Exception as e:
        print(f"ERROR: ONNX model verification failed: {e}")
        return False


def export_model_to_onnx(
    model_path,
    output_path=None,
    imgsz=640,
    batch_size=1,
    opset=12,
    simplify=True,
    dynamic=True,
    half=False,
    device='cpu',
):
    """Main function to convert a YOLO model to ONNX format."""
    if output_path is None:
        output_path = Path(model_path).with_suffix('.onnx')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    if device.type == 'cpu' and half:
        print("WARNING: Half precision (FP16) is not supported on CPU. Forcing FP32.")
        half = False

    model = load_model(model_path, device)
    if model is None:
        return None

    model = model.to(device)
    model = model.half() if half else model.float()

    wrapped_model = ModelWrapper(model)
    
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    dummy_tensor = torch.zeros(batch_size, 3, *imgsz).to(device)
    dummy_tensor = dummy_tensor.half() if half else dummy_tensor.float()

    dummy_input = (dummy_tensor, dummy_tensor) if wrapped_model.is_dual_input else (dummy_tensor,)

    if export_and_verify_onnx(wrapped_model, dummy_input, str(output_path), opset, dynamic, simplify):
        print(f"\nSUCCESS: Model exported to {output_path}")
        return str(output_path)
    else:
        print(f"\nFAILURE: Could not export model to {output_path}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO PyTorch model to ONNX format.")
    parser.add_argument('--model', type=str, required=True, help='Path to the input PyTorch .pt model.')
    parser.add_argument('--output', type=str, help='Path for the output ONNX model. Defaults to same name as input.')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (e.g., 640 for 640x640).')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for the exported model.')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version.')
    parser.add_argument('--simplify', action='store_true', default=True, help='Simplify the ONNX model.')
    parser.add_argument('--no-simplify', action='store_false', dest='simplify', help='Do not simplify the ONNX model.')
    parser.add_argument('--dynamic', action='store_true', default=True, help='Export with dynamic axes (recommended).')
    parser.add_argument('--static', action='store_false', dest='dynamic', help='Export with static axes (for specific backends like TensorRT).')
    parser.add_argument('--half', action='store_true', help='Use FP16 (half precision). Only effective on GPU.')
    parser.add_argument('--device', type=str, default='0', help='Device to use for conversion (e.g., \'cpu\', \'cuda\', \'0\').')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    export_model_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device,
    )
