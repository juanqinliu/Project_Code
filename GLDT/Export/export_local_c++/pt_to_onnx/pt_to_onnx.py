#!/usr/bin/env python3
"""
Convert a PyTorch model to an ONNX model with dynamic batch dimension.
"""

import torch
import torch.onnx
import onnx
import onnxsim
import argparse
import os
import sys
from typing import Dict, List

def get_model_info(model) -> Dict[str, List[str]]:
    """Infer input and output names for the given model.

    Returns a dict with 'inputs' and 'outputs' keys containing name lists.
    """
    input_names: List[str]
    output_names: List[str]

    if hasattr(model, 'names'):
        # YOLO-style models commonly use these names
        input_names = ['images']
        output_names = ['output']
    else:
        import inspect
        sig = inspect.signature(model.forward)
        input_names = list(sig.parameters.keys())

        # Fallback: assume a single output
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            output_names = ['output']
        else:
            output_names = ['output']

    return {'inputs': input_names, 'outputs': output_names}

def convert_pt_to_onnx(pt_path,
                       onnx_path,
                       imgsz: int = 640,
                       simplify: bool = True,
                       keep_intermediate: bool = False,
                       verbose: bool = False,
                       opset: int = 11) -> None:
    """Convert a PyTorch model to ONNX with a dynamic batch dimension.

    Args:
        pt_path: Path to the input PyTorch model file (.pt).
        onnx_path: Path to the output ONNX model file (.onnx).
        imgsz: Square input image size.
        simplify: Whether to simplify the exported ONNX model.
        keep_intermediate: Whether to keep specific intermediate tensors as outputs.
        verbose: Whether to print verbose ONNX export logs.
        opset: ONNX opset version to use for export.
    """

    print("\n=== Start: Convert PyTorch to ONNX ===")
    print(f"Input: {pt_path}")
    print(f"Output: {onnx_path}")
    print(f"Image size: {imgsz}x{imgsz}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n=== Device ===")
    print(f"Using: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    try:
        checkpoint = torch.load(pt_path, map_location=device)
        print("\n=== Load Model ===")
        print(f"Loaded checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")

        model = None

        if isinstance(checkpoint, dict):
            possible_keys = ['model', 'ema', 'state_dict', 'net', 'network', 'weights']
            for key in possible_keys:
                if key in checkpoint:
                    print(f"Found model key: {key}")
                    model = checkpoint[key]
                    break

            if model is None:
                for key, value in checkpoint.items():
                    if hasattr(value, 'eval') and hasattr(value, 'state_dict'):
                        print(f"Using '{key}' as model")
                        model = value
                        break

            if model is None or isinstance(model, dict):
                print("Warning: Could not extract a model object from checkpoint.")
                print("This may be a state_dict; attempting to load via ultralytics for YOLO models.")
                try:
                    from ultralytics import YOLO
                    print("Trying ultralytics.YOLO loader...")
                    yolo_model = YOLO(pt_path)
                    model = yolo_model.model
                    print("ultralytics loaded the model successfully.")
                except ImportError:
                    print("ultralytics not installed; cannot auto-handle YOLOv8 models.")
                    sys.exit(1)
                except Exception as e:
                    print(f"ultralytics load failed: {e}")
                    sys.exit(1)
        else:
            model = checkpoint

        if model is None:
            print("Error: No valid model could be extracted from the file.")
            sys.exit(1)

        if not hasattr(model, 'eval'):
            print("Error: Extracted object is not a valid PyTorch model.")
            print(f"Object type: {type(model)}")
            sys.exit(1)

        print(f"Model extracted: {type(model)}")

        model.eval()
        model.float()
        model = model.to(device)
        print("Model set to eval (float32) and moved to device.")

        model_info = get_model_info(model)
        print("\n=== Model IO ===")
        print(f"Inputs: {model_info['inputs']}")
        print(f"Outputs: {model_info['outputs']}")

    except Exception as e:
        print(f"Model load failed: {e}")
        print("Please check if the model file is valid and not corrupted.")
        sys.exit(1)

    print("\n=== Prepare Export ===")
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    print(f"Dummy input shape: {dummy_input.shape}")

    dynamic_axes = {
        'images': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    output_names = ['output']

    if keep_intermediate:
        intermediate_outputs = [
            'onnx::Shape_1140',
            'onnx::Reshape_1167',
            'onnx::Reshape_1194',
            'onnx::Reshape_1221'
        ]
        output_names.extend(intermediate_outputs)
        for name in intermediate_outputs:
            dynamic_axes[name] = {0: 'batch_size'}

    print("\nDynamic axes:")
    for name, axes in dynamic_axes.items():
        print(f"  {name}: {axes}")

    try:
        print("\nExporting ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )
        print("ONNX export succeeded.")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        sys.exit(1)

    if simplify:
        print("\n=== Simplify ONNX ===")
        try:
            onnx_model = onnx.load(onnx_path)
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, onnx_path)
                print("ONNX simplification completed.")
            else:
                print("ONNX simplification failed, using the original model.")
        except Exception as e:
            print(f"ONNX simplification error: {e}")
            print("Proceeding with the original ONNX model.")

    print("\n=== Done ===")
    print(f"Input: {pt_path}")
    print(f"Output: {onnx_path}")
    print(f"Size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    print("\nTips:")
    print("1) Inspect with Netron to confirm IO shapes.")
    print("2) Use verify_onnx.py to test dynamic batch support.")
    print("3) Ensure correct optimization profiles when converting to TensorRT.")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX with dynamic batch')
    parser.add_argument('--input', '-i', type=str, default='local.pt', help='Input PyTorch model path')
    parser.add_argument('--output', '-o', type=str, default='local.onnx', help='Output ONNX model path')
    parser.add_argument('--imgsz', type=int, default=640, help='Square input image size')
    parser.add_argument('--no-simplify', action='store_true', help='Disable ONNX simplification')
    parser.add_argument('--keep-intermediate', action='store_true', default=False, help='Keep intermediate outputs')
    parser.add_argument('--verbose', action='store_true', help='Verbose export logs')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    convert_pt_to_onnx(
        pt_path=args.input,
        onnx_path=args.output,
        imgsz=args.imgsz,
        simplify=not args.no_simplify,
        keep_intermediate=args.keep_intermediate,
        verbose=args.verbose,
        opset=args.opset
    )

if __name__ == '__main__':
    main() 