#!/usr/bin/env python3
"""
PyTorch to ONNX Converter with Batch Inference Optimization
"""
import torch
import onnx
import onnxsim
import argparse
import os
import sys


def load_model(pt_path, device):
    """
    Load PyTorch model from checkpoint file.
    
    Supports:
    - Direct model objects
    - Checkpoints with 'model', 'ema', or 'state_dict' keys
    - YOLO models (via ultralytics)
    """
    checkpoint = torch.load(pt_path, map_location=device)
    
    # Extract model from checkpoint dictionary
    if isinstance(checkpoint, dict):
        for key in ['model', 'ema', 'state_dict']:
            if key in checkpoint:
                model = checkpoint[key]
                if hasattr(model, 'eval'):
                    return model
        
        # Try ultralytics YOLO loader as fallback
        try:
            from ultralytics import YOLO
            return YOLO(pt_path).model
        except Exception as e:
            print(f"Error: Failed to load model using ultralytics: {e}")
            sys.exit(1)
    
    # Direct model object
    if hasattr(checkpoint, 'eval'):
        return checkpoint
    
    print("Error: Could not extract valid model from checkpoint")
    sys.exit(1)


def prune_intermediate_outputs(onnx_path):
    """
    Remove intermediate layer outputs from ONNX model.
    
    YOLO models often export multi-scale feature layers (P3/P4/P5) as separate
    outputs. While useful for training, these intermediate outputs:
    - Waste GPU memory
    - Break TensorRT batch kernel fusion
    - Cause near-serial batch execution instead of parallel
    
    This function keeps only the final detection output for optimal performance.
    """
    model = onnx.load(onnx_path)
    original_outputs = list(model.graph.output)
    
    # Keep only 'output' node
    final_outputs = [o for o in original_outputs if o.name == 'output']
    
    if not final_outputs:
        print("Warning: No 'output' node found, keeping all outputs")
        return
    
    if len(final_outputs) == len(original_outputs):
        print("✓ Model already has single output")
        return
    
    # Update model outputs
    del model.graph.output[:]
    model.graph.output.extend(final_outputs)
    
    removed_count = len(original_outputs) - len(final_outputs)
    
    # Save pruned model
    onnx.save(model, onnx_path)


def convert(pt_path, onnx_path, imgsz=640, simplify=True, opset=11):
    """
    Convert PyTorch model to ONNX with batch optimization.
    
    Args:
        pt_path: Input PyTorch model path (.pt)
        onnx_path: Output ONNX model path (.onnx)
        imgsz: Input image size (must be multiple of 32)
        simplify: Whether to simplify ONNX graph
        opset: ONNX opset version
    """
    print("\n" + "="*70)
    print("  PyTorch → ONNX Converter (Batch Inference Optimized)")
    print("="*70)
    print(f"\nInput:  {pt_path}")
    print(f"Output: {onnx_path}")
    print(f"Size:   {imgsz}x{imgsz}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    
    # Load and prepare model
    print("\n[1/3] Loading model...")
    model = load_model(pt_path, device)
    model.eval()
    model.float()
    model.to(device)
    print("✓ Model loaded")
    
    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    
    dynamic_axes = {
        'images': {0: 'batch_size'},  # Dynamic batch dimension
        'output': {0: 'batch_size'}
    }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print("✓ ONNX export successful")
    except Exception as e:
        print(f"Error: ONNX export failed - {e}")
        sys.exit(1)
    
    # Prune intermediate outputs (critical for batch performance)
    # print("\n[3/4] Pruning intermediate outputs...")
    prune_intermediate_outputs(onnx_path)
    
    # Simplify ONNX graph
    if simplify:
        print("\n[3/3] Simplifying ONNX graph...")
        try:
            onnx_model = onnx.load(onnx_path)
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, onnx_path)
                print("✓ ONNX simplified")
            else:
                print("Warning: Simplification failed, using original")
        except Exception as e:
            print(f"Warning: Simplification error - {e}")
    
    # Verify final model
    print("\n" + "="*70)
    print("  Verification")
    print("="*70)
    
    model = onnx.load(onnx_path)
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    print(f"\nOutputs:     {len(model.graph.output)} (should be 1)")
    print(f"File size:   {file_size_mb:.2f} MB")
    print(f"Opset:       {opset}")
    print(f"Batch mode:  Dynamic (1-N)")
    
    print("\n" + "="*70)
    print("  Conversion Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Verify with Netron: https://netron.app")
    print("  2. Build TensorRT engine with optimal_batch matching your use case")
    print("  3. Expected performance: ~2x speedup for batch=3 inference\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to ONNX (optimized for TensorRT batching)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input PyTorch model (.pt)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output ONNX model (.onnx)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable ONNX simplification')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    convert(
        pt_path=args.input,
        onnx_path=args.output,
        imgsz=args.imgsz,
        simplify=not args.no_simplify,
        opset=args.opset
    )


if __name__ == '__main__':
    main()
