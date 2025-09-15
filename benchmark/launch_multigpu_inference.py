#!/usr/bin/env python3
"""
Multi-GPU Relighting Inference Launcher
Usage: python launch_multigpu_inference.py [options]
"""

import argparse
import os
import torch
from relit_inference_multigpu import main as run_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU Relighting Inference')
    
    parser.add_argument('--input_path', type=str, 
                       default='/lustre/fsw/portfolios/maxine/users/jingya/data/multi_illumination/multi_illumination_test_mip2_jpg/everett_dining1',
                       help='Path to input images directory')
    
    parser.add_argument('--reference_path', type=str,
                       default='/lustre/fsw/portfolios/maxine/users/jingya/data/hdris/benchmarking_HDRs',
                       help='Path to reference HDR images directory')
    
    parser.add_argument('--output_path', type=str,
                       default='./output_multigpu',
                       help='Output directory')
    
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations per image-reference pair')
    
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM sampling steps')
    
    parser.add_argument('--new_decoder', action='store_true',
                       help='Use new decoder for better identity preservation')
    
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). If not specified, uses all available GPUs')
    
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configuration and exit without running inference')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if specific GPUs are requested
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        print(f"Using GPUs: {gpu_ids}")
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Using all available GPUs: {list(range(num_gpus))}")
    
    # Validate paths
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path does not exist: {args.input_path}")
    
    if not os.path.exists(args.reference_path):
        raise ValueError(f"Reference path does not exist: {args.reference_path}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Count work items
    formats = ['jpg', 'jpeg', 'png', 'bmp']
    imagesets = [f for f in os.listdir(args.input_path) 
                if f.split('.')[-1].lower() in formats]
    refsets = [f for f in os.listdir(args.reference_path) 
              if f.split('.')[-1].lower() in formats]
    
    total_pairs = len(imagesets) * len(refsets)
    total_iterations = total_pairs * args.iterations
    
    print("\n" + "="*60)
    print("MULTI-GPU RELIGHTING INFERENCE CONFIGURATION")
    print("="*60)
    print(f"Input path:          {args.input_path}")
    print(f"Reference path:      {args.reference_path}")
    print(f"Output path:         {args.output_path}")
    print(f"Images found:        {len(imagesets)}")
    print(f"References found:    {len(refsets)}")
    print(f"Total pairs:         {total_pairs}")
    print(f"Iterations per pair: {args.iterations}")
    print(f"Total iterations:    {total_iterations}")
    print(f"DDIM steps:          {args.ddim_steps}")
    print(f"New decoder:         {args.new_decoder}")
    print(f"Available GPUs:      {torch.cuda.device_count()}")
    print("="*60)
    
    if args.dry_run:
        print("Dry run completed. Exiting without running inference.")
        return
    
    # Create configuration
    config = {
        'new_decoder': args.new_decoder,
        'N': 1,
        'ddim_steps': args.ddim_steps,
        'many_iter': args.iterations,
        'PATH_OF_INPUT_IMAGE': args.input_path,
        'PATH_OF_REFERENCE': args.reference_path,
        'PATH_OF_OUTPUT': args.output_path
    }
    
    print("\nStarting multi-GPU inference...")
    print("Press Ctrl+C to interrupt\n")
    
    try:
        run_inference(config)
        print("\nInference completed successfully!")
    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
