#!/usr/bin/env python3
"""
Multi-GPU Relighting Inference Launcher
Usage: python launch_multigpu_inference.py [options]
"""

import argparse
import os
import torch
import fnmatch
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
    
    parser.add_argument('--img_formats', type=str, nargs='+',
                       default=['jpg', 'jpeg', 'png', 'bmp', 'exr', 'hdr'],
                       help='Supported input image formats (space-separated list)')
    
    parser.add_argument('--ref_formats', type=str, nargs='+',
                       default=['png', 'exr', 'hdr'],
                       help='Supported reference image formats (space-separated list)')
    
    parser.add_argument('--img_pattern', type=str, default=None,
                       help='Pattern to match input image names (e.g., "*-albedo.png", "image_*.jpg"). Uses glob-style wildcards.')
    
    parser.add_argument('--ref_pattern', type=str, default=None,
                       help='Pattern to match reference image names (e.g., "*-hdr.exr", "env_*.hdr"). Uses glob-style wildcards.')
    
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
    
    # Count work items using provided formats
    img_formats = [fmt.lower() for fmt in args.img_formats]
    ref_formats = [fmt.lower() for fmt in args.ref_formats]
    
    # Filter by format first
    imagesets = [f for f in os.listdir(args.input_path) 
                if f.split('.')[-1].lower() in img_formats]
    refsets = [f for f in os.listdir(args.reference_path) 
              if f.split('.')[-1].lower() in ref_formats]
    
    # Apply pattern matching if specified
    if args.img_pattern:
        imagesets = [f for f in imagesets if fnmatch.fnmatch(f, args.img_pattern)]
        print(f"Applied input pattern '{args.img_pattern}': {len(imagesets)} images match")
    
    if args.ref_pattern:
        refsets = [f for f in refsets if fnmatch.fnmatch(f, args.ref_pattern)]
        print(f"Applied reference pattern '{args.ref_pattern}': {len(refsets)} references match")
    
    total_pairs = len(imagesets) * len(refsets)
    total_iterations = total_pairs * args.iterations
    
    print("\n" + "="*60)
    print("MULTI-GPU RELIGHTING INFERENCE CONFIGURATION")
    print("="*60)
    print(f"Input path:          {args.input_path}")
    print(f"Reference path:      {args.reference_path}")
    print(f"Output path:         {args.output_path}")
    print(f"Input formats:       {', '.join(args.img_formats)}")
    print(f"Reference formats:   {', '.join(args.ref_formats)}")
    if args.img_pattern:
        print(f"Input pattern:       {args.img_pattern}")
    if args.ref_pattern:
        print(f"Reference pattern:   {args.ref_pattern}")
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
        'PATH_OF_OUTPUT': args.output_path,
        'img_formats': img_formats,
        'ref_formats': ref_formats,
        'img_pattern': args.img_pattern,
        'ref_pattern': args.ref_pattern
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
