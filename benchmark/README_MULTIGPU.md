# Multi-GPU Relighting Inference

This directory contains scripts for running relighting inference across multiple GPUs in parallel.

## Files

- `relit_inference.py` - Original single-GPU sequential inference script
- `relit_inference_multigpu.py` - New multi-GPU parallel inference script
- `launch_multigpu_inference.py` - Easy-to-use launcher with command-line options
- `README_MULTIGPU.md` - This file

## How it Works

The multi-GPU version distributes image-reference pairs across available GPUs:

1. **Work Distribution**: Instead of processing images sequentially, all image-reference combinations are added to a work queue
2. **GPU Processes**: Each GPU runs a separate process that pulls work items from the queue
3. **Model Loading**: Each GPU loads its own copy of the model to avoid memory conflicts
4. **Parallel Processing**: Multiple image-reference pairs are processed simultaneously on different GPUs

## Usage

### Quick Start

```bash
# Use default settings with all available GPUs
python launch_multigpu_inference.py

# Specify custom paths
python launch_multigpu_inference.py \
    --input_path /path/to/input/images \
    --reference_path /path/to/reference/hdrs \
    --output_path /path/to/output

# Use specific GPUs only
python launch_multigpu_inference.py --gpus "0,1,2"

# Reduce iterations for faster testing
python launch_multigpu_inference.py --iterations 10

# Enable new decoder
python launch_multigpu_inference.py --new_decoder

# Dry run to check configuration
python launch_multigpu_inference.py --dry_run
```

### Advanced Usage

```bash
# Run directly with the multi-GPU script
python relit_inference_multigpu.py
```

### Command Line Options

- `--input_path`: Directory containing input images
- `--reference_path`: Directory containing reference HDR images  
- `--output_path`: Output directory for results
- `--iterations`: Number of iterations per image-reference pair (default: 50)
- `--ddim_steps`: Number of DDIM sampling steps (default: 50)
- `--new_decoder`: Enable new decoder for better identity preservation
- `--gpus`: Comma-separated list of GPU IDs to use (e.g., "0,1,2")
- `--dry_run`: Print configuration without running inference

## Performance Comparison

**Single GPU (Original)**:
- Processes one image-reference pair at a time
- For N images and M references: Total time = N × M × iterations × time_per_iteration

**Multi-GPU (New)**:
- Processes multiple pairs simultaneously
- For G GPUs: Total time ≈ (N × M × iterations × time_per_iteration) / G
- Near-linear speedup with number of GPUs

## Example Output Structure

```
output_multigpu/
├── image1.jpg_ref1.hdr/
│   ├── 12345_image1.jpg
│   ├── 67890_image1.jpg
│   └── ...
├── image1.jpg_ref2.hdr/
│   ├── 12345_image1.jpg
│   ├── 67890_image1.jpg
│   └── ...
└── ...
```

## Requirements

- PyTorch with CUDA support
- Multiple CUDA-capable GPUs
- Sufficient GPU memory for the model on each GPU
- All dependencies from the original script

## Monitoring

The script provides real-time updates on:
- GPU assignment and work distribution
- Progress for each GPU
- Completion status
- Error handling

## Notes

1. **Memory Usage**: Each GPU loads its own model copy, so ensure sufficient GPU memory
2. **File I/O**: All GPUs write to the same output directory; folder creation is thread-safe
3. **Error Handling**: If one GPU encounters an error, other GPUs continue processing
4. **Interruption**: Ctrl+C gracefully stops all GPU processes
5. **Work Balancing**: Work is distributed dynamically as GPUs become available

## Troubleshooting

**Out of Memory**: Reduce the number of GPUs used or ensure no other processes are using GPU memory

**File Conflicts**: The script handles concurrent file creation safely

**Slow Performance**: Check that all GPUs are being utilized with `nvidia-smi`

**Import Errors**: Ensure all dependencies are installed and the environment is properly configured
