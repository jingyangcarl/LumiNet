from share import *

import sys
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# Add parent directory to path to find cldm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cldm.model import create_model, load_state_dict
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import os 
import random
from huggingface_hub import hf_hub_download
import itertools
import time
import fnmatch
from typing import List, Tuple
from glob import glob

def load_image_with_hdr_support(image_path: str) -> np.ndarray:
    """
    Load image with HDR support. Handles both LDR and HDR formats.
    For HDR images (EXR, HDR), applies tone mapping to convert to LDR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format, normalized to [0, 1]
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext in ['.exr', '.hdr']:
        # Load HDR image
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load HDR image: {image_path}")
        
        # Convert BGR to RGB
        image = image[..., ::-1]
        
        # compute the percentile value for normalization
        p95 = np.percentile(image, 95)
        
        # Apply tone mapping to convert HDR to LDR
        # Using Reinhard tone mapping
        # tone_mapped = image / (1.0 + image)
        ldr = np.clip(image / p95, 0, 1)
        tone_mapped = ldr ** (1/2.2)  # gamma correction
        
        # Alternatively, you could use OpenCV's built-in tone mappers:
        # tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=0.0, light_adapt=1.0, color_adapt=0.0)
        # tone_mapped = tonemap.process(image)
        
        # Ensure values are in [0, 1] range
        # tone_mapped = np.clip(tone_mapped, 0, 1)
        
        return tone_mapped
    else:
        # Load LDR image (JPG, PNG, etc.)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB and normalize to [0, 1]
        image = image[..., ::-1] / 255.0
        
        return image

def setup_model(gpu_id: int, new_decoder: bool = False):
    """Setup model on specific GPU"""
    torch.cuda.set_device(gpu_id)
    
    resume_path = hf_hub_download(repo_id="xyxingx/LumiNet", filename="LumiNet.ckpt")
    model = create_model('./models/cldm_v21_LumiNet.yaml').cpu()
    model.add_new_layers()
    model.concat = False
    model.load_state_dict(load_state_dict(resume_path, location='cuda'))

    if new_decoder:
        ae_checkpoint = hf_hub_download(repo_id="xyxingx/LumiNet", filename="new_decoder.ckpt")
        model.change_first_stage(ae_checkpoint)
    
    model = model.cuda(gpu_id)
    model.parameterization = "v"
    ddim_sampler = DDIMSampler(model)
    
    return model, ddim_sampler

def process_image_reference_pair(
    gpu_id: int,
    img_path: str,
    shd_path: str,
    output_name: str,
    output_path: str,
    many_iter: int,
    new_decoder: bool,
    N: int = 1,
    ddim_steps: int = 50
):
    """Process a single image-reference pair on specified GPU"""
    try:
        # Setup model on this GPU
        model, ddim_sampler = setup_model(gpu_id, new_decoder)
        
        # Extract filenames for folder naming
        imn = os.path.basename(img_path).replace('.', '_')
        # Use output_name for folder naming (this handles renaming)
        fn = output_name.split('.')[0]  # remove file extension for folder name
        folder_name_static = os.path.join(output_path, 'FR', 'static', fn)
        os.makedirs(folder_name_static, exist_ok=True)

        print(f'GPU {gpu_id}: Processing {imn} with {output_name} (source: {os.path.basename(shd_path)})')
        
        # Load and preprocess images once
        input_image_full = load_image_with_hdr_support(img_path)
        input_shd_full = load_image_with_hdr_support(shd_path)
        orig_h, orig_w = input_image_full.shape[:2]
        
        # for multi-illumination, we need to project to allign
        if 'chrome_envmap.exr' in shd_path:
            input_shd_full = np.flip(input_shd_full, axis=1)
            input_shd_full = np.roll(input_shd_full, input_shd_full.shape[1] // 2, axis=1)

        input_image = cv2.resize(input_image_full, (512, 512))
        input_shd = cv2.resize(input_shd_full, (512, 512))
        
        for i in range(many_iter):
            # Set random seed for reproducibility
            # seeds = torch.seed()
            seeds = '20250915'
            torch.manual_seed(seeds)
            torch.cuda.manual_seed(seeds)
            random.seed(seeds)
            
            control_feat = np.concatenate((input_image, input_shd), axis=2)
            control = torch.from_numpy(control_feat.copy()).float().cuda(gpu_id)
            control = torch.stack([control for _ in range(N)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            input_img = torch.from_numpy(input_image.copy()).float().cuda(gpu_id)
            input_img = input_img.unsqueeze(0)
            input_img = einops.rearrange(input_img, 'b h w c -> b c h w').clone()

            c_cat = control
            c = model.get_unconditional_conditioning(N)
            uc_cross = model.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            b, cch, h, w = cond["c_concat"][0].shape
            shape = (4, h // 8, w // 8)

            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                N,
                shape,
                cond,
                verbose=False,
                eta=0.0,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc_full
            )

            if new_decoder:
                ae_hs = model.encode_first_stage(input_img * 2 - 1)[1]
                x_samples = model.decode_new_first_stage(samples, ae_hs)
            else:
                x_samples = model.decode_first_stage(samples)

            x_samples = x_samples.squeeze(0)
            x_samples = (x_samples + 1.0) / 2.0
            x_samples = x_samples.transpose(0, 1).transpose(1, 2)
            x_samples = x_samples.cpu().numpy()
            x_samples = x_samples.clip(0, 1)
            x_samples = (x_samples * 255).astype(np.uint8)

            # Resize back to original size before saving
            x_samples_resized = cv2.resize(x_samples, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            image_name = os.path.basename(img_path).split('.')[0]
            os.makedirs(os.path.join(folder_name_static, 'separate'), exist_ok=True)
            Image.fromarray(x_samples_resized).save(os.path.join(folder_name_static, 'separate', f'{seeds}_{image_name}.png'))

            # add a concatenated image for reference, where the shd image should be resized to the size height as input image with aspect ratio maintained
            # input_shd_full_resized = cv2.resize(input_shd_full, (input_shd_full.shape[1] * orig_h // input_shd_full.shape[0], orig_h), interpolation=cv2.INTER_LANCZOS4)
            # concat_image = np.concatenate((input_image_full, input_shd_full_resized, x_samples_resized / 255.), axis=1)
            # os.makedirs(os.path.join(folder_name_static, 'mosaic'), exist_ok=True)
            # Image.fromarray((concat_image * 255).astype(np.uint8)).save(os.path.join(folder_name_static, 'mosaic', f'{seeds}_{image_name}_concat.png'))
            
            # stack in the following order:
            # row 1: |input image | shd image|
            # row 2: |black image | output image|
            black_image = np.zeros_like(input_image_full)
            input_shd_full_resized = cv2.resize(input_shd_full, (input_image_full.shape[1], input_image_full.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            row1 = np.concatenate((input_image_full, input_shd_full_resized), axis=1)
            row2 = np.concatenate((black_image, x_samples_resized / 255.), axis=1)
            concat_image = np.concatenate((row1, row2), axis=0)
            os.makedirs(os.path.join(folder_name_static, 'mosaic'), exist_ok=True)
            Image.fromarray((concat_image * 255).astype(np.uint8)).save(os.path.join(folder_name_static, 'mosaic', f'{seeds}_{image_name}_concat.png'))

        print(f'GPU {gpu_id}: Completed {imn} with {output_name}')
        
    except Exception as e:
        print(f'GPU {gpu_id}: Error processing {img_path} with {shd_path}: {str(e)}')
        raise e

def worker_process(gpu_id: int, work_queue: mp.Queue, config: dict):
    """Worker process for a single GPU"""
    print(f'Worker process started on GPU {gpu_id}')
    
    while True:
        try:
            # Get work item from queue
            work_item = work_queue.get(timeout=10)  # 10 second timeout
            
            if work_item is None:  # Sentinel value to stop
                break
                
            img_path, shd_path, output_name = work_item
            
            process_image_reference_pair(
                gpu_id=gpu_id,
                img_path=img_path,
                shd_path=shd_path,
                output_name=output_name,
                output_path=config['output_path'],
                many_iter=config['many_iter'],
                new_decoder=config['new_decoder'],
                N=config['N'],
                ddim_steps=config['ddim_steps']
            )
            
        except Exception as e:
            if "Empty" not in str(e):  # Ignore queue empty exceptions
                print(f'GPU {gpu_id}: Worker error: {str(e)}')
            break
    
    print(f'Worker process on GPU {gpu_id} finished')

def main(custom_config=None):
    """Main function to orchestrate multi-GPU inference"""
    
    # Configuration
    if custom_config is None:
        config = {
            'new_decoder': False,
            'N': 1,
            'ddim_steps': 50,
            'many_iter': 50,
            'PATH_OF_INPUT_IMAGE': '/lustre/fsw/portfolios/maxine/users/jingya/data/multi_illumination/multi_illumination_test_mip2_jpg/everett_dining1',
            'PATH_OF_REFERENCE': '/lustre/fsw/portfolios/maxine/users/jingya/data/hdris/benchmarking_HDRs',
            'PATH_OF_OUTPUT': './output',
            'img_formats': ['jpg', 'jpeg', 'png', 'bmp', 'exr', 'hdr'],
            'ref_formats': ['png', 'exr', 'hdr'],
            'img_pattern': None,
            'ref_pattern': None,
            'original_refsets': [],
            'light_rename_mapping': [],
            'rename_lights': False
        }
    else:
        config = custom_config
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        # raise RuntimeError("No CUDA GPUs available")
        pass
    
    print(f"Found {num_gpus} GPUs")
    
    # Get image and reference lists
    imagesets = os.listdir(config['PATH_OF_INPUT_IMAGE'])
    
    # Handle reference files based on configuration
    if 'original_refsets' in config and config['original_refsets']:
        # Use original refsets if provided (from predefined list or custom selection)
        refsets = config['original_refsets']
        print(f"Using provided reference list: {len(refsets)} files")
    else:
        # Fallback to directory scanning
        refsets = os.listdir(config['PATH_OF_REFERENCE'])
        print(f"Scanning reference directory: {len(refsets)} files found")
    
    # Use formats from config, with fallback to defaults
    img_formats = config.get('img_formats', ['jpg', 'jpeg', 'png', 'bmp', 'exr', 'hdr'])
    ref_formats = config.get('ref_formats', ['png', 'exr', 'hdr'])
    
    # Filter by format first
    imagesets = [imn for imn in imagesets if imn.split('.')[-1].lower() in img_formats]
    refsets = [fn for fn in refsets if fn.split('.')[-1].lower() in ref_formats]
    
    # Apply pattern matching if specified
    img_pattern = config.get('img_pattern')
    ref_pattern = config.get('ref_pattern')
    
    if img_pattern:
        imagesets = [imn for imn in imagesets if fnmatch.fnmatch(imn, img_pattern)]
        print(f"Applied input pattern '{img_pattern}': {len(imagesets)} images match")
    
    if ref_pattern:
        refsets = [fn for fn in refsets if fnmatch.fnmatch(fn, ref_pattern)]
        print(f"Applied reference pattern '{ref_pattern}': {len(refsets)} references match")

    print(f"Found {len(imagesets)} images and {len(refsets)} reference images")
    
    # Get light renaming mapping if available
    light_mapping = config.get('light_rename_mapping', {})
    rename_lights = config.get('rename_lights', False)
    
    if rename_lights and light_mapping:
        print(f"Light renaming enabled: {len(light_mapping)} files will use renamed output folders")
    
    # Create all image-reference pairs
    multi_illumination_reconstruction = True
    work_items = []
    for imn in imagesets:
        img_path = os.path.join(config['PATH_OF_INPUT_IMAGE'], imn)
        for fn in refsets:
            shd_path = os.path.join(config['PATH_OF_REFERENCE'], fn)
            # Get the output name (renamed if renaming is enabled, otherwise original)
            output_name = light_mapping.get(fn, fn) if rename_lights else fn
            
            # only process '0008' and '0009'
            # if output_name.split('.')[0].split('-')[-1] not in ['0008', '0009']:
            #     print(f"Skipping pair: Image '{imn}' with Reference '{fn}' as output '{output_name}'")
            #     continue
            # ask user input to confirm append to the list
            # user_input = input(f"Add pair: Image '{imn}' with Reference '{fn}' as output '{output_name}'? (y/n): ")
            # if user_input.lower() == 'y':
            #     pass
            # else:
            #     print(f"Skipping pair: Image '{imn}' with Reference '{fn}'")
            #     continue
            
            work_items.append((img_path, shd_path, output_name))
        
    
        if multi_illumination_reconstruction and 'multi_illumination' in config['PATH_OF_INPUT_IMAGE']:
            
            dataset_scene_root = '/lustre/fsw/portfolios/maxine/users/jingya/data/multi_illumination/multi_illumination_test_mip2_jpg'
            scenes = sorted(os.listdir(dataset_scene_root))
            scene_id = scenes.index(os.path.basename(os.path.normpath(config['PATH_OF_INPUT_IMAGE'])))
            lighting_id = int(imn.split('_')[1].split('.')[0])
            envlight_path = f'/lustre/fsw/portfolios/nvr/users/kahe/data/UniRelight_paper_data/MIT_test_data_diff12_fixLighting/{scene_id:06d}.{lighting_id:04d}/{lighting_id:04d}.chrome_envmap.exr'
            work_items.append((img_path, envlight_path, 'multi_illumination_reconstruction'))
    
    print(f"Total work items: {len(work_items)}")
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create work queue and add all work items
    work_queue = mp.Queue()
    for item in work_items:
        work_queue.put(item)
    
    # Add sentinel values to stop workers
    for _ in range(num_gpus):
        work_queue.put(None)
    
    # Update config for workers
    config['output_path'] = config['PATH_OF_OUTPUT']
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, work_queue, config))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All work completed!")

if __name__ == "__main__":
    main()
