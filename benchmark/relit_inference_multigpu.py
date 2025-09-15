from share import *

import sys
import os
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
from typing import List, Tuple

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
        imn = os.path.basename(img_path)
        fn = os.path.basename(shd_path)
        fold_name = os.path.join(output_path, imn + '_' + fn)
        
        if not os.path.exists(fold_name):
            os.makedirs(fold_name, exist_ok=True)
            
        if len(os.listdir(fold_name)) > 20:
            print(f'GPU {gpu_id}: Skipping existing: {imn}_{fn}')
            return
        
        print(f'GPU {gpu_id}: Processing {imn} with {fn}')
        
        # Load and preprocess images once
        input_image_full = cv2.imread(img_path)[..., ::-1] / 255.0
        input_shd_full = cv2.imread(shd_path)[..., ::-1] / 255.0
        orig_h, orig_w = input_image_full.shape[:2]
        
        input_image = cv2.resize(input_image_full, (512, 512))
        input_shd = cv2.resize(input_shd_full, (512, 512))
        
        for i in range(many_iter):
            seeds = torch.seed()
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

            image_name = os.path.basename(img_path)
            Image.fromarray(x_samples_resized).save(os.path.join(fold_name, f'{seeds}_{image_name}'))
        
        print(f'GPU {gpu_id}: Completed {imn} with {fn}')
        
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
                
            img_path, shd_path = work_item
            
            process_image_reference_pair(
                gpu_id=gpu_id,
                img_path=img_path,
                shd_path=shd_path,
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
            'PATH_OF_OUTPUT': './output'
        }
    else:
        config = custom_config
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    
    print(f"Found {num_gpus} GPUs")
    
    # Get image and reference lists
    imagesets = os.listdir(config['PATH_OF_INPUT_IMAGE'])
    refsets = os.listdir(config['PATH_OF_REFERENCE'])
    
    formats = ['jpg', 'jpeg', 'png', 'bmp']
    imagesets = [imn for imn in imagesets if imn.split('.')[-1].lower() in formats]
    refsets = [fn for fn in refsets if fn.split('.')[-1].lower() in formats]
    
    print(f"Found {len(imagesets)} images and {len(refsets)} reference images")
    
    # Create all image-reference pairs
    work_items = []
    for imn in imagesets:
        img_path = os.path.join(config['PATH_OF_INPUT_IMAGE'], imn)
        for fn in refsets:
            shd_path = os.path.join(config['PATH_OF_REFERENCE'], fn)
            work_items.append((img_path, shd_path))
    
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
