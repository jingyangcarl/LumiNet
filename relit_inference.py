from share import *

from cldm.model import create_model, load_state_dict
import cv2
import numpy as np
import torch
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import os 
import random
from huggingface_hub import hf_hub_download

resume_path = hf_hub_download(repo_id="xyxingx/LumiNet", filename="LumiNet.ckpt")
N = 1
ddim_steps = 100


model = create_model('./models/cldm_v21_LumiNet.yaml').cpu()
model.add_new_layers()
model.concat = False
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
many_iter = 50
model.parameterization = "v"
ddim_sampler = DDIMSampler(model)

imagesets = PATH_OF_INPUT_IMAGE #add your own data_path here

refsets = PATH_OF_REFERENCE  #add your own data_path here
output_path= PATH_OF_OUTPUT
for imn in imagesets:
    if imn == ('35829.png' or '55238.png'):
        continue
    img_path = PATH_OF_INPUT_IMAGE + imn
    for fn in refsets:
        shd_path = PATH_OF_INPUT_REFERENCE +fn
        fold_name = output_path +imn +'_'+fn+'/'
        if not os.path.exists(fold_name):
            os.makedirs(fold_name)
        if len(os.listdir(fold_name))>20:
            print('skipping exisiting:', imn+fn)
            continue
        for i in range(many_iter):
            seeds = torch.seed()
            torch.manual_seed(seeds)
            torch.cuda.manual_seed(seeds)
            random.seed(seeds)
            input_image = cv2.imread(img_path)[...,::-1]/255
            input_shd = cv2.imread(shd_path)[...,::-1]/255
            input_image = cv2.resize(input_image,(512,512))
            input_shd = cv2.resize(input_shd,(512,512))
            control_feat = np.concatenate((input_image,input_shd),axis=2)  

            control = torch.from_numpy(control_feat.copy()).float().cuda()
            control = torch.stack([control for _ in range(N)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            c_cat = control.cuda()
            c = model.get_unconditional_conditioning(N)
            # print('attn_shape',c.shape)
            uc_cross = model.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            cond={"c_concat": [c_cat], "c_crossattn": [c]}
            b, c, h, w = cond["c_concat"][0].shape
            shape = (4, h // 8, w // 8)

            samples, intermediates = ddim_sampler.sample(ddim_steps, N, 
                                                        shape, cond, verbose=False, eta=0.0, 
                                                        unconditional_guidance_scale=9.0,
                                                        unconditional_conditioning=uc_full
                                                        )

            x_samples = model.decode_first_stage(samples)
            x_samples = x_samples.squeeze(0)
            x_samples = (x_samples + 1.0) / 2.0
            x_samples = x_samples.transpose(0, 1).transpose(1, 2)
            x_samples = x_samples.cpu().numpy()
            x_samples = x_samples.clip(0,1)
            x_samples = (x_samples * 255).astype(np.uint8)

            image_name = img_path.split('/')[-1]
            Image.fromarray(x_samples).save(fold_name + str(seeds) +'_'+image_name)









