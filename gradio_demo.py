import gradio as gr
import torch
import cv2
import numpy as np
import einops
from PIL import Image
import random
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from huggingface_hub import hf_hub_download



def load_model(checkpoint_path):
    model = create_model('./models/cldm_v21_LumiNet.yaml').cpu()
    model.add_new_layers()
    model.concat = False
    model.load_state_dict(load_state_dict(checkpoint_path, location='cuda'))
    model.parameterization = "v"
    return model.cuda()

# Download the checkpoint and load the model.
resume_path = hf_hub_download(repo_id="xyxingx/LumiNet", filename="LumiNet.ckpt")
model = load_model(resume_path)
ddim_sampler = DDIMSampler(model)

def process_images(input_image, reference_image, ddim_steps=50):
    seed_list = [random.randint(0, 100000) for _ in range(3)]  # Generate with 3 random seeds
    output_images = []
    
    for seed in seed_list:
        torch.manual_seed(seed)
        
        input_image_np = np.array(input_image) / 255
        reference_image_np = np.array(reference_image) / 255

        input_image_resized = cv2.resize(input_image_np, (512, 512))
        reference_image_resized = cv2.resize(reference_image_np, (512, 512))
        control_feat = np.concatenate((input_image_resized, reference_image_resized), axis=2)

        control = torch.from_numpy(control_feat.copy()).float().cuda()
        control = einops.rearrange(control, 'h w c -> 1 c h w').clone()

        c_cat = control.cuda()
        c = model.get_unconditional_conditioning(1)
        uc_cross = model.get_unconditional_conditioning(1)
        uc_cat = c_cat
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}
        shape = (4, 64, 64)  # Adjusted latent space shape

        samples, _ = ddim_sampler.sample(ddim_steps, 1, shape, cond, verbose=False, eta=0.0,
                                         unconditional_guidance_scale=9.0,
                                         unconditional_conditioning=uc_full)

        x_samples = model.decode_first_stage(samples)
        x_samples = (x_samples.squeeze(0) + 1.0) / 2.0
        x_samples = x_samples.clamp(0,1)
        x_samples = (x_samples.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_images.append(Image.fromarray(x_samples))
    
    return output_images


with gr.Blocks() as gram:
    gr.Markdown("# LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting")
    gr.Markdown("A demo for [paper](https://luminet-relight.github.io/)")
    gr.Markdown("Upload your own image and reference, our demo will output 3 relit images, with different seeds.")
    gr.Markdown("Note: No post-processing is used in this demo.")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Input Image", sources=["upload"], width=256, height=256)
        ref_img = gr.Image(type="pil", label="Reference Image", sources=["upload"], width=256, height=256)
    
    ddim_slider = gr.Slider(minimum=10, maximum=1000, step=1, label="DDIM Steps", value=50)
    btn = gr.Button("Generate")
    
    with gr.Row():
        output_imgs = [gr.Image(label=f"Generated Image {i+1}", width=256, height=256) for i in range(3)]
    
    btn.click(process_images, inputs=[input_img, ref_img, ddim_slider], outputs=output_imgs)

if __name__ == "__main__":
    gram.launch()

