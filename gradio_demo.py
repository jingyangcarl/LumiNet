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

# -------------------------
# Global settings & helpers
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_N = 1
INF_SIZE = 512  # inference resolution (square)

# Lazy flag for loading the new/bypass decoder weights once
_NEW_DECODER_LOADED = False
_NEW_DECODER_PATH = None

def _ensure_new_decoder_loaded(model):
    """Load weights for the new/bypass decoder only once."""
    global _NEW_DECODER_LOADED, _NEW_DECODER_PATH
    if not _NEW_DECODER_LOADED:
        _NEW_DECODER_PATH = hf_hub_download(repo_id="xyxingx/LumiNet", filename="new_decoder.ckpt")
        model.change_first_stage(_NEW_DECODER_PATH)
        if hasattr(model, "first_stage_model"):
            model.first_stage_model = model.first_stage_model.to(DEVICE)
        _NEW_DECODER_LOADED = True


# -------------------------
# Model loading
# -------------------------
def load_model(checkpoint_path):
    model = create_model("./models/cldm_v21_LumiNet.yaml").cpu()
    model.add_new_layers()         # ensures new decoder layers exist
    model.concat = False
    sd = load_state_dict(checkpoint_path, location=DEVICE)
    model.load_state_dict(sd)
    model.parameterization = "v"
    model = model.to(DEVICE).eval()
    return model


# Download main checkpoint & build sampler
resume_path = hf_hub_download(repo_id="xyxingx/LumiNet", filename="LumiNet.ckpt")
model = load_model(resume_path)
ddim_sampler = DDIMSampler(model)


# -------------------------
# Inference
# -------------------------
def _preprocess_to_np_rgb(img_pil):
    """PIL -> float32 numpy [H,W,3] in [0,1], RGB."""
    return (np.array(img_pil.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0)

def _resize_to_square_512(img_np):
    return cv2.resize(img_np, (INF_SIZE, INF_SIZE), interpolation=cv2.INTER_LANCZOS4)

def _tensor_from_np(img_np):
    """HWC [0..1] -> BCHW float32 on DEVICE."""
    t = torch.from_numpy(img_np.copy()).float()  # HWC
    t = einops.rearrange(t, "h w c -> 1 c h w")  # BCHW
    return t.to(DEVICE)

def process_images(input_image, reference_image, ddim_steps=50, use_new_decoder=False):
    """
    input_image, reference_image: PIL Images
    Returns 3 PIL images with original aspect ratio, generated with different seeds.
    """
    assert input_image is not None and reference_image is not None, "Please upload both input and reference images."

    # Prepare originals (for aspect-ratio restoration)
    input_np_full  = _preprocess_to_np_rgb(input_image)       # [H,W,3] 0..1
    ref_np_full    = _preprocess_to_np_rgb(reference_image)   # [H,W,3] 0..1
    orig_h, orig_w = input_np_full.shape[:2]

    # Inference inputs @ 512×512
    input_np_512 = _resize_to_square_512(input_np_full)
    ref_np_512   = _resize_to_square_512(ref_np_full)

    # Control feature: concat input & reference along channels -> [H,W,6]
    control_feat = np.concatenate((input_np_512, ref_np_512), axis=2).astype(np.float32)
    control = _tensor_from_np(control_feat)  # [1,6,512,512]

    # Also keep the input tensor for new-decoder decoding path (needs input AE features)
    input_tensor = _tensor_from_np(input_np_512)  # [1,3,512,512]

    # Conditioning
    with torch.no_grad():
        c_cat = control
        # Cross-attention uses unconditional embeddings because there is no text prompt
        c = model.get_unconditional_conditioning(BATCH_N)
        uc_cross = model.get_unconditional_conditioning(BATCH_N)
        uc_cat = c_cat

        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        cond    = {"c_concat": [c_cat], "c_crossattn": [c]}

        # Latent shape for 512×512 with factor 8
        shape = (4, INF_SIZE // 8, INF_SIZE // 8)

        # Make 3 different seeds
        seeds = [random.randint(0, 999_999) for _ in range(3)]
        outputs = []

        # Ensure new/bypass decoder weights are loaded if requested
        if use_new_decoder:
            _ensure_new_decoder_loaded(model)

        for seed in seeds:
            torch.manual_seed(seed)

            samples, _ = ddim_sampler.sample(
                S=ddim_steps,
                batch_size=BATCH_N,
                shape=shape,
                conditioning=cond,
                verbose=False,
                eta=0.0,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc_full
            )

            # Decode
            if use_new_decoder:
                # encode_first_stage expects [-1,1] range
                ae_hs = model.encode_first_stage(input_tensor * 2.0 - 1.0)[1]
                x = model.decode_new_first_stage(samples, ae_hs)
            else:
                x = model.decode_first_stage(samples)

            # To image in [0,255], HWC
            x = (x.squeeze(0) + 1.0) / 2.0
            x = x.clamp(0, 1)
            x = (einops.rearrange(x, "c h w -> h w c").detach().cpu().numpy() * 255.0).astype(np.uint8)

            # Resize back to original aspect ratio/size
            x = cv2.resize(x, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            outputs.append(Image.fromarray(x))

    return outputs


# -------------------------
# UI
# -------------------------
with gr.Blocks() as gram:
    gr.Markdown("# LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting")
    gr.Markdown("[Sept. 2025] Update: Incorporating a bypass decoder, which improves identity preservation.")
    gr.Markdown("A demo for [paper](https://luminet-relight.github.io/)")
    gr.Markdown("Upload your own image and a reference. The demo outputs 3 relit images with different random seeds.")
    gr.Markdown("**Note:** No post-processing is used. You may need to try with multiple seeds to get a more preferred relighting.")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Input Image", sources=["upload"])
        ref_img   = gr.Image(type="pil", label="Reference Image", sources=["upload"])

    with gr.Row():
        ddim_slider = gr.Slider(minimum=10, maximum=1000, step=1, label="DDIM Steps", value=50)
        use_new_dec = gr.Checkbox(label="Use bypass decoder (new) for better identity preservation. Click to disable.", value=True)

    btn = gr.Button("Generate")

    with gr.Row():
        # No fixed width/height so images keep their native aspect ratio in the layout
        out1 = gr.Image(type="pil", label="Generated Image 1")
        out2 = gr.Image(type="pil", label="Generated Image 2")
        out3 = gr.Image(type="pil", label="Generated Image 3")

    btn.click(
        fn=process_images,
        inputs=[input_img, ref_img, ddim_slider, use_new_dec],
        outputs=[out1, out2, out3]
    )

if __name__ == "__main__":
    gram.launch()
