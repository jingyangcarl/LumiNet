# LumiNet
[CVPR 2025] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting

### News
[Apr. 2025] Inference code and model are released!

[Mar. 2025] We released the Huggingface [Demo](https://huggingface.co/spaces/xyxingx/LumiNet)!

[Feb. 2025] LumiNet is accepted by CVPR 2025!


### Installation
Follow these steps to install LumiNet and get started:
#### Requirements

open-clip-torch==2.0.1 (very important)

#### Install Dependencies
You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Demo
Run demo.py will allow you to quickly try the relighting performance on your own images. Each run will provide relit results with 3 different seeds.

### Local Relit
To relit the image locally, use the following command:
```bash
python inference.py
```
You need to set the PATH to the input image and reference image. You can also name the number of relit examples you want to generate and the DDIM step. 

### Seed Selection (Optional)
We provide the nearest-neighbor search based on the latent extrinsic. Once you get the result images in the folder, you can run this command to help you find the image that has the closest latent extrinsic to the reference image

```bash
python seed_selection.py
```

Note: The seed selection relies on the representation of the latent extrinsic. If you cannot get a result that you expected, you may also need to check the relit images on your own.

### Flow-based Clean Up (Optional)
Coming soon



