# LumiNet
[CVPR 2025] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting

## News
[Apr. 2025] Inference code and model are released!

[Mar. 2025] We released the Huggingface [Demo](https://huggingface.co/spaces/xyxingx/LumiNet)!

[Feb. 2025] LumiNet is accepted by CVPR 2025!


## Installation
Follow these steps to install LumiNet and get started:
### Requirements
GPU with VRAM > 11G

open-clip-torch==2.0.1 (very important)

## Install Dependencies
You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Pretrained Models Download

The weights of LumiNet and Latent-Intrinsics are hosted on [Hugging Face](https://huggingface.co/xyxingx/LumiNet). Confirmation of accepting the conditions is required. Once you accept the conditions, you can download the models, or the models will be automatically downloaded and loaded when running the demo or inference.

You may need to log in to your HF account before running the code.


## Demo
Run gardio_demo.py will allow you to quickly try the relighting performance on your own images. Each run will provide relit results with 3 different seeds.

## Local Relit
To relit the image locally, use the following command:
```bash
python relit_inference.py
```
You need to set the PATH to the input image and reference image. You can also name the number of relit examples you want to generate and the DDIM step. 

## Seed Selection (Optional)
We provide the nearest-neighbor search based on the latent extrinsic. Once you get the result images in the folder, you can run this command to help you find the image that has the closest latent extrinsic to the reference image

```bash
python seed_selection.py
```

Note: The seed selection relies on the representation of the latent extrinsic. If you cannot get a result that you expected, you may also need to check the relit images on your own.

## Flow-based Clean Up (Optional)
We provide a rectified-flow-based cleanup process to obtain higher-resolution and less noisy images. Our implementation is based on the diffusers library (version ≥ 0.31.0). Please note that access to the [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-schnell) model may require permission. A minimum of 14 GB of VRAM is recommended for successful execution.


```bash
python flux_cleanup.py
```
Note: You may need to adjust the hyperparameters (mainly timesteps) to achieve better visual results, as the optimal values often vary depending on the input images.


## Contact and Citation
For any questions related to the code or model, please contact [Xiaoyan Xing](mailto:x.xing@uva.nl). 
If you find our work useful for your research, please consider citing:

**LumiNet**
```bash
@inproceedings{Xing2024luminet,
      title={LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting},
      author={Xing, Xiaoyan and Groh, Konrad and Karagolu, Sezer and Gevers, Theo and Bhattad, Anand},
      booktitle={CVPR},
      year={2025}}
```
**Latent-intrinsics**
```bash
@inproceedings{Zhang2024Latent,
    title={Latent Intrinsics Emerge from Training to Relight},
    author={Zhang, Xiao and Gao, William and Jain, Seemandhar and Maire, Michael and Forsyth, David and Bhattad, Anand},
    booktitle={NeurIPS},
    year={2024}
  }
```
