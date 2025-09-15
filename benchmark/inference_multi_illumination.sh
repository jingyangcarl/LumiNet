# bash benchmark/inference.sh

# change the huggingface default cache folder to luster to avoid disk quota
# put this to ~/.bashrc if necessary
export HF_HOME=/lustre/fsw/portfolios/maxine/users/jingya/.cache/huggingface
export PIP_CACHE_DIR=/lustre/fsw/portfolios/maxine/users/jingya/.cache/pip

eval "$(conda shell.bash hook)"
conda activate luminet

# single gpu
# python relit_inference.py

# multi-gpu (change the --gpus and --ntasks-per-node according to your need)

DATA_DIR=multi_illumination/multi_illumination_test_mip2_jpg/everett_dining1
python benchmark/launch_multigpu_inference.py \
    --gpus "0,1,2,3,4,5,6,7" \
    --input_path /lustre/fsw/portfolios/maxine/users/jingya/data/$DATA_DIR \
    --input_formats jpg \
    --input_pattern "*-mip2.jpg" \
    --reference_path /lustre/fsw/portfolios/maxine/users/jingya/data/hdris/benchmarking_HDRs \
    --reference_formats png \
    --output_path ./output/benchmark/$DATA_DIR \
    --iterations 1
wait

# export OPENCV_IO_ENABLE_OPENEXR=1 # enable openexr support in opencv
# DATA_DIR=interiorverse/samples/L3D124S8ENDIDRGLFYUI5NFSLUF3P3WA888
# python benchmark/launch_multigpu_inference.py \
#     --gpus "0,1,2,3,4,5,6,7" \
#     --input_path /lustre/fsw/portfolios/maxine/users/jingya/data/$DATA_DIR \
#     --img_formats exr \
#     --img_pattern "*_im.exr" \
#     --reference_path /lustre/fsw/portfolios/maxine/users/jingya/data/hdris/benchmarking_HDRs \
#     --ref_formats png hdr \
#     --output_path ./output/benchmark/$DATA_DIR \
#     --iterations 1
# wait