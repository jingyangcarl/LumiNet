# bash benchmark/inference.sh

# change the huggingface default cache folder to luster to avoid disk quota
# put this to ~/.bashrc if necessary
export HF_HOME=/lustre/fsw/portfolios/maxine/users/jingya/.cache/huggingface
export PIP_CACHE_DIR=/lustre/fsw/portfolios/maxine/users/jingya/.cache/pip


# single gpu
# python relit_inference.py

# multi-gpu (change the --gpus and --ntasks-per-node according to your need)

export OPENCV_IO_ENABLE_OPENEXR=1 # enable openexr support in opencv

export DATASET_ROOT="/lustre/fsw/portfolios/maxine/users/jingya/data/getty/test_samples"


# list all folders in DATASET_ROOT
for folder in "$DATASET_ROOT"/*/; do
    folder_name=$(basename "$folder")

    # Check if folder_name is in skip_folders
    # skip_folders=("getty-v1-images-only")
    # skip=true
    # for skip_folder in "${skip_folders[@]}"; do
    #     if [[ "$folder_name" == "$skip_folder" ]]; then
    #         skip=false
    #         break
    #     fi
    # done

    # if $skip; then
    #     echo "Skipping folder: $folder_name"
    #     continue
    # fi

    DATA_DIR=getty/test_samples/$folder_name
    python benchmark/launch_multigpu_inference.py \
        --gpus "0,1,2,3,4,5,6,7" \
        --input_path /lustre/fsw/portfolios/maxine/users/jingya/data/$DATA_DIR \
        --img_formats png \
        --img_pattern "*.png" \
        --first_n 20 \
        --reference_path /lustre/fsw/portfolios/maxine/users/jingya/data/hdris/benchmarking_HDRs \
        --ref_formats png hdr \
        --use_predefined_env_list \
        --rename_lights \
        --output_path ./output/benchmark/$DATA_DIR \
        --iterations 1
    wait
done