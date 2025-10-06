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
export DATASET_ROOT="/lustre/fsw/portfolios/maxine/users/jingya/data/multi_illumination/multi_illumination_test_mip2_jpg"

# list all folders in DATASET_ROOT
for folder in "$DATASET_ROOT"/*/; do
    folder_name=$(basename "$folder")

    # Check if folder_name is in skip_folders
    skip_folders=(
        "everett_dining1" 
        "everett_dining2" 
        "everett_kitchen12" 
        "everett_kitchen14" 
        "everett_kitchen17" 
        "everett_kitchen18"
        "everett_kitchen2"
        "everett_kitchen4"
        "everett_kitchen5"
        "everett_kitchen6"
        "everett_kitchen7"
        "everett_kitchen8"
        "everett_kitchen9"
        "everett_living2"
        "everett_living4"
        "everett_lobby1"
        "everett_lobby2"
        "everett_lobby11"
        "everett_lobby12"
        "everett_lobby13"
        "everett_lobby14"
        "everett_lobby15"
        "everett_lobby16"
    )
    skip=false
    for skip_folder in "${skip_folders[@]}"; do
        if [[ "$folder_name" == "$skip_folder" ]]; then
            skip=true
            break
        fi
    done

    if $skip; then
        echo "Skipping folder: $folder_name"
        continue
    fi

    DATA_DIR=multi_illumination/multi_illumination_test_mip2_jpg/$folder_name
    python benchmark/launch_multigpu_inference.py \
        --gpus "0,1,2,3,4,5,6,7" \
        --input_path /lustre/fsw/portfolios/maxine/users/jingya/data/$DATA_DIR \
        --img_formats jpg \
        --img_pattern "*_mip2.jpg" \
        --reference_path /lustre/fsw/portfolios/maxine/users/jingya/data/hdris/benchmarking_HDRs \
        --ref_formats png hdr \
        --use_predefined_env_list \
        --rename_lights \
        --output_path ./output/benchmark/$DATA_DIR \
        --iterations 1
    wait
done