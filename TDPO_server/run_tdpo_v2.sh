#!/bin/bash
source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

# Base paths and settings
initial_model="RLHFlow/LLaMA3-SFT"
base_dataset_path="dataset_path/datasets"
base_model_path="model_path/models"
ratio=$(echo "scale=10; 1/3" | bc)
eta=0.005
beta=0.01
iteration_prefix="tdpo"
max_history_t=2
num_rounds=3  # Total number of rounds to run - CUSTOMIZE THIS

# Create an array to store all history model paths
history_paths=()

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local previous_model=$2
    local input_path=$3
    local json_output=$4
    local pref_output=$5

    my_world_size=2
    sanity_check=True
    use_tour=True
    K=8

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate /home/zbz5349/anaconda3/envs/mypo

    echo "Starting generation for iteration ${iteration}..."

    # Run parallel generation
    CUDA_VISIBLE_DEVICES=2 conda run -n mypo python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=3 conda run -n mypo python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} --eos_ids 128009 &
    wait

    echo "Merging generation data..."
    conda run -n mypo python ./generation/merge_data.py --base_path $json_output --output_dir "${json_output}.json" --num_datasets $my_world_size

    echo "Starting preference modeling..."
    CUDA_VISIBLE_DEVICES=2 conda run -n mypo python annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_2.json" --output_dir "${pref_output}_2.json" --K $K &
    CUDA_VISIBLE_DEVICES=3 conda run -n mypo python annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_3.json" --output_dir "${pref_output}_3.json" --K $K &
    wait

    echo "Merging preference data..."
    conda run -n mypo python ./annotate_data/merge.py --base_path $pref_output --output_dir "${pref_output}_data.json" --num_datasets $my_world_size

    # Create directories for precomputed data and output model
    pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data_pref_prob"
    mkdir -p $pref_prob_path

    output_model_path="${base_model_path}/${iteration_prefix}_iter${iteration}"
    mkdir -p $output_model_path

    # Format history paths for command line argument
    history_args=""
    if [ ${#history_paths[@]} -gt 0 ]; then
        history_args="--history_paths ${history_paths[@]}"
    fi

    conda run -n mypo accelerate launch --config_file ./configs/zero2.yaml ./tdpo/precompute.py \
    --base_model_path $previous_model \
    --reference_model_path $initial_model \
    --output_dir $pref_prob_path \
    --data_path "${pref_output}_data.json" \
    --max_history_t $max_history_t \
    $history_args \
    --beta $beta

    echo "Starting TDPO training for iteration ${iteration}..."
    # Run TDPO for a single iteration
    CUDA_VISIBLE_DEVICES=2,3 conda run -n mypo accelerate launch --config_file ./configs/zero3.yaml ./tdpo/tdpo_train.py \
    --base_model_path $previous_model \
    --precomputed_dir $pref_prob_path \
    --output_dir $output_model_path \
    --learning_rate 5e-7 \
    --ratio $ratio \
    --eta $eta \
    --beta $beta \
    --lr_scheduler_type cosine \
    --report_to wandb 

    # merge deepspeed checkpoints
    echo "Merging deepspeed checkpoints..."
    conda run -n mypo python $output_model_path/zero_to_fp32.py $output_model_path $output_model_path 

    # Add this model to history paths
    history_paths+=("$output_model_path")

    echo "Completed iteration ${iteration}"
}

# Main loop for iterations
for ((i=1; i<=$num_rounds; i++))
do
    iteration_name="iter${i}"
    input_path="RLHFlow/iterative-prompt-v1-iter${i}-20K"
    mkdir -p "${base_dataset_path}/${iteration_prefix}_${iteration_name}"
    json_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data"
    pref_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/pref"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        previous_model=$initial_model
    else
        previous_iteration=$((i-1))
        previous_model="${base_model_path}/${iteration_prefix}_iter${previous_iteration}"
    fi

    echo "Starting iteration ${i} using model: ${previous_model}"
    run_iteration $i $previous_model $input_path $json_output $pref_output
done

echo "All iterations completed successfully!"
