set -e

# Base paths and settings
initial_model="RLHFlow/LLaMA3-SFT"
base_dataset_path="dataset_path/datasets"
base_model_path="model_path/models"
ratio=$(python -c "print(f'{1/3:.10f}')")
# ratio=0
eta=0.005
iteration_prefix="tdpo"
export WANDB_PROJECT="INPO"
num_rounds=4

history_paths=("Timia123/inpo_iter1_jun19" "Timia123/inpo_iter2_jun19" "Timia123/tdpo_iter4_jun21")

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local previous_model="Timia123/tdpo_iter4_jun21"
    local input_path=$3
    local json_output=$4
    local pref_output=$5

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate /home/hubing/.conda/envs/vllm

    my_world_size=8 # todo
    sanity_check=False # todo
    use_tour=True
    K=8 # todo
    
    echo "Starting generation for iteration ${iteration}..."

    CUDA_VISIBLE_DEVICES=0 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=1 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=2 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=3 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=4 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 4 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=5 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 5 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=6 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 6 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=7 conda run -n vllm python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 7 --my_world_size ${my_world_size} --eos_ids 128009 &
    wait

    conda run -n vllm python ./generation/merge_data.py --base_path $json_output --output_dir "${json_output}.json" --num_datasets $my_world_size

    echo "Starting preference modeling..."

    CUDA_VISIBLE_DEVICES=0 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_0.json" --output_dir "${pref_output}_0.json" --K $K &
    CUDA_VISIBLE_DEVICES=1 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_1.json" --output_dir "${pref_output}_1.json" --K $K &
    CUDA_VISIBLE_DEVICES=2 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_2.json" --output_dir "${pref_output}_2.json" --K $K &
    CUDA_VISIBLE_DEVICES=3 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_3.json" --output_dir "${pref_output}_3.json" --K $K &
    CUDA_VISIBLE_DEVICES=4 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_4.json" --output_dir "${pref_output}_4.json" --K $K &
    CUDA_VISIBLE_DEVICES=5 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_5.json" --output_dir "${pref_output}_5.json" --K $K &
    CUDA_VISIBLE_DEVICES=6 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_6.json" --output_dir "${pref_output}_6.json" --K $K &
    CUDA_VISIBLE_DEVICES=7 /home/hubing/.conda/envs/vllm/bin/accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_7.json" --output_dir "${pref_output}_7.json" --K $K &

    wait
    conda run -n vllm python ./annotate_data/merge.py --base_path $pref_output --output_dir "${pref_output}_data.json" --num_datasets $my_world_size
    
    conda activate /home/hubing/.conda/envs/rlhf

    pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data_pref_prob"
    mkdir -p $pref_prob_path

    history_args=""
    if [ ${#history_paths[@]} -gt 0 ]; then
        history_args="--history_paths ${history_paths[@]}"
    fi

    echo "Starting precomputing for iteration ${iteration}..."

    /home/hubing/.conda/envs/rlhf/bin/accelerate launch --config_file ./configs/zero2.yaml ./inpo/precompute.py --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}_data.json" \
    --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model \
    --loss_type inpo --lr_scheduler_type cosine \
    $history_args

    output_model_path="${base_model_path}/${iteration_prefix}_iter${iteration}"
    mkdir -p $output_model_path

    echo "Starting TDPO training for iteration ${iteration}..."
    
    /home/hubing/.conda/envs/rlhf/bin/accelerate launch --config_file ./configs/zero3.yaml ./inpo/inpo_train.py --run_name "${iteration_prefix}_${iteration}" \
        --output_dir $output_model_path --model_name_or_path $previous_model --learning_rate 5e-7 --ratio $ratio --eta $eta \
        --train_dir $pref_prob_path --loss_type inpo --lr_scheduler_type cosine --report_to wandb

    history_paths+=("$output_model_path")

    echo "Completed iteration ${iteration}"
}


# Main loop for iterations
for ((i=4; i<=$num_rounds; i++))
do
    iteration_name="iter${i}"
    input_path="RLHFlow/iterative-prompt-v1-iter${i}-20K"
    mkdir -p "${base_dataset_path}/${iteration_prefix}_${iteration_name}"
    json_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data"
    pref_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/pref"

    # json_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}"
    # reward_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/${iteration_prefix}_${iteration_name}_reward.json"
    
    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        previous_model=$initial_model
    else
        previous_iteration=$((i-1))
        previous_model="${base_model_path}/${iteration_prefix}_iter${previous_iteration}"
    fi

    run_iteration $i $previous_model $input_path $json_output $pref_output

done

echo "All iterations completed successfully!"