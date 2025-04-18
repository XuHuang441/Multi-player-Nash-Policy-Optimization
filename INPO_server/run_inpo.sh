source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

# Base paths and settings
initial_model="RLHFlow/LLaMA3-SFT" # "initial_model_path/LLaMA3-SFT"
base_dataset_path="dataset_path/datasets"
base_model_path="model_path/models"
ratio=$(echo "scale=10; 1/3" | bc)
# ratio=0
eta=0.005
iteration_prefix="inpo"


# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local previous_model=$2
    local input_path=$3
    local json_output=$4
    local pref_output=$5

    conda activate mypo
    my_world_size=8
    sanity_check=False
    use_tour=True
    K=8
    
    CUDA_VISIBLE_DEVICES=0 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=1 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=2 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=3 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=4 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 4 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=5 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 5 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=6 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 6 --my_world_size ${my_world_size} --eos_ids 128009 &
    CUDA_VISIBLE_DEVICES=7 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 7 --my_world_size ${my_world_size} --eos_ids 128009 &
    wait
    
    python ./generation/merge_data.py --base_path $json_output --output_dir "${json_output}.json" --num_datasets $my_world_size

    CUDA_VISIBLE_DEVICES=0 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_0.json" --output_dir "${pref_output}_0.json" --K $K &
    CUDA_VISIBLE_DEVICES=1 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_1.json" --output_dir "${pref_output}_1.json" --K $K &
    CUDA_VISIBLE_DEVICES=2 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_2.json" --output_dir "${pref_output}_2.json" --K $K &
    CUDA_VISIBLE_DEVICES=3 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_3.json" --output_dir "${pref_output}_3.json" --K $K &
    CUDA_VISIBLE_DEVICES=4 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_4.json" --output_dir "${pref_output}_4.json" --K $K &
    CUDA_VISIBLE_DEVICES=5 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_5.json" --output_dir "${pref_output}_5.json" --K $K &
    CUDA_VISIBLE_DEVICES=6 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_6.json" --output_dir "${pref_output}_6.json" --K $K &
    CUDA_VISIBLE_DEVICES=7 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_7.json" --output_dir "${pref_output}_7.json" --K $K &

    wait
    python ./annotate_data/merge.py --base_path $pref_output --output_dir "${pref_output}_data.json" --num_datasets $my_world_size

#   conda activate rlhf
    pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data_pref_prob"
    mkdir $pref_prob_path

# initializing with config file would cause an error
#    accelerate launch --config_file ./configs/zero2.yaml ./inpo/precompute.py --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}_data.json" \
#    --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model \
#    --loss_type inpo --lr_scheduler_type cosine

    accelerate launch ./inpo/precompute.py --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}_data.json" \
    --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model \
    --loss_type inpo --lr_scheduler_type cosine

    output_model_path="${base_model_path}/${iteration_prefix}_iter${iteration}"
    mkdir $output_model_path

    # initializing with config file would cause an error
#    accelerate launch --config_file ./configs/zero3.yaml ./inpo/inpo_train.py --run_name "${iteration_prefix}_${iteration}" \
#        --output_dir $output_model_path --model_name_or_path $previous_model --learning_rate 5e-7 --ratio $ratio --eta $eta \
#        --train_dir $pref_prob_path --loss_type inpo --lr_scheduler_type cosine

    accelerate ./inpo/inpo_train.py --run_name "${iteration_prefix}_${iteration}" \
    --output_dir $output_model_path --model_name_or_path $previous_model --learning_rate 5e-7 --ratio $ratio --eta $eta \
    --train_dir $pref_prob_path --loss_type inpo --lr_scheduler_type cosine --report_to wandb
}


# Main loop for iterations
for i in {1..3}
do
    iteration_name="iter${i}"
    input_path="RLHFlow/iterative-prompt-v1-iter${i}-20K"
    mkdir "${base_dataset_path}/${iteration_prefix}_${iteration_name}"
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