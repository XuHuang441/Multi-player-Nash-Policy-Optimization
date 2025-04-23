source ~/.bashrc

set -e # Exit immediately if a command exits with a non-zero status

# Initialize Conda environment
eval "$(conda shell.bash hook)"
source /home/zbz5349/anaconda3/bin/activate /home/zbz5349/anaconda3/envs/mypo

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

    my_world_size=1
    sanity_check=True
    use_tour=True
    K=8
    
    CUDA_VISIBLE_DEVICES=1 python ./generation/get_hf2.py --model_name_or_path ${previous_model} --dataset_name_or_path ${input_path} --output_dir ${json_output} --sanity_check $sanity_check --K $K --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
    wait
    
    python ./generation/merge_data.py --base_path $json_output --output_dir "${json_output}.json" --num_datasets $my_world_size

    CUDA_VISIBLE_DEVICES=1 accelerate launch annotate_data/get_pref_single.py --use_tournament $use_tour --dataset_name_or_path "${json_output}_0.json" --output_dir "${pref_output}_0.json" --K $K &
    wait
    python ./annotate_data/merge.py --base_path $pref_output --output_dir "${pref_output}_data.json" --num_datasets $my_world_size

    pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data_pref_prob"
    mkdir -p $pref_prob_path

# initializing with config file would cause an error
#    accelerate launch --config_file ./configs/zero2.yaml ./inpo/precompute.py --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}_data.json" \
#    --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model \
#    --loss_type inpo --lr_scheduler_type cosine

    accelerate launch ./inpo/precompute.py --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}_data.json" \
    --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model \
    --loss_type inpo --lr_scheduler_type cosine \
    --max_length 512 # default is 2048

    output_model_path="${base_model_path}/${iteration_prefix}_iter${iteration}"
    mkdir -p $output_model_path

    # initializing with config file would cause an error
#    accelerate launch --config_file ./configs/zero3.yaml ./inpo/inpo_train.py --run_name "${iteration_prefix}_${iteration}" \
#        --output_dir $output_model_path --model_name_or_path $previous_model --learning_rate 5e-7 --ratio $ratio --eta $eta \
#        --train_dir $pref_prob_path --loss_type inpo --lr_scheduler_type cosine

    # accelerate launch ./inpo/inpo_train.py --run_name "${iteration_prefix}_${iteration}" \
    # --output_dir $output_model_path --model_name_or_path $previous_model --learning_rate 5e-7 --ratio $ratio --eta $eta \
    # --train_dir $pref_prob_path --loss_type inpo --lr_scheduler_type cosine --report_to wandb

    CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 ./inpo/inpo_train.py \
    --run_name "${iteration_prefix}_${iteration}" \
    --output_dir $output_model_path \
    --model_name_or_path $previous_model \
    --learning_rate 5e-7 \
    --ratio $ratio \
    --eta $eta \
    --train_dir $pref_prob_path --loss_type inpo --lr_scheduler_type cosine --report_to wandb
}


# Main loop for iterations
for i in {1..3}
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
