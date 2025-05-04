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

set -e # Exit immediately if a command exits with a non-zero status

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local previous_model=$2
    local input_path=$3
    local json_output=$4
    local pref_output=$5

    sanity_check=False
    use_tour=True
    K=8

    echo "Starting generation for iteration ${iteration}..."

    CUDA_VISIBLE_DEVICES=0,1,2,3 python generation/get_hf2.py \
    --model_name_or_path ${previous_model} \
    --dataset_name_or_path ${input_path} \
    --output_dir ${json_output}.json \
    --sanity_check $sanity_check \
    --K $K \
    --temperature 1.0 \
    --eos_ids 128009

    echo "Starting preference modeling..."

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 29601 \
    annotate_data/get_pref_single.py \
    --use_tournament $use_tour \
    --dataset_name_or_path "${json_output}.json" \
    --output_dir "${pref_output}.json" \
    --K $K

    pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data_pref_prob"
    mkdir -p $pref_prob_path

    # Make sure to change the number of GPUs both here and in the `zero2/3.yaml` config file

    accelerate launch --config_file ./configs/zero2.yaml ./inpo/precompute.py \
    --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}.json" \
    --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model \
    --loss_type inpo --lr_scheduler_type cosine \

    output_model_path="${base_model_path}/${iteration_prefix}_iter${iteration}"
    mkdir -p $output_model_path

    accelerate launch --config_file ./configs/zero3.yaml ./inpo/inpo_train.py \
    --run_name "${iteration_prefix}_${iteration}" \
    --output_dir $output_model_path \
    --model_name_or_path $previous_model \
    --learning_rate 5e-7 \
    --ratio $ratio \
    --eta $eta \
    --train_dir $pref_prob_path \
    --loss_type inpo \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --gradient_accumulation_steps 1 # this has to be the same as the one in the zero3 config

    # merge deepspeed checkpoints
    echo "Merging deepspeed checkpoints..."
    python $output_model_path/zero_to_fp32.py $output_model_path $output_model_path
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
