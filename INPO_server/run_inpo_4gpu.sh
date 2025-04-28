# source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

# Base paths and settings
initial_model="RLHFlow/LLaMA3-SFT"
base_dataset_path="dataset_path/datasets"
base_model_path="model_path/models"
ratio=$(echo "scale=10; 1/3" | bc)
eta=0.005
iteration_prefix="inpo"

run_iteration() {
    local iteration=$1
    local previous_model=$2
    local input_path=$3
    local json_output=$4
    local pref_output=$5

    # conda activate mypo
    my_world_size=4       # ← now using 4 GPUs
    sanity_check=False
    use_tour=True
    K=4                   # ← match the number of GPUs

    # ── 1) Generation on 4 GPUs ──────────────────────────────────────────
    for local_index in $(seq 0 $((my_world_size-1))); do
      CUDA_VISIBLE_DEVICES=$local_index \
        python ./generation/get_hf2.py \
          --model_name_or_path ${previous_model} \
          --dataset_name_or_path ${input_path} \
          --output_dir ${json_output} \
          --sanity_check $sanity_check \
          --K $K \
          --temperature 1.0 \
          --local_index $local_index \
          --my_world_size ${my_world_size} \
          --eos_ids 128009 &
    done
    wait

    python ./generation/merge_data.py \
      --base_path $json_output \
      --output_dir "${json_output}.json" \
      --num_datasets $my_world_size

    # ── 2) Preference annotation on 4 GPUs ────────────────────────────
    for local_index in $(seq 0 $((my_world_size-1))); do
      CUDA_VISIBLE_DEVICES=$local_index \
        accelerate launch annotate_data/get_pref_single.py \
          --use_tournament $use_tour \
          --dataset_name_or_path "${json_output}_${local_index}.json" \
          --output_dir "${pref_output}_${local_index}.json" \
          --K $K &
    done
    wait

    python ./annotate_data/merge.py \
      --base_path $pref_output \
      --output_dir "${pref_output}_data.json" \
      --num_datasets $my_world_size

    # ── 3) Precompute & training (unchanged) ───────────────────────────
    pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration}/data_pref_prob"
    mkdir -p $pref_prob_path

    accelerate launch ./inpo/precompute.py \
      --run_name "${iteration_prefix}_${iteration}" \
      --train_dir "${pref_output}_data.json" \
      --output_dir $pref_prob_path \
      --ref_model $initial_model \
      --last_model $previous_model \
      --loss_type inpo \
      --lr_scheduler_type cosine

    output_model_path="${base_model_path}/${iteration_prefix}_iter${iteration}"
    mkdir -p $output_model_path

    accelerate launch ./inpo/inpo_train.py \
      --run_name "${iteration_prefix}_${iteration}" \
      --output_dir $output_model_path \
      --model_name_or_path $previous_model \
      --learning_rate 5e-7 \
      --ratio $ratio \
      --eta $eta \
      --train_dir $pref_prob_path \
      --loss_type inpo \
      --lr_scheduler_type cosine \
      --report_to wandb
}

# Main loop for iterations
for i in {1..3}; do
    iteration_name="iter${i}"
    input_path="RLHFlow/iterative-prompt-v1-iter${i}-20K"
    mkdir -p "${base_dataset_path}/${iteration_prefix}_${iteration_name}"
    json_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data"
    pref_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/pref"

    if [ $i -eq 1 ]; then
        previous_model=$initial_model
    else
        previous_iteration=$((i-1))
        previous_model="${base_model_path}/${iteration_prefix}_iter${previous_iteration}"
    fi

    run_iteration $i $previous_model $input_path $json_output $pref_output
done