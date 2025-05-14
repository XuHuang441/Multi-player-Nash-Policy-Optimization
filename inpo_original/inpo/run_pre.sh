source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate rlhf

initial_model="/apdcephfs_us/share_300814644/user/yuhenyzhang/hf_models/LLaMA3-SFT" 
base_dataset_path="/apdcephfs_us/share_300814644/user/yuhenyzhang/pref8_datasets"
base_model_path="/apdcephfs_us/share_300814644/user/yuhenyzhang/pref8_models"
tau=0
eta=0.005
iteration_prefix="ipo8pref0_0.05"

i=1
iteration=$i
iteration_name="iter${i}"
if [ $i -eq 1 ]; then
    previous_model=$initial_model
else
    previous_iteration=$((i-1))
    previous_model="${base_model_path}/${iteration_prefix}_iter${previous_iteration}"
fi
# pref_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/${iteration_prefix}_${iteration_name}_pref.json"
pref_output="${base_dataset_path}/${iteration_prefix}_${iteration_name}/pref"
pref_prob_path="${base_dataset_path}/${iteration_prefix}_${iteration_name}/data_pref_prob"
mkdir $pref_prob_path

accelerate launch --config_file ./configs/zero2.yaml ./ipo/precompute.py --run_name "${iteration_prefix}_${iteration}" --train_dir "${pref_output}_data.json" \
 --output_dir $pref_prob_path --ref_model $initial_model --last_model $previous_model --learning_rate 5e-7 \
 --loss_type ipo --lr_scheduler_type cosine
