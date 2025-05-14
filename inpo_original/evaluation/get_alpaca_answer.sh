source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

model_name="inpo_iter3"
model_path="model_path/inpo_iter3"
CUDA_VISIBLE_DEVICES=0 python get_answer.py --model_name $model_name --model_path $model_path --conv_temp "myllama3"