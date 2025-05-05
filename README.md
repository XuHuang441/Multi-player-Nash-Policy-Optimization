# Multi-player Nash Policy Optimization

## Environment Setup

This guide explains how to set up the development environment for running Multi-player Nash Policy Optimization.

### System Requirements
- Operating System: Linux

### Step-by-step Setup

#### 1. Create a Conda Environment

Create a fresh Conda environment with Python 3.11:
```bash
conda create --name mypo python=3.11
conda activate mypo
```

#### 2. Install PyTorch

Install PyTorch and associated libraries:
```bash
pip install torch torchvision torchaudio
```

#### 3. Install Project Dependencies

Install essential dependencies:
```bash
pip install transformers==4.48.2 trl==0.13.0 accelerate datasets unsloth wandb
```

#### 4. Install Flash Attention

Install pre-built Flash Attention optimized for CUDA:
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

After completing this step, you can verify the installation by running:
```bash
python test_flash_attn.py
```

#### 5. Install DeepSpeed

1. Install CUDA compiler:
```bash
conda install -c nvidia cuda-compiler
```

2. Set `CUDA_HOME` environment variable:
```bash
export CUDA_HOME=/home/xu/anaconda3/envs/mypo/
```

3. Install DeepSpeed:
```bash
pip install deepspeed
```

## Post-Deployment Adjustments on Your Server

After setting up the environment on your own server, make the following adjustments to ensure proper multi-GPU training and smooth execution:

1. **Update `num_processes` in the config**
   In your configuration file (e.g., `zero2.yaml` or `zero3.yaml`), set `num_processes` to match the number of GPUs you plan to use. For example:

   ```yaml
   num_processes: 4  # if using 4 GPUs
   ```

2. **Adjust `CUDA_VISIBLE_DEVICES` in shell scripts**
   In your shell script used to launch training, set the `CUDA_VISIBLE_DEVICES` environment variable to the appropriate GPU IDs:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

3. **Login to Weights & Biases (wandb)**
   Make sure to log into your wandb account from the command line before training (only needs to be done once):

   ```bash
   wandb login
   ```

4. **Disable sanity check**
   Disable it to use the full dataset:

   ```yaml
   sanity_check=false
   ```


## Additional Information

To use the model **`meta-llama/Llama-3.2-1B`**, ensure you request and receive access via Hugging Face beforehand.

