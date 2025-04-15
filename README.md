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
pip install transformers accelerate datasets unsloth
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


## Additional Information

To use the model **`meta-llama/Llama-3.2-1B`**, ensure you request and receive access via Hugging Face beforehand.

