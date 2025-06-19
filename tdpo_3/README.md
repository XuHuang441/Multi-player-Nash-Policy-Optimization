# INPO

Use ``run_inpo.sh`` to run the pipeline

**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets
pip install torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0-cp310-cp310-manylinux1_x86_64.whl 
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install accelerate==0.27.2
pip install deepspeed

pip install transformers==4.38.2
pip install numpy==1.26.4
```

**Training Environment**

```sh
conda create -n rlhf python=3.10.9
conda activate rlhf
pip install torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
python -m pip install .
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install accelerate==0.27.2
```

**Evaluation**
See ``get_alpaca_answer.sh`` for an example.

## 📌 Additional Setup Notes

### 🔗 PyTorch Wheel Download

To install PyTorch 2.1.2 with CUDA 11.8 support for Python 3.10, use the following wheel:

```
https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=60396358193f238888540f4a38d78485f161e28ec17fa445f0373b5350ef21f0
```

Install it using:

```bash
pip install https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl
```

### 🛠 `run_inpo.sh` Adjustment

To avoid errors when creating directories that may already exist, update all `mkdir` commands in `run_inpo.sh` by adding the `-p` flag. For example:

```bash
mkdir -p your/path/here
```

### ⚠️ NumPy Version Compatibility

In the training environment, make sure to downgrade NumPy to avoid potential compatibility issues:

```bash
pip install numpy==1.26.4
```

### Install wandb


