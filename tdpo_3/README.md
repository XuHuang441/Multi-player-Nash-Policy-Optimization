# INPO

Use ``run_inpo.sh`` to run the pipeline

**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets
pip install https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl
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
pip install https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl
pip install torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
python -m pip install .
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install accelerate==0.27.2 numpy==1.26.4 wandb transformers==4.38.2
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


## To Run Gemma 

```shell
# for vllm
/home/hubing/miniconda3/envs/vllm/bin/pip install transformers==4.42.3
/home/hubing/miniconda3/envs/vllm/bin/pip install vllm==0.5.3
/home/hubing/miniconda3/envs/vllm/bin/pip install flashinfer==0.0.9 -i https://flashinfer.ai/whl/cu121/torch2.3/
/home/hubing/miniconda3/envs/vllm/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# for rlhf
/home/hubing/miniconda3/envs/rlhf/bin/pip install transformers==4.42.3

```
Manually add function in `core.py` :
```python
# file directory:
# /home/hubing/miniconda3/envs/rlhf/lib/python3.10/site-packages/trl/core.py

# comment out this line:
# from transformers import top_k_top_p_filtering 

# add the function below:
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
```

### July 11th

New inference environment

```shell
conda create -n vllm python=3.10.9
conda activate vllm
/home/hubing/miniconda3/envs/vllm/bin/pip install datasets
# torch2.6+cu124
/home/hubing/miniconda3/envs/vllm/bin/pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#vllm
/home/hubing/miniconda3/envs/vllm/bin/pip install vllm==0.8.5
#flashinfer
/home/hubing/miniconda3/envs/vllm/bin/pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
/home/hubing/miniconda3/envs/vllm/bin/pip install accelerate deepspeed
# flash attention
/home/hubing/miniconda3/envs/vllm/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```