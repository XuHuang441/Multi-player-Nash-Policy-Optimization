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





