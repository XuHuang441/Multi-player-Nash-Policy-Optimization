import copy
import sys

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trainer import PreComputer
from trl import DPOConfig
import os

# ========== 配置 ==========
DATASET_PATH = "output/output_pref_0.json"
OUTPUT_PATH = "output/precomputed_dataset"
MODEL_PATH = "meta-llama/Llama-3.2-1B"

BETA = 0.005
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1000
EOT_TOKEN = ""
SANITY_CHECK = False
last_name = "meta-llama/Llama-3.2-1B"

# ========== Step 1: 加载模型和 tokenizer ==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ========== 加载偏好数据 ==========
ds = load_dataset("json", data_files=DATASET_PATH, split="train", field="instances")

prompts, chosen, rejected = [], [], []
tie_count = 0

for sample in ds:
    responses = sample["responses"]
    if sample["chosen"] == "A":
        prompts.append(sample["prompt"])
        chosen.append(responses[0] + EOT_TOKEN)
        rejected.append(responses[1] + EOT_TOKEN)
    elif sample["chosen"] == "B":
        prompts.append(sample["prompt"])
        chosen.append(responses[1] + EOT_TOKEN)
        rejected.append(responses[0] + EOT_TOKEN)
    else:
        tie_count += 1

print(f"Tie count: {tie_count}")

dataset = Dataset.from_dict({
    "prompt": prompts,
    "chosen": chosen,
    "rejected": rejected
})

if SANITY_CHECK:
    dataset = dataset.select(range(min(len(dataset), 100)))

# ========== 构建 TrainingArguments ==========
training_args = DPOConfig(
    per_device_train_batch_size=1,
    num_train_epochs=1,
    output_dir="./tmp_debug",
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
)

# ========== 初始化 PreComputer：使用相同 model 和 ref_model ==========
ref_model = copy.deepcopy(model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

pre = PreComputer(
    model=model,            # last_model
    ref_model=ref_model,        # ref_model
    args=training_args,
    beta=BETA,
    train_dataset=dataset,
    tokenizer=tokenizer,
    loss_type="sigmoid",       # 视后续训练而定
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_length=MAX_LENGTH,
    mask_prompt=False,
    len_penalty=0,
)

print("[✓] Computing reference logps...")
reference_chosen_logps, reference_rejected_logps = pre.precompute()
print(f"reference_chosen_logps: {reference_chosen_logps} reference_rejected_logps: {reference_rejected_logps}")

ref_model = AutoModelForCausalLM.from_pretrained(
    last_name,
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
)

pre = PreComputer(
    model=model,            # last_model
    ref_model=ref_model,        # ref_model
    args=training_args,
    beta=BETA,
    train_dataset=dataset,
    tokenizer=tokenizer,
    loss_type="sigmoid",       # 视后续训练而定
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_length=MAX_LENGTH,
    mask_prompt=False,
    len_penalty=0,
)

print("[✓] Computing last logps...")
last_chosen_logps, last_rejected_logps = pre.precompute()
print(f"last_chosen_logps: {last_chosen_logps} last_rejected_logps: {last_rejected_logps}")

# ========== 保存带 logp 的 dataset ==========
pre_dataset = pre.train_dataset
pre_dataset = pre_dataset.add_column("reference_chosen_logps", reference_chosen_logps)
pre_dataset = pre_dataset.add_column("reference_rejected_logps", reference_rejected_logps)
pre_dataset = pre_dataset.add_column("last_chosen_logps", last_chosen_logps)
pre_dataset = pre_dataset.add_column("last_rejected_logps", last_rejected_logps)

os.makedirs(OUTPUT_PATH, exist_ok=True)
pre_dataset.save_to_disk(OUTPUT_PATH)
print(f"[✓] Precomputed dataset saved to: {OUTPUT_PATH}")
