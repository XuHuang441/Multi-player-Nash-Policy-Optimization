import sys
from pprint import pprint

import torch
import copy
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

from trainer import MyPreferenceTrainer, INPOTrainer, PrecomputeDataCollator

# ========== arguments ==========
MODEL_PATH = "meta-llama/Llama-3.2-1B"
PRECOMPUTED_DATA_PATH = "output/precomputed_dataset"
OUTPUT_DIR = "output/inpo_hf_model"
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1024

ETA = 0.005
RATIO = 1/3
BATCH_SIZE = 1
EPOCHS = 1
GRAD_ACC = 1
LEARNING_RATE = 5e-7
LOGGING_STEPS = 1

# ========== Step 1: load model and tokenizer ==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ========== Step 2:  ref_model ==========
ref_model = copy.deepcopy(model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ========== Step 3: load dataset ==========
train_dataset = load_from_disk(PRECOMPUTED_DATA_PATH)
print(f"[✓] Loaded dataset with {len(train_dataset)} examples.")

# ========== Step 4: training arguments ==========
training_args = DPOConfig(
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACC,
    num_train_epochs = EPOCHS,
    learning_rate = LEARNING_RATE,
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.03,
    output_dir = OUTPUT_DIR,
    logging_steps = LOGGING_STEPS,
    save_strategy = "epoch",
    bf16 = True,
    remove_unused_columns = False,
    run_name = "inpo_hf_run",
    report_to = "none",
)

# ========== Step 5: Initialize Trainer ==========
# trainer = MyPreferenceTrainer(
#     model = model,
#     ref_model = ref_model,
#     args = training_args,
#     ratio = RATIO,
#     eta = ETA,
#     train_dataset = train_dataset,
#     tokenizer = tokenizer,
#     loss_type = "inpo",
#     max_prompt_length = MAX_PROMPT_LENGTH,
#     max_length = MAX_LENGTH,
#     mask_prompt = False,
#     len_penalty = 0,
#     precompute_ref_log_probs=False
#
# )

trainer = INPOTrainer(
    model=model,
    args=training_args,  # transformers.TrainingArguments
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=PrecomputeDataCollator(tokenizer),
    ratio=RATIO,
    eta=ETA,
    beta=0.01,
    len_penalty=0.0,
)

# ========== Step 6: train and save ==========
print("[✓] Begin INPO training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"[✓] INPO HF model saved to {OUTPUT_DIR}")
