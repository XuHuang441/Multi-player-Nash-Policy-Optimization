import json
import os

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ------------------------
# Arguments
# ------------------------
SEED = 42
OUTPUT_DIR = "output/generation_output"
LOCAL_INDEX = 0
SANITY_CHECK = False
WORLD_SIZE = 4
K = 8
MAX_INPUT_LENGTH = 10000
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
USE_BEAM_SEARCH = False
DATASET_KEY = "content"
EOS_IDS = []

# ------------------------
# Random seed
# ------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------
# Load model and tokenizer
# ------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LENGTH = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ------------------------
# Load dataset
# ------------------------
with open("dataset/toy.json", "r") as file:
    data = json.load(file)
ds = Dataset.from_list(data)

# ------------------------
# build_chat
# ------------------------
def build_chat(prompt):
    return [{"role": "user", "content": prompt}]

def apply_chat_template(messages):
    # llama3 style chat template
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}\n"
    prompt += "<|assistant|>\n"
    return prompt

ds = ds.map(lambda x: {"messages": build_chat(x[DATASET_KEY])})

# ------------------------
# Generation
# ------------------------
gathered_data = []
print(f"Generating with {len(ds)} prompts...")

for i in tqdm(range(len(ds))):
    messages = ds[i]["messages"]
    prompt = apply_chat_template(messages)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=not USE_BEAM_SEARCH,
        temperature=TEMPERATURE,
        top_p=1.0,
        num_return_sequences=K,
        eos_token_id=[tokenizer.eos_token_id] + EOS_IDS,
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    gathered_data.append({
        "prompt": prompt,
        "responses": responses
    })

# ------------------------
# Save results
# ------------------------
output_eval_dataset = {
    "type": "text_only",
    "instances": gathered_data,
}

output_path = f"{OUTPUT_DIR}_{LOCAL_INDEX}.json"
os.makedirs("output", exist_ok=True)
with open(output_path, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False, indent=2)

print(f"[✓] Saved {len(gathered_data)} samples to {output_path}")
