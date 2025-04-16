#!/usr/bin/env python
import sys

from unsloth import FastLanguageModel, get_chat_template
import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm
import json
import torch


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize LLM with unsloth

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Load dataset
with open("dataset/toy.json", "r") as file:
    ds = json.load(file)
ds = Dataset.from_list(ds)

def build_chat(prompt):
    return [{"role": "user", "content": prompt}]

ds = ds.map(lambda x: {"messages": build_chat(x["content"])})

# ------------------------
# generation
# ------------------------
gathered_data = []
print(f"Generating with {len(ds)} prompts...")

for i in tqdm(range(len(ds))):
    messages = ds[i]["messages"]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to(model.device)

    # 生成多个 response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=script_args.max_new_tokens,
        do_sample=True,
        temperature=script_args.temperature,
        top_p=1.0,
        num_return_sequences=script_args.K,
        eos_token_id=[tokenizer.eos_token_id] + script_args.eos_ids,
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    gathered_data.append({
        "prompt": prompt,
        "responses": responses
    })

# ------------------------
# saving results
# ------------------------
output_eval_dataset = {
    "type": "text_only",
    "instances": gathered_data,
}

output_path = f"output/{script_args.output_dir}_{script_args.local_index}.json"

with open(output_path, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False, indent=2)

print(f"[✓] Saved {len(gathered_data)} samples to {output_path}")
