#!/usr/bin/env python
import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
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

model_path = script_args.model_name_or_path
print("model_path", model_path)
print("Dataset_path", script_args.dataset_name_or_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize LLM with multi-GPU support (vLLM will use all available GPUs)
llm = LLM(
    model=script_args.model_name_or_path,
    tokenizer=script_args.model_name_or_path,
    dtype="bfloat16",
    tensor_parallel_size=torch.cuda.device_count(),  # use all visible GPUs
    swap_space=8,
    seed=script_args.seed,
)

# eos_token_id: 128009
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.chat_template is None:  # fix: INPO didn't include chat template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|user|>\n{{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|>\n{{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "<|assistant|>\n"
    )

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    # max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    #stop=["<|user|>"],
)


ds = load_dataset(script_args.dataset_name_or_path, split="train")
if script_args.sanity_check:
    ds = ds.select(range(min(len(ds), 5))) # was 100

ds = ds.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)
    }
)

data_size = len(ds["prompt"])
print("Data Size:{}".format(data_size))

prompts = ds["prompt"]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {"context_messages": ds[i]["context_messages"], "prompt": prompts[i], "responses": [out.text for out in output.outputs]}
    gathered_data.append(tmp_data)

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
print("I collect ", len(gathered_data), "samples")

# Save results
os.makedirs(os.path.dirname(script_args.output_dir) or ".", exist_ok=True)
with open(script_args.output_dir, "w", encoding="utf-8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(gathered_data)} samples to {script_args.output_dir}")
