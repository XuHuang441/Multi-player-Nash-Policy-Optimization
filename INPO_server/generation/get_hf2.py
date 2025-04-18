#!/usr/bin/env python
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

# previous swap space is 8
llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_new_tokens,
    load_format="auto",
    swap_space=2, # todo was 16
    seed=42,
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
    max_tokens=script_args.max_new_tokens,
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



one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

prompts = ds["prompt"]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {"context_messages": ds[i]["context_messages"], "prompt": prompts[i], "responses": [out.text for out in output.outputs]}
    gathered_data.append(tmp_data)

# for sample in tqdm(ds):
#     prompt = sample["prompt"]
#     output = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
#     responses = [out.text for out in output.outputs]
#     print(responses)
#     tmp_data = {"context": sample, "prompt": prompt, "responses": responses}
#     gathered_data.append(tmp_data)
#     exit()

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
print("I collect ", len(gathered_data), "samples")

output_path = script_args.output_dir + '_' + str(script_args.local_index) + ".json"
print(output_path)

with open(output_path, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
