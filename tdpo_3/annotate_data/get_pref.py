import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="iter2_K64.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="iter2_K64_Mreward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    # RLHFlow/pair-preference-model-LLaMA3-8B
    preference_name_or_path: Optional[str] = field(
        default="RLHFlow/pair-preference-model-LLaMA3-8B",
        metadata={"help": "the name of the preference model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=2,
        metadata={"help": "the number of responses per prompt"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
accelerator = Accelerator()
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(script_args.preference_name_or_path,
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
tokenizer = AutoTokenizer.from_pretrained(script_args.preference_name_or_path, use_fast=True)
tokenizer_plain = AutoTokenizer.from_pretrained(script_args.preference_name_or_path, use_fast=True)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
token_id_A = token_id_A[0]
token_id_B = token_id_B[0]
model.eval()
temperature = 1.0

def get_pref(context, responses):
    probs_chosen = []
    for chosen_position in [0, 1]:
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.encode(tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, ""), return_tensors='pt', add_special_tokens=False).to(device)

        with torch.no_grad():
            output = model(input_ids)
        logit_A = output.logits[0, -1, token_id_A].item()
        logit_B = output.logits[0, -1, token_id_B].item()
        # take softmax to get the probability; using numpy
        Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
        logit_chosen = [logit_A, logit_B][chosen_position]
        prob_chosen = np.exp(logit_chosen / temperature) / Z
        probs_chosen.append(prob_chosen)
    
    avg_prob_chosen = np.mean(probs_chosen)
    correct = 0.5 if avg_prob_chosen == 0.5 else float(avg_prob_chosen > 0.5)
    return correct

ds_dir = script_args.dataset_name_or_path
# "prompt", "responses"
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")
accelerator = Accelerator()

world_size = int(os.getenv("WORLD_SIZE", "1"))
local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
print(len(ds))

data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    cnt = 0
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        responses = sample["responses"]
        if len(responses) < script_args.K:
            continue
        
        context_message = sample["context_messages"]
        context = tokenizer_plain.apply_chat_template(context_message, tokenize=False)
        chosen_A = get_pref(context, responses)
        if chosen_A > 0.5:
            flag = 'A'
        elif chosen_A < 0.5:
            flag = 'B'
        else:
            flag = 'tie'
        data.append({"prompt": sample["prompt"], "responses": responses, "chosen": flag})


all_process_list = [{}] * world_size
data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)



if local_rank == 0:
    print("Collect {} data from {} inputs".format(len(gathered_data), data_size))
    if len(gathered_data) < data_size:
        print(
            "Some of the prompts are with responses < {}. This can happen because the prompt is too long and is ignored by VLLM".format(
                script_args.K
            )
        )
    output_eval_dataset = {}
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

   