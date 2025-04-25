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
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})
    use_tournament: Optional[bool] = field(default=False, metadata={"help": "only train on 100 samples"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
accelerator = Accelerator()
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(script_args.preference_name_or_path,
                                             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
tokenizer = AutoTokenizer.from_pretrained(script_args.preference_name_or_path, use_fast=True)
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

def get_match_res(context, responses, id_0, id_1):
    response_pair = [responses[id_0], responses[id_1]]
    chosen_A = get_pref(context, response_pair)
    if chosen_A >= 0.5:
        return id_0, id_1
    else:
        return id_1, id_0

ds_dir = script_args.dataset_name_or_path
# "prompt", "responses"
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")
accelerator = Accelerator()

if script_args.sanity_check:
    ds = ds.select(range(min(len(ds), 100)))

# world_size = int(os.getenv("WORLD_SIZE", "1"))
# local_rank = Accelerator().local_process_index

# data_size = len(ds["prompt"])

# share = int(data_size / world_size) + 1
# ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
print(len(ds))

data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    cnt = 0
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        responses = sample["responses"]
        n = len(responses)
        if n < script_args.K:
            continue
        
        context_message = sample["context_messages"]
        context = tokenizer_plain.apply_chat_template(context_message, tokenize=False)
        if n == 2:
            win_id, lose_id = 0, 1
        elif n == 8 and script_args.use_tournament:
            win_1, lose_1 = get_match_res(context, responses, 0, 1)
            win_2, lose_2 = get_match_res(context, responses, 2, 3)
            win_3, lose_3 = get_match_res(context, responses, 4, 5)
            win_4, lose_4 = get_match_res(context, responses, 6, 7)
            
            win_5, __ = get_match_res(context, responses, win_1, win_2)
            win_6, __ = get_match_res(context, responses, win_3, win_4)
            win_id, __ = get_match_res(context, responses, win_5, win_6)

            __, lose_5 =  get_match_res(context, responses, lose_1, lose_2)
            __, lose_6 =  get_match_res(context, responses, lose_3, lose_4)
            __, lose_id = get_match_res(context, responses, lose_5, lose_6)

            # print("Yes! We use tournament!")
        else:
            win_id = 0
            for i in range(1, n):
                response_pair = [responses[win_id], responses[i]]
                chosen_A = get_pref(context, response_pair)
                if chosen_A < 0.5:
                    win_id = i
            
            if win_id == 0:
                lose_id = 1
            else:
                lose_id = 0
            
            for i in range(n):
                if i == win_id or i == lose_id:
                    continue
                response_pair = [responses[lose_id], responses[i]]
                chosen_A = get_pref(context, response_pair)
                if chosen_A >= 0.5:
                    lose_id = i
            
        assert win_id != lose_id

        response_pair = [responses[win_id], responses[lose_id]]
        chosen_A = get_pref(context, response_pair)
        if n > 2 and chosen_A <= 0.5:
            print("we don't know which one is better")
            continue
        
        if chosen_A > 0.5:
            flag = 'A'
        elif chosen_A < 0.5:
            flag = 'B'
        else:
            flag = 'tie'
        data.append({"prompt": sample["prompt"], "responses": response_pair, "chosen": flag})

output_eval_dataset = {}
output_eval_dataset["instances"] = data
with open(script_args.output_dir, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)

   
