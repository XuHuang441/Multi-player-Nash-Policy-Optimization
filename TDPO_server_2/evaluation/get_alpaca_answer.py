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
import json
import time

import tiktoken
from tqdm import tqdm
from conversation import get_conv_template



@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name: Optional[str] = field(
        default="INPO",
        metadata={"help": "model name"},
    )
    model_path: Optional[str] = field(
        default="",
        metadata={"help": "evaluation model path"},
    )
    conv_temp: Optional[str] = field(
        default="myllama3",
        metadata={"help": "conversation template"},
    )
    max_new_tokens: Optional[int] = field(
        default=4096,
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
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name = script_args.model_name
model_path = script_args.model_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="float16",
    max_model_len=script_args.max_new_tokens,
    load_format="auto",
    seed=42,
)

chat_template = script_args.conv_temp
print(chat_template)
conv = get_conv_template(chat_template)
sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=1,
    stop_token_ids=conv.stop_token_ids,
    #stop=["<|user|>"],
)


eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
answers = []
for sample in tqdm(eval_set):
    # generate here is a placeholder for your models generations
    qs = sample["instruction"]
    conv = get_conv_template(chat_template)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    out = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    assert len(out) == 1
    output = out[0].outputs[0].text 
    ans = {
        "instruction": qs,
        "output": output,
        "generator": model_name,
        "dataset": sample["dataset"]
    }
    answers.append(ans)
   
answer_file = os.path.join("res", "{}.json".format(model_name))
print("Output to {}".format(answer_file))
with open(answer_file, "w") as file:
    json.dump(answers, file)





# for question in tqdm(questions):
#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
#     choices = []
#     for i in range(num_choices):
#         turns = []
#         conv = get_conv_template("myllama3")
#         for j in range(len(question["turns"])):
#             qs = question["turns"][j]["content"]
#             conv.append_message(conv.roles[0], qs)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()
#             out = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
#             assert len(out) == 1
#             output = out[0].outputs[0].text 
#             conv.update_last_message(output)
#             turns.append({"content": output, "token_len": len(encoding.encode(output))})
#         choices.append({"index": i, "turns": turns})

#     # Dump answers
#     ans = {
#         "question_id": question["question_id"],
#         "answer_id": shortuuid.uuid(),
#         "model_id": model_name,
#         "choices": choices,
#         "tstamp": time.time(),
#     }

#     os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#    

# print(model_name)

