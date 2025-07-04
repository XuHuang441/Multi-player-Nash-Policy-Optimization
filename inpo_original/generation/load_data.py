import json
import random
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser

"""
If we use multiple VLLM processes to accelerate the generation, we need to use this script to merge them.
"""


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    base_path: Optional[str] = field(
        default="./data/",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    num_datasets: Optional[int] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


all_dirs = [script_args.base_path + str(i) + ".json" for i in range(script_args.num_datasets)]

gathered_data = []
for my_dir in all_dirs:
    ds = load_dataset("json", data_files=my_dir, split="train", field="instances")
    print(len(ds))
    for sample in ds:
        gathered_data.append(sample)

random.shuffle(gathered_data)

print("I collect ", len(gathered_data), "samples")

output_eval_dataset = {}
output_eval_dataset["type"] = "text_only"
output_eval_dataset["instances"] = gathered_data
with open(script_args.output_dir, "w", encoding="utf8") as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)
