import os
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from datasets import Dataset, load_dataset
from trl import DPOConfig

from trainer import PreComputer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    beta: Optional[float] = field(default=0.005, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="sshleifer/tiny-gpt2",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    last_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the last iteratioin model name or path"},
    )
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default=None,  # "/export/home/data/gemma_it_2b_3w_k8_with_pairrm_rewards.json",
        metadata={"help": "the location of the evalset name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the saving strategy"})
    save_steps: Optional[int] = field(default=50000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "the run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type"})
    output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_min", metadata={"help": "the choose type"})

    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})


def prepare_data(
    data_dir: str = "/home/xiongwei/data/helpful/rm/rm1003.json",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    eot_token="",
    length_penalty=0,
) -> Dataset:
    """Prepare the dataset for DPO training by rejection sampling.
    We implement different strategies to select pairs, including
    max_min: best v.s. worst
    max_random: best v.s. random from the remaining;
    max_max: best v.s. second best
    max_min_p: best v.s. worst but we additionally add a length penalty in the reward value
    """
    ds = load_dataset("json", data_files=data_dir, split="train", field="instances")
    pos = []
    neg = []
    prompts = []

    for sample in ds:
        responses = sample["responses"]
        chosen = sample["chosen"]
        if chosen == 'A':
            prompts.append(sample["prompt"])
            pos.append(responses[0] + eot_token)
            neg.append(responses[1] + eot_token)
        elif chosen == 'B':
            prompts.append(sample["prompt"])
            pos.append(responses[1] + eot_token)
            neg.append(responses[0] + eot_token)

    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg})
    print("Tie {} samples".format(len(ds)-len(dataset)))

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # model = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     use_flash_attention_2=True,
    #     torch_dtype=torch.float16,
    # )
    # model.config.use_cache = False

    random_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        random_model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in random_model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    last_name = script_args.last_model

    model = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    

    tokenizer = AutoTokenizer.from_pretrained(ref_name)
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token


    # 2. Load the Stack-exchange paired dataset
    train_dataset = prepare_data(
        data_dir=script_args.train_dir,
        sanity_check=script_args.sanity_check,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
    )
    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        # optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
    )
    # print(training_args)

    # 5. initialize the DPO trainer

    pre = PreComputer(
        random_model,
        ref_model=model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        loss_type=script_args.loss_type,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        mask_prompt=script_args.mask_prompt,
        len_penalty=script_args.len_penalty,
    )
    print("begin to precompute")
    reference_chosen_logps, reference_rejected_logps = pre.precompute()
    # for s in pre_dataset:
    #     print(len(s["chosen_input_ids"]), len(s["chosen_attention_mask"]), len(s["chosen_labels"]))

    model = AutoModelForCausalLM.from_pretrained(
        last_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    pre = PreComputer(
        random_model,
        ref_model=model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        loss_type=script_args.loss_type,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        mask_prompt=script_args.mask_prompt,
        len_penalty=script_args.len_penalty,
    )
    last_chosen_logps, last_rejected_logps = pre.precompute()
    pre_dataset = pre.train_dataset

    pre_dataset = pre_dataset.add_column(name="reference_chosen_logps", column=reference_chosen_logps)
    pre_dataset = pre_dataset.add_column(name="reference_rejected_logps", column=reference_rejected_logps)
    pre_dataset = pre_dataset.add_column(name="last_chosen_logps", column=last_chosen_logps)
    pre_dataset = pre_dataset.add_column(name="last_rejected_logps", column=last_rejected_logps)

    pre_dataset.save_to_disk(script_args.output_dir, num_shards=1)
    # with open(output_path, "w", encoding="utf8") as f:
    #     json.dump(pre_dataset, f, ensure_ascii=False)


   
   