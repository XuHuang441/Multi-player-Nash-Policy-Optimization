import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from trl import DPOConfig

from trainer import MyPreferenceTrainer, INPOTrainer, PrecomputeDataCollator, INPOTrainer_v2
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForImageClassification
)
from accelerate import Accelerator, DistributedType

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    ratio: Optional[float] = field(default=0, metadata={"help": "the parameter for KL"})
    eta: Optional[float] = field(default=0.0075, metadata={"help": "the parameter for OMD"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="/apdcephfs_us/share_300814644/user/yuhenyzhang/hf_models/LLaMA3-SFT",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="sshleifer/tiny-gpt2",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="./data/iter1/data_reward_preprob",
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
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
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
    save_strategy: Optional[str] = field(default="no", metadata={"help": "the saving strategy"})
    save_steps: Optional[int] = field(default=19980526, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=19980218, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="inpo", metadata={"help": "the run name"})
    loss_type: Optional[str] = field(default="inpo", metadata={"help": "the loss type"})
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

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    train_dataset = load_from_disk(script_args.train_dir)
    if script_args.sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 100)))

    # Load evaluation dataset if provided
    eval_dataset = None
    if script_args.eval_dir is not None:
        eval_dataset = load_from_disk(script_args.eval_dir)
        if script_args.sanity_check:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 100)))


    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        model_ref.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        model_ref.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))

    # Create the data collator
    data_collator = PrecomputeDataCollator(
        tokenizer,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        mask_prompt=script_args.mask_prompt,
    )

    # 4. initialize training arguments:

    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        num_train_epochs=script_args.num_train_epochs,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        output_dir=script_args.output_dir,
        logging_steps=script_args.logging_steps,
        save_strategy=script_args.save_strategy,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        report_to=script_args.report_to,
        beta=0.1,  # Set the default beta for DPO directly in the config
    )
    # print(training_args)

    # 5. initialize the DPO trainer

    trainer = INPOTrainer_v2(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        ratio=script_args.ratio,
        eta=script_args.eta,
        len_penalty=script_args.len_penalty,
    )
    print("begin to train")
    # print(dpo_trainer._precomputed_train_ref_log_probs, dpo_trainer.precompute_ref_log_probs)

    # dataloader = dpo_trainer.get_train_dataloader()
    # for batch in dataloader:
    #     print(batch['reference_chosen_logps'].dtype, batch['reference_rejected_logps'].dtype)
    #     print(batch['reference_chosen_logps'].shape, batch['reference_rejected_logps'].shape)
    #     exit()

    # 6. train
    # inpo_trainer.train()
    # inpo_trainer.save_model(script_args.output_dir)

    trainer.train()
    trainer.save_model(script_args.output_dir)

      
    # # 7. save
    # output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    # dpo_trainer.model.save_pretrained(output_dir)
