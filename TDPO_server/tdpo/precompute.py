import os
import argparse
import torch
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer import PreComputer
import sys
from trl import DPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="TDPO precompute script")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--reference_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)  
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_history_t", type=int, default=2)
    parser.add_argument("--history_paths", type=str, nargs="*", default=[])
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--eot_token", type=str, default="")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=50000)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--run_name", type=str, default="precompute_run")
    return parser.parse_args()


def prepare_data(data_dir: str, eot_token="") -> Dataset:
    ds = load_dataset("json", data_files=data_dir, split="train", field="instances")
    prompts, pos, neg = [], [], []
    for sample in ds:
        if sample["chosen"] == "A":
            prompts.append(sample["prompt"])
            pos.append(sample["responses"][0] + eot_token)
            neg.append(sample["responses"][1] + eot_token)
        elif sample["chosen"] == "B":
            prompts.append(sample["prompt"])
            pos.append(sample["responses"][1] + eot_token)
            neg.append(sample["responses"][0] + eot_token)
    return Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg})


def precompute_multi_history(
    pre: PreComputer,
    history_model_paths: List[str],
):
    history_logps_list = []
    for step_idx, model_path in enumerate(history_model_paths):
        print(f"[History Step {step_idx}] Loading model from: {model_path}")
        history_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        pre.set_ref_model(history_model)

        chosen_logps, rejected_logps = pre.precompute()
        history_logps_list.append((chosen_logps, rejected_logps))
    return history_logps_list


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.reference_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    random_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.reference_model_path, torch_dtype=torch.bfloat16)

    train_dataset = prepare_data(args.data_path, args.eot_token)

    training_args = DPOConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        # report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        # optim=args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=args.run_name,
    )
    
    # precompute reference model logps
    pre = PreComputer(
        model=random_model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        beta=args.beta,
        max_prompt_length=1024,
        max_length=2048,
        len_penalty=0,
    )
    ref_chosen_logps, ref_rejected_logps = pre.precompute()

    # precompute history model logps
    history_paths = args.history_paths
    if args.max_history_t > 0 and history_paths:
        history_paths = history_paths[-args.max_history_t:][::-1]

    history_logps = []
    if history_paths:
        history_logps = precompute_multi_history(
            pre=pre,
            history_model_paths=history_paths,
        )

    pre_dataset = pre.train_dataset
    pre_dataset = pre_dataset.add_column("reference_chosen_logps", ref_chosen_logps)
    pre_dataset = pre_dataset.add_column("reference_rejected_logps", ref_rejected_logps)
    for j, (cj, rj) in enumerate(history_logps):
        pre_dataset = pre_dataset.add_column(f"history{j}_chosen_logps", cj)
        pre_dataset = pre_dataset.add_column(f"history{j}_rejected_logps", rj)

    os.makedirs(args.output_dir, exist_ok=True)
    pre_dataset.save_to_disk(args.output_dir)
    print(f"✅ Precompute finished and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
