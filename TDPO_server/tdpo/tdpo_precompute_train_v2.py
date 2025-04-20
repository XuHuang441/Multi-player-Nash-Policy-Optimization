import sys
from typing import List
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset, Dataset
from trl import DPOConfig
from trainer import TDPOTrainer, PreComputer, PrecomputeDataCollator
import os


def parse_args():
    parser = argparse.ArgumentParser(description="TDPO training script")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--reference_model_path", type=str, help="Path to the reference model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model")
    parser.add_argument("--precomputed_dir", type=str, required=True, help="Directory for precomputed dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--current_round", type=int, required=True, help="Current training round")
    parser.add_argument("--max_history_t", type=int, default=2, help="Maximum history length")
    parser.add_argument("--history_paths", type=str, nargs="*", default=[], help="List of paths to historical models")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--eta", type=float, default=0.01, help="Eta parameter")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio parameter")
    parser.add_argument("--beta", type=float, default=0.01, help="Beta parameter")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting system (wandb, tensorboard, none)")
    return parser.parse_args()


def prepare_data(data_dir: str, eot_token="", length_penalty=0) -> Dataset:
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
        prompts_dataset,
        random_model,
        history_model_paths: List[str],
        tokenizer,
        precomputer_class,
        training_args,
        **kwargs
):
    """
    Calculate log-probs from each of the multiple historical policy model paths, returning a list of historical logprobs

    return：
    - history_logps_list: List of (chosen_logps, rejected_logps) tensors
    """
    history_logps_list = []

    for step_idx, model_path in enumerate(history_model_paths):
        print(f"[History Step {step_idx}] Loading model from: {model_path}")
        history_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Construct PreComputer with the same random_model each time, change ref_model to history strategy
        pre = precomputer_class(
            random_model,
            ref_model=history_model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=prompts_dataset,
            **kwargs
        )

        print(f"[History Step {step_idx}] Computing log-probs...")
        chosen_logps, rejected_logps = pre.precompute()
        history_logps_list.append((torch.tensor(chosen_logps), torch.tensor(rejected_logps)))

    return history_logps_list


def main():
    args = parse_args()

    # Set reference model path to base model path if not provided
    if not args.reference_model_path:
        args.reference_model_path = args.base_model_path

    print(f"=== ROUND {args.current_round} ===")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)

    # Prepare data
    train_dataset = prepare_data(args.data_path)
    reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model_path, torch_dtype=torch.bfloat16)

    # Precompute reference
    pre = PreComputer(
        model=model,
        ref_model=reference_model,
        args=DPOConfig(output_dir="tmp", per_device_train_batch_size=1),
        beta=args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
        len_penalty=0
    )

    ref_chosen_logps, ref_rejected_logps = pre.precompute()

    # Precompute history (get most recent max_history_t models)
    history_model_paths = args.history_paths
    if args.max_history_t > 0 and history_model_paths:
        history_model_paths = history_model_paths[-args.max_history_t:][::-1]  # most recent comes first

    history_logps = []
    if history_model_paths:
        history_logps = precompute_multi_history(
            prompts_dataset=train_dataset,
            random_model=model,
            history_model_paths=history_model_paths,
            tokenizer=tokenizer,
            precomputer_class=PreComputer,
            training_args=DPOConfig(output_dir="tmp", per_device_train_batch_size=1),
            beta=args.beta,
            max_prompt_length=1024,
            max_length=2048,
            len_penalty=0,
        )

    # Add precomputed values to dataset
    pre_dataset = pre.train_dataset
    pre_dataset = pre_dataset.add_column("reference_chosen_logps", ref_chosen_logps)
    pre_dataset = pre_dataset.add_column("reference_rejected_logps", ref_rejected_logps)

    for j, (cj, rj) in enumerate(history_logps):
        pre_dataset = pre_dataset.add_column(f"history{j}_chosen_logps", cj.tolist())
        pre_dataset = pre_dataset.add_column(f"history{j}_rejected_logps", rj.tolist())

    # Save precompute results
    os.makedirs(args.precomputed_dir, exist_ok=True)
    pre_dataset.save_to_disk(args.precomputed_dir)

    # Setup training arguments
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=args.report_to,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type
    )

    # Create and run trainer
    trainer = TDPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=pre_dataset,
        ratio=args.ratio,
        eta=args.eta,
        beta=args.beta,
        data_collator=PrecomputeDataCollator(tokenizer)
    )

    trainer.train()

    # Save the model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()