import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import DPOConfig
from trainer import TDPOTrainer, PrecomputeDataCollator, TDPOTrainer_v2
import torch 

def parse_args():
    parser = argparse.ArgumentParser(description="TDPO training script")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--precomputed_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--ref_model", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1000)
    parser.add_argument("--mask_prompt", type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)


    if args.ref_model:
        ref_name = args.ref_model
    else:
        ref_name = args.base_model_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        )

    train_dataset = load_from_disk(args.precomputed_dir)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=args.report_to,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
    )

    data_collator = PrecomputeDataCollator(
    tokenizer,
    max_length=args.max_length,
    max_prompt_length=args.max_prompt_length,
    mask_prompt=args.mask_prompt,
    )

    trainer = TDPOTrainer_v2(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        ratio=args.ratio,
        eta=args.eta,
        beta=args.beta,
        data_collator=data_collator,
    )

    trainer.train()
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)
    print(f"✅ Training finished and model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
