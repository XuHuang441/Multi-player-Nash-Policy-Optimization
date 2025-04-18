from dataclasses import dataclass, field
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_from_disk, load_dataset, Dataset
from trl import DPOConfig
from trainer import TDPOTrainer, PreComputer, PrecomputeDataCollator


@dataclass
class ScriptArguments:
    base_model_path: str = field(default="meta-llama/Llama-3.2-1B")
    reference_model_path: str = field(default="sshleifer/tiny-gpt2")
    data_path: str = field(default="output/output_pref_0.json")
    precomputed_dir_root: str = field(default="./output/precomputed_dataset")
    output_dir_root: str = field(default="./model")
    num_rounds: int = field(default=5)
    max_history_t: int = field(default=2)
    batch_size: int = field(default=1)
    num_epochs: int = field(default=1)
    ratio: float = field(default=1.0)
    eta: float = field(default=0.01)
    beta: float = field(default=0.01)
    max_prompt_length: int = field(default=1024)
    max_length: int = field(default=2048)
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    report_to: Optional[str] = field(default="none")
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )

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

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    train_dataset = load_from_disk(script_args.train_dir)
    if script_args.sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 100)))

    tokenizer = AutoTokenizer.from_pretrained(script_args.reference_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    all_history_paths = []
    for round_idx in range(script_args.num_rounds):
        print(f"=== ROUND {round_idx} ===")

        model_path = script_args.base_model_path if round_idx == 0 else f"{script_args.output_dir_root}/step_{round_idx}"
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        history_model_paths = all_history_paths[-script_args.max_history_t:][::-1]

        train_dataset = prepare_data(script_args.data_path)
        reference_model = AutoModelForCausalLM.from_pretrained(script_args.reference_model_path,
                                                               torch_dtype=torch.bfloat16)

        pre = PreComputer(
            model=model,
            ref_model=reference_model,
            args=DPOConfig(output_dir="tmp", per_device_train_batch_size=script_args.batch_size),
            beta=script_args.beta,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
            len_penalty=0
        )

        ref_chosen_logps, ref_rejected_logps = pre.precompute()

        history_logps = precompute_multi_history(
            prompts_dataset=train_dataset,
            random_model=model,
            history_model_paths=history_model_paths,
            tokenizer=tokenizer,
            precomputer_class=PreComputer,
            training_args=DPOConfig(output_dir="tmp", per_device_train_batch_size=script_args.batch_size),
            beta=script_args.beta,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
            len_penalty=0,
        )

        pre_dataset = pre.train_dataset
        pre_dataset = pre_dataset.add_column("reference_chosen_logps", ref_chosen_logps)
        pre_dataset = pre_dataset.add_column("reference_rejected_logps", ref_rejected_logps)
        for j, (cj, rj) in enumerate(history_logps):
            pre_dataset = pre_dataset.add_column(f"history{j}_chosen_logps", cj.tolist())
            pre_dataset = pre_dataset.add_column(f"history{j}_rejected_logps", rj.tolist())

        precompute_out = f"{script_args.precomputed_dir_root}/round_{round_idx}"
        pre_dataset.save_to_disk(precompute_out)

        training_args = DPOConfig(
            output_dir=f"{script_args.output_dir_root}/step_{round_idx + 1}",
            per_device_train_batch_size=script_args.batch_size,
            num_train_epochs=script_args.num_epochs,
            gradient_accumulation_steps=4,
            remove_unused_columns=False,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to=script_args.report_to,
            learning_rate=script_args.learning_rate,
            lr_scheduler_type=script_args.lr_scheduler_type,
        )

        trainer = TDPOTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=pre_dataset,
            ratio=script_args.ratio,
            eta=script_args.eta,
            beta=script_args.beta,
            data_collator=PrecomputeDataCollator(tokenizer)
        )

        trainer.train()

        save_path = f"{script_args.output_dir_root}/step_{round_idx + 1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        all_history_paths.append(save_path)
