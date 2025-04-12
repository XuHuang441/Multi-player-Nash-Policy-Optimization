# === 多轮 TDPO 训练主循环 === #
import sys
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset, Dataset
from trl import DPOConfig
from trainer import TDPOTrainer, PreComputer, PrecomputeDataCollator
import os

# ==== 配置项 ==== #
base_model_path = "meta-llama/Llama-3.2-1B"               # 当前策略初始点
reference_model_path = "meta-llama/Llama-3.2-1B"            # 固定参考策略
output_dir_root = "./model"
precomputed_dir_root = "./output/precomputed_dataset"
data_path = "output/output_pref_0.json"
num_rounds = 5
max_history_t = 2

tokenizer = AutoTokenizer.from_pretrained(reference_model_path)
tokenizer.pad_token = tokenizer.eos_token

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
    从多个历史策略模型路径中分别计算 log-probs，返回历史 logprobs 列表

    返回：
    - history_logps_list: List of (chosen_logps, rejected_logps) tensors
    """
    history_logps_list = []

    for step_idx, model_path in enumerate(history_model_paths):
        print(f"[History Step {step_idx}] Loading model from: {model_path}")
        history_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 构造 PreComputer，每次用相同 random_model，变 ref_model 为历史策略
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


# ==== 外部训练循环 ==== #
all_history_paths = []
for round_idx in range(num_rounds):
    print(f"=== ROUND {round_idx} ===")

    model_path = base_model_path if round_idx == 0 else f"model/step_{round_idx}"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # 构造历史策略路径（最多保留 max_history_t 个）
    history_model_paths = all_history_paths[-max_history_t:][::-1]  # 最近的靠前

    # === Precompute === #
    train_dataset = prepare_data(data_path)
    reference_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.bfloat16)

    # Precompute reference

    pre = PreComputer(
        model=model,
        ref_model=reference_model,
        args=DPOConfig(output_dir="tmp", per_device_train_batch_size=1),
        beta=0.01,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
        len_penalty=0
    )

    ref_chosen_logps, ref_rejected_logps = pre.precompute()

    # Precompute history
    history_logps = precompute_multi_history(
        prompts_dataset=train_dataset,
        random_model=model,
        history_model_paths=history_model_paths,
        tokenizer=tokenizer,
        precomputer_class=PreComputer,
        training_args=DPOConfig(output_dir="tmp", per_device_train_batch_size=1),
        beta=0.01,
        max_prompt_length=1024,
        max_length=2048,
        len_penalty=0,
    )
    pre_dataset = pre.train_dataset

    pre_dataset = pre_dataset.add_column("reference_chosen_logps", ref_chosen_logps)
    pre_dataset = pre_dataset.add_column("reference_rejected_logps", ref_rejected_logps)

    for j, (cj, rj) in enumerate(history_logps):
        pre_dataset = pre_dataset.add_column(f"history{j}_chosen_logps", cj.tolist())
        pre_dataset = pre_dataset.add_column(f"history{j}_rejected_logps", rj.tolist())

    # 保存预计算数据
    precompute_out = f"{precomputed_dir_root}/round_{round_idx}"
    pre_dataset.save_to_disk(precompute_out)

    # === Train === #
    training_args = DPOConfig(
        output_dir=f"{output_dir_root}/step_{round_idx+1}",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        gradient_accumulation_steps=4,
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )

    trainer = TDPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=pre_dataset,
        ratio=1.0,
        eta=0.01,
        beta=0.01,
        data_collator=PrecomputeDataCollator(tokenizer)
    )

    trainer.train()

    # 保存当前策略模型
    save_path = f"model/step_{round_idx + 1}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 更新历史路径列表
    all_history_paths.append(save_path)