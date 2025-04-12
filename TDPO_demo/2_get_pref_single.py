import torch
import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer

# ========== 手动配置部分 ==========
DATASET_PATH = "output/output_0.json"  # 上一步生成的文件
OUTPUT_PATH = "output/output_pref_0.json"  # 偏好结果输出路径
PREFERENCE_MODEL_PATH = "unsloth/Llama-3.2-1B-Instruct"
K = 2
USE_TOURNAMENT = False
SANITY_CHECK = False
TEMPERATURE = 1.5

# ========== 加载模型与tokenizer ==========
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=PREFERENCE_MODEL_PATH,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)

# 第二个 tokenizer_plain 仅用于 context 渲染（chat_template 不加 generation prompt）
tokenizer_plain = AutoTokenizer.from_pretrained(PREFERENCE_MODEL_PATH, use_fast=True)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
token_id_A = tokenizer.encode("A", add_special_tokens=False)[0]
token_id_B = tokenizer.encode("B", add_special_tokens=False)[0]

model.eval()
device = model.device


# ========== 偏好判断函数 ==========
def get_pref(context, responses):
    probs_chosen = []
    for chosen_position in [0, 1]:
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [{"role": "user", "content": prompt}]

        prompt_encoded = tokenizer.apply_chat_template(message, tokenize=False).replace(tokenizer.bos_token, "")
        input_ids = tokenizer.encode(prompt_encoded, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            output = model(input_ids)
        logit_A = output.logits[0, -1, token_id_A].item()
        logit_B = output.logits[0, -1, token_id_B].item()
        Z = np.exp(logit_A / TEMPERATURE) + np.exp(logit_B / TEMPERATURE)
        logit_chosen = [logit_A, logit_B][chosen_position]
        prob_chosen = np.exp(logit_chosen / TEMPERATURE) / Z
        probs_chosen.append(prob_chosen)

    avg_prob_chosen = np.mean(probs_chosen)
    correct = 0.5 if avg_prob_chosen == 0.5 else float(avg_prob_chosen > 0.5)
    return correct


def get_match_res(context, responses, id_0, id_1):
    response_pair = [responses[id_0], responses[id_1]]
    chosen_A = get_pref(context, response_pair)
    return (id_0, id_1) if chosen_A >= 0.5 else (id_1, id_0)


# ========== 加载数据 ==========
ds = load_dataset("json", data_files=DATASET_PATH, split="train", field="instances")

if SANITY_CHECK:
    ds = ds.select(range(min(len(ds), 100)))

# ========== 遍历样本 & 构造偏好 ==========
results = []

with torch.no_grad():
    for sample in tqdm(ds):
        responses = sample["responses"]
        if len(responses) < K:
            continue

        context = ""  # 默认空上下文
        if "context_messages" in sample:
            context_messages = sample["context_messages"]
            context = tokenizer_plain.apply_chat_template(context_messages, tokenize=False)

        if len(responses) == 2:
            win_id, lose_id = 0, 1
        elif len(responses) == 8 and USE_TOURNAMENT:
            win_1, lose_1 = get_match_res(context, responses, 0, 1)
            win_2, lose_2 = get_match_res(context, responses, 2, 3)
            win_3, lose_3 = get_match_res(context, responses, 4, 5)
            win_4, lose_4 = get_match_res(context, responses, 6, 7)
            win_5, _ = get_match_res(context, responses, win_1, win_2)
            win_6, _ = get_match_res(context, responses, win_3, win_4)
            win_id, _ = get_match_res(context, responses, win_5, win_6)
            _, lose_5 = get_match_res(context, responses, lose_1, lose_2)
            _, lose_6 = get_match_res(context, responses, lose_3, lose_4)
            _, lose_id = get_match_res(context, responses, lose_5, lose_6)
        else:
            win_id = 0
            for i in range(1, len(responses)):
                chosen_A = get_pref(context, [responses[win_id], responses[i]])
                if chosen_A < 0.5:
                    win_id = i
            lose_id = 0 if win_id != 0 else 1
            for i in range(len(responses)):
                if i in (win_id, lose_id):
                    continue
                chosen_A = get_pref(context, [responses[lose_id], responses[i]])
                if chosen_A >= 0.5:
                    lose_id = i

        if win_id == lose_id:
            continue

        response_pair = [responses[win_id], responses[lose_id]]
        chosen_A = get_pref(context, response_pair)
        if len(responses) > 2 and chosen_A <= 0.5:
            continue

        flag = 'A' if chosen_A > 0.5 else 'B' if chosen_A < 0.5 else 'tie'
        results.append({
            "prompt": sample["prompt"],
            "responses": response_pair,
            "chosen": flag
        })

# ========== 保存偏好结果 ==========
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"instances": results}, f, ensure_ascii=False, indent=2)

print(f"[✓] Done. {len(results)} preference samples saved to {OUTPUT_PATH}")
