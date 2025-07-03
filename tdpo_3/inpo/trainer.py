from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
# from peft import AutoPeftModelForCausalLM, LoraConfig
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from accelerate.utils import is_deepspeed_available, tqdm

@dataclass
class PreferenceDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    mask_prompt: Optional[bool] = False

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            # "inputs_ids", "attention_mask" both list
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            # False
            if self.mask_prompt:
                new_attention_mask = [0 for i, p in enumerate(prompt_tokens["attention_mask"])]
            else:
                new_attention_mask = [
                    0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
                ]
            # all 1 if i not in eos_indices_prompt
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt

            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            raise NotImplementedError

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        # 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask'
        # 'prompt', 'chosen', 'rejected', 'chosen_response_only', 'rejected_response_only'
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

@dataclass
class PrecomputeDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    mask_prompt: Optional[bool] = False

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        # 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask'
        # 'prompt', 'chosen', 'rejected', 'chosen_response_only', 'rejected_response_only'
        # 
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    to_pad = []
                    # Iterate through the batch with an index to know the location
                    for i, ex in enumerate(batch):

                        if ex[k][i] is None:
                            ex[k][i] = ex[k][i + 1]

                        try:
                            # This is the operation that might fail
                            tensor_data = torch.LongTensor(ex[k])
                            to_pad.append(tensor_data)
                        except TypeError:
                            # This block executes ONLY if the above line fails!
                            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            print(f"!!! DATA ERROR FOUND in collate function !!!")
                            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            print(f"Batch Index: The error is in the {i}-th sample of this batch.")
                            print(f"Problematic Key: '{k}'")
                            print(f"Problematic Value: The value is '{ex.get(k)}', which is causing the TypeError.")
                            print("\n--- Full content of the problematic sample ---")
                            print(ex)
                            print("------------------------------------------------\n")

                            # After printing, re-raise the exception to stop the program
                            raise

                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("logps"):
                padded_batch[k] = torch.FloatTensor([ex[k] for ex in batch])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # return collated batch
        return self.collate(features)

class MyPreferenceTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        ratio: float = 0,
        eta: float = 0.0075,
        loss_type: Literal["sigmoid", "hinge", "cross_entropy", "kl", "rev_kl", "raft"] = "rev_kl",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        mask_prompt: Optional[bool] = False,
        len_penalty: float = 0,
        max_history_t: Optional[int] = 2,
    ):

        if data_collator is None:
            # 2048, 1000, -100, 0, keep_end, None, False
            data_collator = PrecomputeDataCollator(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=False,
                max_target_length=max_target_length,
                mask_prompt=mask_prompt,
            )
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=eta,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            disable_dropout=disable_dropout
        )
        self.use_dpo_data_collator = True
        self.len_penalty = len_penalty
        self.ref_model = None
        self.denom = eta
        self.ratio = ratio
        print(self.ratio, self.denom)
        self.max_history_t = max_history_t

    def inpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        history_logps_list,
        reference_free: bool = False,
        margin: Optional[torch.FloatTensor] = None,
        len_penalty: float = 0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        t = self.max_history_t
        weighted_logratios = 0.0
        self.ratio = 1/6
        weights = [4/5, 1/5] # t-1, t-2 ....

        if history_logps_list and t > 0:
            effective_t = len(history_logps_list)

            if effective_t == 1:
                chosen_j, rejected_j = history_logps_list[0]
                weighted_logratios = chosen_j - rejected_j

            elif effective_t > 1:
                for j, (chosen_j, rejected_j) in enumerate(history_logps_list):
                    lambda_j = weights[j]
                    weighted_logratios += lambda_j * (chosen_j - rejected_j)


        # if reference_free:
        #     ref_logratios = 0

        logits = pi_logratios -  self.ratio * ref_logratios - (1 - self.ratio) * weighted_logratios
        losses = (logits - 1 / (2 * self.denom)) ** 2
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def pack_history_logps_from_dataset(self, inputs):
        history_logps_list = []
        for j in range(self.max_history_t):
            key_c = f"history{j}_chosen_logps"
            key_r = f"history{j}_rejected_logps"
            if key_c in inputs and key_r in inputs:
                history_logps_list.append((inputs[key_c], inputs[key_r]))
            else:
                break
        return history_logps_list

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        return self.get_batch_metrics(model, batch, train_eval)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.Tensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        
        reference_chosen_logps = batch['reference_chosen_logps'].to(self.accelerator.device)
        reference_rejected_logps = batch['reference_rejected_logps'].to(self.accelerator.device)
        history_logps_list = self.pack_history_logps_from_dataset(batch)
        history_logps_list = [
            (c.to(self.accelerator.device), r.to(self.accelerator.device))
            for c, r in history_logps_list
        ]

        if self.len_penalty > 0:
            chosen_len = batch["chosen_input_ids"].shape[1] * self.len_penalty
            rejected_len = batch["rejected_input_ids"].shape[1] * self.len_penalty
            len_penalty = chosen_len - rejected_len
        else:
            chosen_len = 1
            rejected_len = 1
            len_penalty = 0

        losses, chosen_rewards, rejected_rewards = self.inpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            history_logps_list,
            len_penalty=len_penalty,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
       
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

class PreComputer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "cross_entropy", "kl", "rev_kl", "raft"] = "rev_kl",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        mask_prompt: Optional[bool] = False,
        len_penalty: float = 0,
    ):

        if data_collator is None:
            # 2048, 1000, -100, 0, keep_end, None, False
            data_collator = PreferenceDataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=False,
                max_target_length=max_target_length,
                mask_prompt=mask_prompt,
            )
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
        )
        self.use_dpo_data_collator = True
        self.len_penalty = len_penalty
        self.precompute_ref_log_probs = True
        self._precomputed_train_ref_log_probs = True

    def precompute(self):
        dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

        reference_chosen_logps = []
        reference_rejected_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
            reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
            reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                (reference_chosen_logp, reference_rejected_logp)
            )
            reference_chosen_logps.append(reference_chosen_logp.cpu())
            reference_rejected_logps.append(reference_rejected_logp.cpu())

        all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
        all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

        print(all_reference_chosen_logps.shape, all_reference_rejected_logps.shape)
        return all_reference_chosen_logps, all_reference_rejected_logps


