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
    TrainingArguments, Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer, DPOConfig
from accelerate.utils import is_deepspeed_available, tqdm
import sys

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
            elif k.endswith("logps"):
                padded_batch[k] = torch.FloatTensor([ex[k] for ex in batch])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # truncation
        for ex in features:
            # truncate prompt
            if self.max_prompt_length is not None:
                ids = ex["prompt_input_ids"]
                if len(ids) > self.max_prompt_length:
                    if self.truncation_mode == "keep_end":
                        ex["prompt_input_ids"] = ids[-self.max_prompt_length :]
                    else:
                        ex["prompt_input_ids"] = ids[: self.max_prompt_length]
            # truncate chosen/rejected
            if self.max_length is not None:
                for key in ("chosen_input_ids", "rejected_input_ids"):
                    ids = ex[key]
                    if len(ids) > self.max_length:
                        if self.truncation_mode == "keep_end":
                            ex[key] = ids[-self.max_length :]
                        else:
                            ex[key] = ids[: self.max_length]
        batch = self.collate(features)
        return batch


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
        output_dir: str = None,
        precompute_ref_log_probs: bool = False
    ):
        args.precompute_ref_log_probs = precompute_ref_log_probs
        args.beta=eta
        args.loss_type = loss_type,
        args.label_pad_token_id = label_pad_token_id,
        args.padding_value = padding_value,
        args.truncation_mode = truncation_mode,
        args.max_length = max_length,
        args.max_prompt_length = max_prompt_length,
        args.max_target_length = max_target_length,
        args.is_encoder_decoder = is_encoder_decoder,
        args.disable_dropout = disable_dropout,
        args.generate_during_eval = generate_during_eval,
        args.output_dir = output_dir

        if data_collator is None:
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
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )

        self.use_dpo_data_collator = True
        self.len_penalty = len_penalty
        self.ref_model = None
        self.denom = eta
        self.ratio = ratio
        print(self.ratio, self.denom)
        print(f"[DEBUG] precompute_ref_log_probs = {args.precompute_ref_log_probs}")


    def inpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        last_chosen_logps: torch.FloatTensor,
        last_rejected_logps: torch.FloatTensor,
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
        last_logratios = last_chosen_logps - last_rejected_logps
        
        # if reference_free:
        #     ref_logratios = 0

        logits = pi_logratios -  self.ratio * ref_logratios - (1 - self.ratio) * last_logratios
        losses = (logits - 1 / (2 * self.denom)) ** 2
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

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
        last_chosen_logps = batch['last_chosen_logps'].to(self.accelerator.device)
        last_rejected_logps = batch['last_rejected_logps'].to(self.accelerator.device)
        
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
            last_chosen_logps,
            last_rejected_logps,
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

        args.beta=beta
        args.loss_type=loss_type
        args.label_pad_token_id=label_pad_token_id
        args.padding_value=padding_value
        args.truncation_mode=truncation_mode
        args.max_length=max_length
        args.max_prompt_length=max_prompt_length
        args.max_target_length=max_target_length
        args.is_encoder_decoder=is_encoder_decoder
        args.disable_dropout=disable_dropout
        args.generate_during_eval=generate_during_eval

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )
        self.use_dpo_data_collator = True
        self.len_penalty = len_penalty
        self.precompute_ref_log_probs = True
        self._precomputed_train_ref_log_probs = True

    
    def set_ref_model(self, new_ref_model: Union[PreTrainedModel, nn.Module]):
        """Update the reference model and rewrap the Trainer's internal state."""
        self.ref_model = new_ref_model.to(self.accelerator.device)
        self.model_wrapped = None  # Force rebuild
        self._wrap_model(self.model, training=True)  # Re-wrap with new ref model if needed

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

            # reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
            reference_chosen_logp, reference_rejected_logp = self.compute_ref_log_probs(padded_batch)
            
            reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                (reference_chosen_logp, reference_rejected_logp)
            )
            reference_chosen_logps.append(reference_chosen_logp.cpu())
            reference_rejected_logps.append(reference_rejected_logp.cpu())

        all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
        all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

        print(all_reference_chosen_logps.shape, all_reference_rejected_logps.shape)
        return all_reference_chosen_logps, all_reference_rejected_logps


class INPOTrainer(Trainer):
    def __init__(self, tokenizer, ratio=1.0, eta=0.01, len_penalty=0.0, beta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.denom = eta
        self.beta = beta
        self.len_penalty = len_penalty
        self.tokenizer=tokenizer

    def inpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            last_chosen_logps: torch.FloatTensor,
            last_rejected_logps: torch.FloatTensor,
            len_penalty: float = 0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        last_logratios = last_chosen_logps - last_rejected_logps

        logits = pi_logratios - self.ratio * ref_logratios - (1 - self.ratio) * last_logratios
        losses = (logits - 1 / (2 * self.denom)) ** 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, batch):
        """
        Use the model to forward propagate the chosen and rejected samples separately and compute logp.
        Note: Need to adapt to different model outputs.
        """

        def get_logps(input_ids):
            # construct attention_mask dynamically
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits  # (B, T, V)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = shift_labels.unsqueeze(-1)
            token_logps = torch.gather(log_probs, dim=-1, index=shift_labels).squeeze(-1)

            mask = attention_mask[..., 1:]  # align with log_probs
            sentence_logp = (token_logps * mask).sum(dim=-1)

            return sentence_logp, logits

        chosen_logps, chosen_logits = get_logps(batch["chosen_input_ids"])
        rejected_logps, rejected_logits = get_logps(batch["rejected_input_ids"])
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = \
            self.concatenated_forward(model, inputs)

        reference_chosen_logps = inputs["reference_chosen_logps"].to(policy_chosen_logps.device)
        reference_rejected_logps = inputs["reference_rejected_logps"].to(policy_chosen_logps.device)
        last_chosen_logps = inputs["last_chosen_logps"].to(policy_chosen_logps.device)
        last_rejected_logps = inputs["last_rejected_logps"].to(policy_chosen_logps.device)

        losses, chosen_rewards, rejected_rewards = self.inpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            last_chosen_logps,
            last_rejected_logps,
            len_penalty=self.len_penalty
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # Logging metrics
        self.log({
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
            "rewards/accuracies": reward_accuracies.mean().item(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            "logps/rejected": policy_rejected_logps.mean().item(),
            "logps/chosen": policy_chosen_logps.mean().item(),
        })

        return (losses.mean(), policy_chosen_logits) if return_outputs else losses.mean()

class INPOTrainer_v2(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        ratio: float = 0,
        eta: float = 0.0075,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
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
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        len_penalty: float = 0,
    ):
        # Initialize with standard DPO params, but we'll override the loss function
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )
        # INPO specific parameters
        self.ratio = ratio
        self.denom = eta
        self.len_penalty = len_penalty
        print(f"INPO parameters: ratio={self.ratio}, denom={self.denom}, len_penalty={self.len_penalty}")

    def concatenated_forward(self, model, batch):
        """
        Use the model to forward propagate the chosen and rejected samples separately and compute logp.
        Note: Need to adapt to different model outputs.
        """

        def get_logps(input_ids):
            # construct attention_mask dynamically
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            if hasattr(model, "encoder"):  # encoder-decoder model
                decoder_input_ids = input_ids
                output = model(
                    input_ids=None,  # encoder-decoder model, so input_ids is None
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=attention_mask,
                )
            else:  # causal language model
                output = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = output.logits  # (B, T, V)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = shift_labels.unsqueeze(-1)
            token_logps = torch.gather(log_probs, dim=-1, index=shift_labels).squeeze(-1)

            mask = attention_mask[..., 1:]  # align with log_probs
            sentence_logp = (token_logps * mask).sum(dim=-1)

            return sentence_logp, logits

        chosen_logps, chosen_logits = get_logps(batch["chosen_input_ids"])
        rejected_logps, rejected_logits = get_logps(batch["rejected_input_ids"])
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits


    def inpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        last_chosen_logps: torch.FloatTensor,
        last_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        margin: Optional[torch.FloatTensor] = None,
        len_penalty: float = 0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the INPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses.
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses.
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses.
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses.
            last_chosen_logps: Log probabilities of the previous model for the chosen responses.
            last_rejected_logps: Log probabilities of the previous model for the rejected responses.
            reference_free: If True, we ignore the reference model.
            margin: Optional margin for the loss.
            len_penalty: Length penalty for the responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        last_logratios = last_chosen_logps - last_rejected_logps
        
        if reference_free:
            ref_logratios = 0

        # Apply length penalty if specified
        if len_penalty > 0:
            # Implementation would depend on how you want to incorporate length penalty
            pass

        # INPO loss calculation
        logits = pi_logratios - self.ratio * ref_logratios - (1 - self.ratio) * last_logratios
        losses = (logits - 1 / (2 * self.denom)) ** 2
        
        # Compute rewards for logging
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

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
        """Compute the INPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        
        # Get log probabilities from the policy model
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        
        # Get the precomputed log probabilities
        reference_chosen_logps = batch['reference_chosen_logps'].to(self.accelerator.device)
        reference_rejected_logps = batch['reference_rejected_logps'].to(self.accelerator.device)
        last_chosen_logps = batch['last_chosen_logps'].to(self.accelerator.device)
        last_rejected_logps = batch['last_rejected_logps'].to(self.accelerator.device)
        
        # Apply length penalty if needed
        if self.len_penalty > 0:
            chosen_len = batch["chosen_input_ids"].shape[1] * self.len_penalty
            rejected_len = batch["rejected_input_ids"].shape[1] * self.len_penalty
            len_penalty = chosen_len - rejected_len
        else:
            len_penalty = 0

        # Compute the INPO loss
        losses, chosen_rewards, rejected_rewards = self.inpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            last_chosen_logps,
            last_rejected_logps,
            len_penalty=len_penalty,
        )
        
        # Calculate accuracy metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
       
        # Prepare metrics for logging
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

class TDPOTrainer(Trainer):
    def __init__(self, tokenizer, ratio=1.0, eta=0.01, len_penalty=0.0, beta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.denom = eta
        self.beta = beta
        self.len_penalty = len_penalty
        self.tokenizer = tokenizer

    def inpo_loss_extended(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        history_logps_list,  # List of (chosen_logps_j, rejected_logps_j)
        t: int
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        """
        n = 0 without history term, fall back to DPO
        n = 1 use logratios directly, no weighting required
        n > 1 Use $\lambda_j$ formula for weighting
        """
        n = len(history_logps_list)
        if n == 1:
            weighted_logratios = history_logps_list[0][0] - history_logps_list[0][1]
        elif n > 1:
            weighted_logratios = 0.0
            for j, (chosen_j, rejected_j) in enumerate(history_logps_list):
                time_j = t - (n - 1 - j)
                lambda_j = 2 * (t - time_j) / ((2 * t - n + 1) * (n - 1))
                weighted_logratios += lambda_j * (chosen_j - rejected_j)
        else:
            weighted_logratios = 0.0  # not using weighted_logratios without history term

        logits = pi_logratios - self.ratio * ref_logratios - (1 - self.ratio) * weighted_logratios
        losses = (logits - 1 / (2 * self.denom)) ** 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, batch):
        def get_logps(input_ids):
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = shift_labels.unsqueeze(-1)
            token_logps = torch.gather(log_probs, dim=-1, index=shift_labels).squeeze(-1)
            mask = attention_mask[..., 1:]
            sentence_logp = (token_logps * mask).sum(dim=-1)

            return sentence_logp, logits

        chosen_logps, chosen_logits = get_logps(batch["chosen_input_ids"])
        rejected_logps, rejected_logits = get_logps(batch["rejected_input_ids"])
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits


    def pack_history_logps_from_dataset(self, inputs, max_history_t):
        history_logps_list = []
        for j in range(max_history_t):
            key_c = f"history{j}_chosen_logps"
            key_r = f"history{j}_rejected_logps"
            if key_c in inputs and key_r in inputs:
                history_logps_list.append((inputs[key_c], inputs[key_r]))
            else:
                # no more histories, stop early
                break
        return history_logps_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        max_history_t = 2  # t-2
        history_logps_list = self.pack_history_logps_from_dataset(inputs, max_history_t)
        t = len(history_logps_list)

        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = \
            self.concatenated_forward(model, inputs)

        reference_chosen_logps = inputs["reference_chosen_logps"].to(policy_chosen_logps.device)
        reference_rejected_logps = inputs["reference_rejected_logps"].to(policy_chosen_logps.device)

        # Move history tensors to device
        history_logps_list = [
            (chosen.to(policy_chosen_logps.device), rejected.to(policy_chosen_logps.device))
            for chosen, rejected in history_logps_list
        ]

        losses, chosen_rewards, rejected_rewards = self.inpo_loss_extended(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            history_logps_list,
            t
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        self.log({
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
            "rewards/accuracies": reward_accuracies.mean().item(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            "logps/rejected": policy_rejected_logps.mean().item(),
            "logps/chosen": policy_chosen_logps.mean().item(),
        })

        return (losses.mean(), policy_chosen_logits) if return_outputs else losses.mean()
