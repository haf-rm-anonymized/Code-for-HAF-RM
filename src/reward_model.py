# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import json
import math
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
# from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from trl import is_peft_available
from trl import RewardConfig
from trl import DPOTrainer
from trl.trainer.utils import PeftSavingCallback, RewardDataCollatorWithPadding, DPODataCollatorWithPadding, compute_accuracy

from .args import MoRMConfig


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


NO_SPLIT_MODULE_MAPPING = {
    "chatglm": ["GLMBlock"],
    "phi-2": ["CodeGenBlock"],
}


def load_model(mpath, cuda_list="0,1,2", memory="30GiB", host_low_loading=False):
    cuda_list = cuda_list.split(',')
    no_split_module_classes = []
    for k, v in NO_SPLIT_MODULE_MAPPING.items():
        if k in mpath:
            no_split_module_classes.extend(v)
    max_memory = {int(cuda): memory for cuda in cuda_list}
    if host_low_loading:
        max_memory[int(cuda_list[0])] = "1GiB"
    config = AutoConfig.from_pretrained(mpath, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes)
    load_checkpoint_in_model(model, mpath, device_map=device_map)
    model = dispatch_model(model,device_map=device_map)
    return model


def fetch(key, dic, default):
    if key in dic:
        return dic.pop(key)
    else:
        return default


class PairwiseTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        save_steps: int = 99999,
        output_dir: str = ""
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self.count += 1
        if self.count % self.save_steps == 0:
            torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")

        mu_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
        mu_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"])

        # 0.69315 : E(RM(w) - RM(l)) = 0
        reward_loss = -nn.functional.logsigmoid(mu_w - mu_l).mean()
        self.log({"loss": reward_loss.item(), "chosen": mu_w.mean().item(), "rejected": mu_l.mean().item()})
        if return_outputs:
            return reward_loss, {"rewards_chosen": mu_w,
                "rewards_rejected": mu_l,
            }
        # pdb.set_trace()
        return reward_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        predict = predict["rewards_chosen"] - predict["rewards_rejected"]
        if "consensus" in inputs:
            label = inputs["consensus"]
        else:
            label = torch.ones_like(predict)
        return loss, predict, label
    

class Collator(DPODataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)
        result = self.collate(tokenized_batch)
        # if "is_response_0_safe" in features[0] and "type" not in features[0]:
        #     result["type"] = [int(features[0]["is_response_0_safe"]) + int(features[0]["is_response_1_safe"])]
        if "type" in features[0]:
            result["type"] = [features[0]["type"]]
        return result


class SLiCTrainer(DPOTrainer):
    def __init__(self, *args, loss_type="sigmoid", **kwargs):
        datacollator = Collator(
            kwargs["tokenizer"],
            max_length=kwargs["max_length"],
            max_prompt_length=kwargs["max_prompt_length"],
            label_pad_token_id=kwargs.get("label_pad_token_id", -100),
            padding_value=kwargs.get("padding_value", 0),
            truncation_mode=kwargs.get("truncation_mode", "keep_end"),
            is_encoder_decoder=False,
            max_target_length=kwargs["max_target_length"],
        )
        super().__init__(*args, data_collator=datacollator, **kwargs)
        self.loss_type = loss_type

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        return (chosen_logps, rejected_logps)
    
    def reward_loss(self, chosen_rewards, rejected_rewards):
        return -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
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

        logits = pi_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "IPO":
            losses = (logits - 1 / self.beta / 2) ** 2
        elif self.loss_type == "IPO-sqrt":
            losses = self.beta * (logits - 1 / self.beta / 2) ** 2
        elif self.loss_type == "IPO-full":
            losses = (self.beta * logits - 0.5) ** 2
        elif self.loss_type == "SLiC":
            losses = -policy_chosen_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'IPO']")
        return losses
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
        ) = self.concatenated_forward(model, batch)

        losses = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )
        losses = losses.mean()
        prob_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}logps/accuracies"] = prob_accuracies.cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/variance/rejected"] = metrics[f"{prefix}logps/rejected"]
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/variance/chosen"] = metrics[f"{prefix}logps/chosen"]
        metrics[f"{prefix}logps/margin"] = (policy_chosen_logps - policy_rejected_logps).detach().cpu().mean()
        metrics[f"{prefix}prob"] = F.sigmoid(policy_chosen_logps - policy_rejected_logps).detach().cpu().mean()
        metrics[f"{prefix}min/prob"] = metrics[f"{prefix}prob"]
        metrics[f"{prefix}max/prob"] = metrics[f"{prefix}prob"]
        metrics[f"{prefix}variance/prob"] = metrics[f"{prefix}prob"]
        metrics[f"{prefix}loss/prob"] = losses.cpu()
        return losses, metrics
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            if "min" in key:
                logs[key] = torch.tensor(metrics).min().item()
            elif "max" in key:
                logs[key] = torch.tensor(metrics).max().item()
            elif "variance" in key:
                logs[key] = torch.tensor(metrics).var().item()
            else:
                logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)


class DPORMTrainer(DPOTrainer):
    def __init__(self, *args, dpo_ratio=0.2, rm_ratio=1.0, loss_type="sigmoid", alternate_step=None, with_rm=True, eval_log_save_path="", **kwargs):
        self.dpo_ratio = dpo_ratio
        self.rm_ratio = rm_ratio
        print("RM_RATIO", self.rm_ratio)
        datacollator = Collator(
            kwargs["tokenizer"],
            max_length=kwargs["max_length"],
            max_prompt_length=kwargs["max_prompt_length"],
            label_pad_token_id=kwargs.get("label_pad_token_id", -100),
            padding_value=kwargs.get("padding_value", 0),
            truncation_mode=kwargs.get("truncation_mode", "keep_end"),
            is_encoder_decoder=False,
            max_target_length=kwargs["max_target_length"],
        )
        super().__init__(*args, data_collator=datacollator, **kwargs)
        self.loss_type = loss_type
        self.alternate_step = alternate_step
        if eval_log_save_path:
            self.save_eval_result = True
        else:
            self.save_eval_result = False
        self.eval_log_save_path=eval_log_save_path
        if with_rm == False:
            raise KeyError
        # self.rm_ratio = 1 if with_rm else 0
        # if isinstance(self.model, PeftModel):
        #     self.accelerator.unwrap_model(self.model).base_model.model.score.requires_grad_(True)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], ref=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        if ref:
            all_logits = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                return_dict=True,
                output_hidden_states=True,
            ).logits.to(torch.float32)
            chosen_rewards = None
            rejected_rewards = None
        else:
            rewards, all_logits = model.calculate(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                return_dict=True,
                output_hidden_states=True,
            )
            rewards = rewards.to(torch.float32)
            chosen_rewards = rewards[:len_chosen]
            rejected_rewards = rewards[len_chosen:]
            all_logits = all_logits.logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False if self.loss_type!="SimPO" else True,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_rewards, rejected_rewards)
    
    def reward_loss(self, chosen_rewards, rejected_rewards):
        return -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
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

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "IPO":
            losses = (logits - 1 / self.beta / 2) ** 2
        elif self.loss_type == "IPO-sqrt":
            losses = self.beta * (logits - 1 / self.beta / 2) ** 2
        elif self.loss_type == "IPO-full":
            losses = (self.beta * logits - 0.5) ** 2
        elif self.loss_type == "SLiC":
            losses = -policy_chosen_logps
        elif self.loss_type == "SimPO":
            losses = -F.logsigmoid(self.beta * pi_logratios)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'IPO']")

        if self.loss_type == "SimPO":
            chosen_rewards = self.beta * policy_chosen_logps.detach()
            rejected_rewards = self.beta * policy_rejected_logps.detach()
        else:
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            true_chosen_rewards,
            true_rejected_rewards,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                # if ref model and policy model are the same, there is also a reward in the output.
                # self.accelerator.unwrap_model(self.model).disable_adapters()
                # (
                #     reference_chosen_logps,
                #     reference_rejected_logps,
                #     _,
                #     _,
                #     _,
                #     _,
                # ) = self.concatenated_forward(self.model, batch)
                # self.accelerator.unwrap_model(self.model).enable_adapters()
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch, ref=True)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        losses = self.dpo_ratio * losses.mean()
        reward_loss = self.reward_loss(true_chosen_rewards, true_rejected_rewards) * self.rm_ratio
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        true_reward_accuracies = (true_chosen_rewards > true_rejected_rewards).float()
        prob_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}logps/accuracies"] = prob_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/variance/rejected"] = metrics[f"{prefix}logps/rejected"]
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/variance/chosen"] = metrics[f"{prefix}logps/chosen"]
        # metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}rewards/true_margins"] = (true_chosen_rewards - true_rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/variance/true_margins"] = metrics[f"{prefix}rewards/true_margins"]
        if "type" in batch:
            # eval and test are not shuffled
            metrics[f"{prefix}rewards/true_accuracies_t{batch['type'][0]}"] = true_reward_accuracies.cpu().mean()
            metrics[f"{prefix}rewards/accuracies_t{batch['type'][0]}"] = reward_accuracies.cpu().mean()
            metrics[f"{prefix}logps/accuracies_t{batch['type'][0]}"] = prob_accuracies.cpu().mean()
            metrics[f"{prefix}rewards/true_accuracies_total"] = true_reward_accuracies.cpu().mean()
            metrics[f"{prefix}rewards/accuracies_total"] = reward_accuracies.cpu().mean()
            metrics[f"{prefix}logps/accuracies_total"] = prob_accuracies.cpu().mean()
        else:
            metrics[f"{prefix}rewards/true_accuracies"] = true_reward_accuracies.cpu().mean()
        metrics[f"{prefix}loss/reward"] = reward_loss.cpu()
        metrics[f"{prefix}loss/prob"] = losses.cpu()

        if self.alternate_step is not None:
            if (self.state.global_step // self.alternate_step) % 2:
                losses *= 0
                metrics[f"{prefix}loss_state"] = 0.
            else:
                reward_loss *= 0
                metrics[f"{prefix}loss_state"] = 1.
        # return self.rm_ratio * losses + rewatio * rd_loss, metrics
        return losses + reward_loss, metrics
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            if "min" in key:
                logs[key] = torch.tensor(metrics).min().item()
            elif "max" in key:
                logs[key] = torch.tensor(metrics).max().item()
            elif "variance" in key:
                logs[key] = torch.tensor(metrics).var().item()
            else:
                logs[key] = torch.tensor(metrics).mean().item()
        if f"rewards/true_margins" in logs and f"rewards/variance/true_margins" in logs:
            logs[f"variance/normed_margins"] = logs[f"rewards/variance/true_margins"] / (logs[f"rewards/true_margins"]**2 + 1e-5)
        if train_eval == "eval" and self.save_eval_result:
            with open(self.eval_log_save_path, "a") as f:
                json.dump(logs, f, indent=4)
                f.write('\n')
        del self._stored_metrics[train_eval]
        return super().log(logs)
      

class NoRefDPOTrainer(DPOTrainer):
    def __init__(self, *args, loss_type="sigmoid", **kwargs):
        super().__init__(*args, **kwargs)
        del self.ref_model
        self.loss_type = loss_type

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logits = policy_chosen_logps - policy_rejected_logps

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "IFT":
            losses = -policy_chosen_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
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

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prob_accuracies = (policy_chosen_logps > policy_rejected_logps).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}logps/accuracies"] = prob_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics
    

class TokenizeMixin:
    TEMPLATE = {
        "ziya": "Human:{}\n\nAssistant:{}",
        "phi2": "Instruct: {}\nOutput: {}\n<|endoftext|>",
    }

    def __init__(self, model):
        func_mapping = defaultdict(lambda: 'bert')
        func_mapping.update({
            "ziya": "ziya",
        })

        self.func = eval(f'self.tokenize_{func_mapping[model.lower()]}')
        if model.lower() in self.TEMPLATE:
            self.template = self.TEMPLATE[model.lower()]

    def tokenize_bert(self, prompt, chosen, rejected):
        chosen_result = self.tokenizer(
            prompt, chosen,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        rejected_result = self.tokenizer(
            prompt, rejected,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return chosen_result, rejected_result
    
    def tokenize_ziya(self, prompt, chosen, rejected):
        chosen_result = self.tokenizer(
            self.template.format(prompt, chosen),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        rejected_result = self.tokenizer(
            self.template.format(prompt, rejected),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return chosen_result, rejected_result

    def tokenize_phi2(self, prompt, chosen, rejected):
        chosen_result = self.tokenizer(
            self.template.format(prompt, chosen),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        rejected_result = self.tokenizer(
            self.template.format(prompt, rejected),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return chosen_result, rejected_result


    def __init__(self, *args, reward_loss_ratio=0.008, kl_reg_ratio=0.05, soft_eps=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_eps = 1e-8
        self.kl_reg_ratio = kl_reg_ratio
        self.reward_loss_ratio = reward_loss_ratio
        self.soft_eps = soft_eps

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # print(batch)
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_input_ids"].shape[0]
        # print(concatenated_batch)
        all_rewards, all_logvar, _ = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        )
        
        all_rewards = all_rewards.squeeze()
        all_logvar = all_logvar.squeeze()

        chosen_rewards = all_rewards[:len_chosen]
        rejected_rewards = all_rewards[len_chosen:]
        chosen_logvar = all_logvar[:len_chosen]
        rejected_logvar = all_logvar[len_chosen:]

        return (chosen_rewards, chosen_logvar, rejected_rewards, rejected_logvar)
    
    def prob_loss(self, mu_w, logvar_w, mu_l, logvar_l):
        """
        \mathcal{r}(y)-\mathcal{r}(y') ~ N(mu_w-mu_l, \exp(logvar_w)+\exp(logvar_l))
        win prob = \Phi((mu_l - mu_w) / (\exp(logvar_w)+\exp(logvar_l)))
        still, use cross entropy
        """
        X = (mu_w - mu_l) * (((logvar_w).exp()+(logvar_l).exp()).sqrt() + self.var_eps).reciprocal() / math.sqrt(2)
        prob = 0.5 * (1 + torch.erf(X))
        return -((1-self.soft_eps)*(prob.log()) + self.soft_eps*((1-prob).log())).mean(), prob, X
    
    def reward_loss(self, chosen_rewards, rejected_rewards):
        # logistic prob loss
        dif = chosen_rewards - rejected_rewards
        return -nn.functional.logsigmoid(dif).mean(), nn.functional.sigmoid(dif).mean()

    def KL_regularization_loss(self, mu_w, logvar_w, mu_l, logvar_l):
        mu = torch.concat([mu_w, mu_l], dim=0)
        logvar = torch.concat([logvar_w, logvar_l], dim=0)
        return 0.5*(mu**2 + logvar.exp() - 1 - logvar).mean()
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            chosen_rewards,
            chosen_logvar,
            rejected_rewards,
            rejected_logvar,
        ) = self.concatenated_forward(model, batch)

        Ploss, prob, _ = self.prob_loss(chosen_rewards, chosen_logvar, rejected_rewards, rejected_logvar)
        Rloss, Rprob = self.reward_loss(chosen_rewards, rejected_rewards)
        KLloss = self.KL_regularization_loss(chosen_rewards, chosen_logvar, rejected_rewards, rejected_logvar)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}probabilistic_loss"] = Ploss.detach().cpu()
        metrics[f"{prefix}kl_reg"] = KLloss.detach().cpu()
        metrics[f"{prefix}reward_loss"] = Rloss.detach().cpu()
        metrics[f"{prefix}correct_prob"] = prob.detach().cpu().mean()  
        metrics[f"{prefix}logistic/correct_prob"] = Rprob.detach().cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/variance/margins"] = metrics[f"{prefix}rewards/margins"]
        metrics[f"{prefix}rewards/logvar"] = ((chosen_logvar.detach()+rejected_logvar.detach())/2).cpu().mean()
        metrics[f"{prefix}rewards/variance/logvar"] = chosen_logvar.tolist() + rejected_logvar.tolist()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        if self.reward_loss_ratio >= 0:
            return Ploss + self.reward_loss_ratio*Rloss + self.kl_reg_ratio*KLloss, metrics
        return Rloss + self.kl_reg_ratio*KLloss, metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            if type(value) != torch.Tensor:
                self._stored_metrics[train_eval][key].extend(value)
            else:
                self._stored_metrics[train_eval][key].append(value)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            if "min" in key:
                logs[key] = torch.tensor(metrics).min().item()
            elif "max" in key:
                logs[key] = torch.tensor(metrics).max().item()
            elif "variance" in key:
                logs[key] = torch.tensor(metrics).var().item()
            else:
                logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)