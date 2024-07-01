# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig, HfArgumentParser, TrainingArguments
from trl import DPOTrainer

from .reward_model import DPORMTrainer, DPOPRMTrainer, NoRefDPOTrainer


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    seed: int = 0
    no_ref: bool = False
    rm_with_dpo: bool = True  ########
    alternate_step: Optional[int] = None
    loss_type: str = "sigmoid"  #########
    dpo_ratio: float = 0.2  ####
    default_save_steps: float = 0.32
    eval_steps: float = 0.025
    label_for_data: str = "total"  #################
    data_path: str = "hh"  #################
    data_ratio: float = 0.0
    scaling_factor: float = 10.0
    test: bool = False
    eos: bool = True
    not_save_model: bool = True  ####################
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    data_inverse: bool = False  #############################
    with_rm: bool = True
    # training parameters
    model_name_or_path: Optional[str] = field(default="/root/model/phi-2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=3200, metadata={"help": "max number of training steps"})
    output_dir: str = "/root/exp-modeling/output/checkpoint/DPO/{}"
    logging_dir: str = "/root/exp-modeling/tensorboard/DPO/{}"
    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
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
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    random: bool = False,


def get_suffix(p):
    idx = 1
    while Path(p + f"_Exp{idx}").exists():
        idx += 1
    return f"_Exp{idx}"


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


# def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
#     """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

#     The dataset is converted to a dictionary with the following structure:
#     {
#         'prompt': List[str],
#         'chosen': List[str],
#         'rejected': List[str],
#     }

#     Prompts should be structured as follows:
#       \n\nHuman: <prompt>\n\nAssistant:
#     Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
#     """
#     dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
#     if sanity_check:
#         dataset = dataset.select(range(min(len(dataset), 1000)))

#     def split_prompt_and_responses(sample) -> Dict[str, str]:
#         prompt = extract_anthropic_prompt(sample["chosen"])
#         return {
#             "prompt": prompt,
#             "chosen": sample["chosen"][len(prompt) :],
#             "rejected": sample["rejected"][len(prompt) :],
#         }

#     return dataset.map(split_prompt_and_responses)

def get_hh(inv=False, eos=True, label="helpful", seed=0, model='phi-2'):
    if label == "helpful":
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/helpful-train.json"}, split="train")
        eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/helpful-test.json"}, split="eval")
    if label == "harmless":
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/harmless-train.json"}, split="train")
        eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/harmless-test.json"}, split="eval")
    if label == "total":
        train_dataset = concatenate_datasets([
            load_dataset("json", data_files={"train": "/root/exp-modeling/data/helpful-train.json"}, split="train").map(lambda x: {"type": 0}),
            load_dataset("json", data_files={"train": "/root/exp-modeling/data/harmless-train.json"}, split="train").map(lambda x: {"type": 1})
        ])
        eval_dataset = concatenate_datasets([
            load_dataset("json", data_files={"eval": "/root/exp-modeling/data/helpful-test.json"}, split="eval").map(lambda x: {"type": 0}),
            load_dataset("json", data_files={"eval": "/root/exp-modeling/data/harmless-test.json"}, split="eval").map(lambda x: {"type": 1})
        ])
    if model == "phi-2":
        # suffix = "\n<|endoftext|>" if eos else "\n"
        suffix = ""
        reform = lambda examples: {
            "prompt": "Instruct: {}\nOutput:".format(examples["input"]),
            "chosen": ' ' + examples["win"] + suffix,
            "rejected": ' ' + examples["lose"] + suffix,
        }
    elif model == "mistral":
        suffix = ""
        reform = lambda examples: {
            "prompt": "[INST] {} [/INST]".format(examples["input"]),
            "chosen": examples["win"] + suffix,
            "rejected": examples["lose"] + suffix,
        }
    else:
        raise KeyError
    return train_dataset.map(reform).shuffle(seed=seed), eval_dataset.map(reform).shuffle(seed=seed)


def get_data_arena(inv=False, eos=True, seed=0, model="phi-2"):
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/chatbot_arena.json"}, split="train")
    tl = len(dataset)
    eval_dataset = dataset.select(range(int(0.05*tl)))
    train_dataset = dataset.select(range(int(0.05*tl), tl))
    suffix = "\n<|endoftext|>" if eos else "\n"
    if model == "phi-2":
        # suffix = "\n<|endoftext|>" if eos else "\n"
        suffix = ""
        reform = lambda examples: {
            "prompt": "Instruct: {}\nOutput:".format(examples["input"]),
            "chosen": ' ' + examples["win"] + suffix,
            "rejected": ' ' + examples["lose"] + suffix,
        }
    elif model == "mistral":
        suffix = ""
        reform = lambda examples: {
            "prompt": "[INST] {} [/INST]".format(examples["input"]),
            "chosen": examples["win"] + suffix,
            "rejected": examples["lose"] + suffix,
        }
    else:
        raise KeyError
    return train_dataset.map(reform).shuffle(seed=seed), eval_dataset.map(reform).shuffle(seed=seed)


def get_data_beaver(inv=False, eos=True, label="safe", ratio=None, seed=0, model="phi-2"):
    """
    {
        "prompt": ...
        "response_0": ...
        "response_1": ...
        "is_response_0_safe":false,
        "is_response_1_safe":false,
        "better_response_id":0,
        "safer_response_id":1
    }
    """
    KEY = {
        "helpful": "better_response_id",
        "helpful-small-dif": "better_response_id",
        "safe": "safer_response_id",
        "total-helpful-without-both-unsafe": "better_response_id"
    }[label]
    if label == "total-helpful-without-both-unsafe":
        # when one response is safe while the other is unsafe, better response is the safe one.
        if ratio is not None:
            train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/BeaverTails_train.jsonl"}, split="train")
            train_ss = train_dataset.filter(lambda x: x["is_response_0_safe"] and x["is_response_1_safe"]).select(range(int(ratio*40000)))
            train_su = train_dataset.filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]).select(range(40000))
            train_dataset = concatenate_datasets([train_ss, train_su]).shuffle(seed=seed)
            del train_ss
            del train_su
        else:
            # ratio ~ 2.2
            train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/BeaverTails_train.jsonl"}, split="train").filter(lambda x: x["is_response_0_safe"] or x["is_response_1_safe"])
        eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/BeaverTails_eval.jsonl"}, split="eval")
        eval_ss = eval_dataset.filter(lambda x: x["is_response_0_safe"] and x["is_response_1_safe"]).select(range(700))
        eval_su = eval_dataset.filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]).select(range(700))
        eval_uu = eval_dataset.filter(lambda x: not(x["is_response_0_safe"] or x["is_response_1_safe"])).select(range(300))
        eval_dataset = concatenate_datasets([eval_ss, eval_su, eval_uu]).shuffle(seed=seed)
    elif label == "total-helpful-with-both-unsafe":
        # there is some noise that both responses are unsafe.
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/BeaverTails_train.jsonl"}, split="train")
        eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/BeaverTails_eval.jsonl"}, split="eval")
        raise NotImplementedError
    elif label in ["safe", "helpful"]:
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/BeaverTails_train.jsonl"}, split="train").filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"])
        eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/BeaverTails_eval.jsonl"}, split="eval").filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"])
    elif label == "helpful-small-dif":
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/BeaverTails_train.jsonl"}, split="train").filter(lambda x: x["is_response_0_safe"] and x["is_response_1_safe"])
        eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/BeaverTails_eval.jsonl"}, split="eval").filter(lambda x: x["is_response_0_safe"] and x["is_response_1_safe"])

    if model == "phi-2":
        # suffix = "\n<|endoftext|>" if eos else "\n"
        suffix = ""
        reform = lambda examples: {
            "prompt": "Instruct: {}\nOutput:".format(examples["prompt"]),
            "chosen": ' ' + examples[f"response_{int(examples[KEY])}"] + suffix,
            "rejected": ' ' + examples[f"response_{1 - int(examples[KEY])}"] + suffix,
        }
    elif model == "mistral":
        suffix = ""
        reform = lambda examples: {
            "prompt": "[INST] {} [/INST]".format(examples["prompt"]),
            "chosen": examples[f"response_{int(examples[KEY])}"] + suffix,
            "rejected": examples[f"response_{1 - int(examples[KEY])}"] + suffix,
        }
    else:
        raise KeyError
    return train_dataset.map(reform).shuffle(seed=seed), eval_dataset.map(reform).shuffle(seed=0).select(range(1000))


def get_data_alpaca(ratio="total", eos=True, seed=0, model="phi-2", **kwargs):
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json"}, split="train")
    tl = len(dataset)
    eval_dataset = dataset.select(range(int(0.1*tl)))
    train_dataset = dataset.select(range(int(0.1*tl), tl))
    if ratio == "gt":
        train_dataset = train_dataset.filter(lambda x: x["consensus"] == True)
    elif ratio != "total":
        pass
    if model == "phi-2":
        # suffix = "\n<|endoftext|>" if eos else "\n"
        suffix = ""
        reform = lambda examples: {
            "prompt": "Instruct: {}\nOutput:".format(examples["input"]),
            "chosen": ' ' + examples["win"] + suffix,
            "rejected": ' ' + examples["lose"] + suffix,
            # "type": 1 if examples["consensus"] else 0
        }
    elif model == "mistral":
        suffix = ""
        reform = lambda examples: {
            "prompt": "[INST] {} [/INST]".format(examples["input"]),
            "chosen": examples["win"] + suffix,
            "rejected": examples["lose"] + suffix,
            # "type": 1 if examples["consensus"] else 0
        }
    else:
        raise KeyError
    return train_dataset.map(reform).shuffle(seed=seed), eval_dataset.map(reform).shuffle(seed=seed)


def get_data_compare_exp(name, seed=0):
    if name.startswith("beaver"):
        dpath = {
            "beaver_L_beaver": "/root/exp-modeling/data/beaver_CPL_88_labeled_train.json",
            "beaver_L_Bbaseline": "/root/exp-modeling/data/beaver_BASELINE_89_labeled_train.json",
            "beaver_L_Cbaseline": "/root/exp-modeling/data/beaver_L_BASELINE_150_chatbot_train.json",
            "beaver_L_chatbot": "/root/exp-modeling/data/beaver_L_CPL_149_chatbot_train.json",
            "beaver_L_total": "/root/exp-modeling/data/beaver_CPL_152_labeled_train.json",
            "beaver_L_Tbaseline": "/root/exp-modeling/data/beaver_BASELINE_151_labeled_train.json",
            "beaver_L_helpful": "/root/exp-modeling/data/beaver_CPL_153_labeled_train.json",
            "beaver_L_Hbaseline": "/root/exp-modeling/data/beaver_BASELINE_154_labeled_train.json",
        }[name]
        train_dataset = load_dataset("json", data_files={"train": dpath}, split="train")
        eval_dataset = load_dataset("json", data_files={"eval": dpath.replace("train", "eval")}, split="eval").select(range(1000))
    else:
        dpath = {
            "alpaca_L_beaver": "/root/exp-modeling/data/alpaca_CPL_88_labeled.json",
            "alpaca_L_Bbaseline": "/root/exp-modeling/data/alpaca_BASELINE_89_labeled.json",
            "alpaca_L_chatbot": "/root/exp-modeling/data/alpaca_CPL_149_labeled.json",
            "alpaca_L_Cbaseline": "/root/exp-modeling/data/alpaca_BASELINE_150_labeled.json",
            "alpaca_L_total": "/root/exp-modeling/data/alpaca_CPL_152_labeled.json",
            "alpaca_L_Tbaseline": "/root/exp-modeling/data/alpaca_BASELINE_151_labeled.json",
            "alpaca_L_helpful": "/root/exp-modeling/data/alpaca_CPL_153_labeled.json",
            "alpaca_L_Hbaseline": "/root/exp-modeling/data/alpaca_BASELINE_154_labeled.json",
            "chatbot_L_beaver": "/root/exp-modeling/data/chatbot_CPL_88_labeled.json",
            "chatbot_L_Bbaseline": "/root/exp-modeling/data/chatbot_BASELINE_89_labeled.json",
            "chatbot_L_chatbot": "/root/exp-modeling/data/chatbot_CPL_149_labeled.json",
            "chatbot_L_Cbaseline": "/root/exp-modeling/data/chatbot_BASELINE_150_labeled.json",
            "chatbot_L_total": "/root/exp-modeling/data/chatbot_CPL_152_labeled.json",
            "chatbot_L_Tbaseline": "/root/exp-modeling/data/chatbot_BASELINE_151_labeled.json",
            "chatbot_L_helpful": "/root/exp-modeling/data/chatbot_CPL_153_labeled.json",
            "chatbot_L_Hbaseline": "/root/exp-modeling/data/chatbot_BASELINE_154_labeled.json",
        }[name]
        dataset = load_dataset("json", data_files={"train": dpath}, split="train")

        tl = len(dataset)
        eval_dataset = dataset.select(range(int(0.1*tl)))
        train_dataset = dataset.select(range(int(0.1*tl), tl))

    suffix = '\n'
    reform = lambda examples: {
        "prompt": "Instruct: {}\nOutput:".format(examples["input"]),
        "chosen": ' ' + examples["win"] + suffix,
        "rejected": ' ' + examples["lose"] + suffix,
    }
    return train_dataset.map(reform).shuffle(seed=seed), eval_dataset.map(reform).shuffle(seed=seed)


def train_dpo():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if not script_args.sanity_check:
        if script_args.test:
            script_args.logging_dir = script_args.logging_dir.format("test")
            script_args.output_dir = script_args.output_dir.format("test")
        else:
            script_args.logging_dir = script_args.logging_dir.format("phi-2")
            script_args.output_dir = script_args.output_dir.format("phi-2")
            if script_args.no_ref:
                script_args.logging_dir = script_args.logging_dir.replace("DPO", "NO_REF")
                script_args.output_dir = script_args.output_dir.replace("DPO", "NO_REF")
            suffix = get_suffix(script_args.output_dir)
            script_args.logging_dir += suffix
            script_args.output_dir += suffix
            print("logging dir:", script_args.logging_dir)
            Path(script_args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(script_args.output_dir) / "config.txt", "w") as f:
                f.write(json.dumps(dataclasses.asdict(script_args), indent=4))

    # 1. load a pretrained model
    if script_args.rm_with_dpo:
        config = AutoConfig.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
        assert config.auto_map["AutoModel"] == "modeling_phi.PhiRMWithCausalLM"
        model = AutoModel.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
        if not script_args.random:
            torch.nn.init.constant_(model.rm_head.linear.weight, 0)
            torch.nn.init.constant_(model.rm_head.linear.bias, 0)
    else:
        model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Anthropic Helpful-Harmless dataset
    # train_dataset = get_hh("train", sanity_check=script_args.sanity_check)
    if "_L_" not in script_args.data_path:
        get_dataset = {
            "alpaca-phi-2": get_data,  # wrong data
            "chatbot-arena": get_data_arena, 
            "beaver": get_data_beaver,
            "hh": get_hh,
            "alpaca": get_data_alpaca
        }[script_args.data_path]
    if script_args.data_path == "beaver":
        train_dataset, eval_dataset = get_dataset(inv=script_args.data_inverse, eos=script_args.eos, label=script_args.label_for_data, ratio=script_args.data_ratio, seed=script_args.seed)
    elif script_args.data_path == "hh":
        train_dataset, eval_dataset = get_dataset(inv=script_args.data_inverse, eos=script_args.eos, label=script_args.label_for_data, seed=script_args.seed)
    elif script_args.data_path == "alpaca":
        train_dataset, eval_dataset = get_dataset(ratio=script_args.label_for_data, seed=script_args.seed)
    elif script_args.data_path in ["chatbot-arena", "alpaca-phi-2"]:
        train_dataset, eval_dataset = get_dataset(inv=script_args.data_inverse, eos=script_args.eos, seed=script_args.seed)
    else:
        train_dataset, eval_dataset = get_data_compare_exp(script_args.data_path, seed=script_args.seed)
    total_length = len(train_dataset)
    print("===")
    print(total_length)

    # 3. Load evaluation dataset
    # eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        logging_dir=script_args.logging_dir,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        optim="adamw_torch",
        # warmup_steps=150,
        adam_epsilon=1e-5,
        max_grad_norm=1.,
        report_to=script_args.report_to,
        bf16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
        save_steps=99999 if script_args.not_save_model else script_args.default_save_steps,
        # TODO: uncomment that on the next transformers release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )
    
    # 5. initialize the DPO trainer
    if script_args.no_ref:
        dpo_trainer = NoRefDPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            max_target_length=script_args.max_target_length,
            max_prompt_length=script_args.max_prompt_length,
            generate_during_eval=False,
            loss_type=script_args.loss_type
        )
        del model_ref
    else:
        if script_args.rm_with_dpo:
            dpo_trainer = DPORMTrainer(
                model,
                model_ref,
                args=training_args,
                beta=script_args.beta,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                max_length=script_args.max_length,
                max_target_length=script_args.max_target_length,
                max_prompt_length=script_args.max_prompt_length,
                generate_during_eval=False,
                dpo_ratio=script_args.dpo_ratio,
                with_rm=script_args.with_rm,
                loss_type=script_args.loss_type,
                alternate_step = script_args.alternate_step,
            )
        else:
            dpo_trainer = DPOTrainer(
                model,
                model_ref,
                args=training_args,
                beta=script_args.beta,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                max_length=script_args.max_length,
                max_target_length=script_args.max_target_length,
                max_prompt_length=script_args.max_prompt_length,
                generate_during_eval=False,
            )

    # 6. train
    dpo_trainer.train()


if __name__ == "__main__":
    train_dpo()
