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
import dataclasses
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import accelerate
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import BitsAndBytesConfig, HfArgumentParser, AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, is_xpu_available
# from trl.core import LengthSampler


# input_min_text_length = 6
# input_max_text_length = 12
accelerator = accelerate.Accelerator()
device = accelerator.device
warnings.filterwarnings('ignore')

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    phi: bool = False 
    model_name: Optional[str] = field(default="/root/model/Mistral-7B-v02", metadata={"help": "the model name"})
    dataset: Optional[str] = field(default="helpful", metadata={"help": "the dataset name"})
    baseline: bool = False
    rm_adapter: Optional[str] = field(
        default="/root/exp-modeling/output/checkpoint/HAF-lora/mistral_Exp{}/checkpoint-{}", metadata={"help": "the rm adapter name"}
    )
    checkpoint: Optional[str] = field(default=None, metadata={"help": "{exp}-{checkpoint}"})
    # exp: int = 45
    # checkpoint: int = 2160
    mini_bs: int = 1
    gradient_accumulation_steps: int = 2
    lr: float = 1e-6
    bs: int = 2
    max_new_tokens: int = 128
    eval_num: int = 40
    eval_step: Optional[int] = None
    total_episodes: int = 30000
    # max_length: int = 512
    output_dir: str = "/root/exp-modeling/output/checkpoint/RLHF/{}"
    logging_dir: str = "/root/exp-modeling/tensorboard/RLHF/{}"
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"})
    use_safetensors: Optional[bool] = field(default=True, metadata={"help": "Use safetensors"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=True, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=True, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    test: bool = False
    inference: bool = False
    score_clip: Optional[float] = field(default=3.0, metadata={"help": "Score clipping"})


class PPOFullTrainer(PPOTrainer):
    def __init__(self, *args, eval_dataset=None, test_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataloader = self.prepare_dataloader(eval_dataset, kwargs.get("data_collator", None))
        self.test_dataloader = self.prepare_dataloader(test_dataset, kwargs.get("data_collator", None))
        self.eval_dataloader, self.test_dataloader = self.accelerator.prepare(self.eval_dataloader, self.test_dataloader)


def get_suffix(p):
    idx = 1
    while Path(p + f"_Exp{idx}").exists():
        idx += 1
    return f"_Exp{idx}"


def get_suffix_tb(p):
    idx = 1
    while Path(p + f"_{idx}").exists():
        idx += 1
    return f"_{idx}"


def rm_tree(pth: Path):
    for f in os.listdir(pth):
        child = pth / f
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


# def create_and_prepare_dataset(tokenizer):
#     dataset = load_dataset(script_args.dataset_name, split="train[:1%]")

#     input_size = LengthSampler(input_min_text_length, input_max_text_length)

#     def tokenize(example):
#         text_size = input_size()
#         example["input_ids"] = tokenizer(example["chosen"])[:text_size]
#         example["query"] = tokenizer.decode(example["input_ids"])
#         return example

#     dataset = dataset.map(tokenize, batched=False)
#     dataset.set_format("torch")
#     return dataset


def get_data_helpful(tokenizer):
    def reform(example):
        example["input_ids"] = tokenizer.encode('[INST] ' + example["input"] + ' [/INST]')
        example["query"] = tokenizer.decode(example["input_ids"])
        return example
    
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/helpful-train.json"}, split="train").map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384)
    test_dataset = load_dataset("json", data_files={"test": "/root/exp-modeling/data/helpful-test.json"}, split="test").map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384).select(range(500))
    eval_dataset = dataset.select(range(64))
    train_dataset = dataset.select(range(64, len(dataset)))

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")
    return train_dataset.shuffle(), eval_dataset, test_dataset


def get_data_harmless(tokenizer):
    def reform(example):
        example["input_ids"] = tokenizer.encode('[INST] ' + example["input"] + ' [/INST]')
        example["query"] = tokenizer.decode(example["input_ids"])
        return example
    
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/harmless-train.json"}, split="train").map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384)
    test_dataset = load_dataset("json", data_files={"test": "/root/exp-modeling/data/harmless-test.json"}, split="test").map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384).select(range(500))
    eval_dataset = dataset.select(range(64))
    train_dataset = dataset.select(range(64, len(dataset)))

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")
    return train_dataset.shuffle(), eval_dataset, test_dataset


def get_data_chatbot(tokenizer):
    def reform(example):
        example["input_ids"] = tokenizer.encode('[INST] ' + example["input"] + ' [/INST]')
        example["query"] = tokenizer.decode(example["input_ids"])
        return example
    
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/chatbot_arena.json"}, split="train").map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384)
    tl = len(dataset)
    test_dataset = dataset.select(range(0, 500))
    eval_dataset = dataset.select(range(500, 564))
    train_dataset = dataset.select(range(564, tl))
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")
    return train_dataset.shuffle(), eval_dataset, test_dataset


def get_data_beaver(tokenizer):
    def reform(example):
        example["input_ids"] = tokenizer.encode('[INST] ' + example["prompt"] + ' [/INST]')
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/BeaverTails_train.jsonl"}, split="train").filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]).map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384)
    eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/BeaverTails_eval.jsonl"}, split="eval").filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]).map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384)
    test_dataset = dataset.select(range(0, 500))
    eval_dataset = eval_dataset.select(range(64))
    train_dataset = dataset.select(range(500, len(dataset)))
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")
    return train_dataset.shuffle(), eval_dataset, test_dataset


def get_data_alpaca(tokenizer):
    def reform(example):
        example["input_ids"] = tokenizer.encode('[INST] ' + example["input"] + ' [/INST]')
        example["query"] = tokenizer.decode(example["input_ids"])
        return example
    
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json"}, split="train").map(reform, batched=False).filter(lambda x: len(x["input_ids"]) < 384)
    tl = len(dataset)
    # dataset = dataset.map(reform, batched=False).filter(lambda x: len(x["input_ids"])<384)
    test_dataset = dataset.select(range(0, 500))
    eval_dataset = dataset.select(range(500, 564))
    train_dataset = dataset.select(range(564, tl))
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")
    return train_dataset.shuffle(), eval_dataset, test_dataset


def get_data_all(tokenizer):
    train_dataset, eval_dataset, test_dataset = [], [], []
    for func in [get_data_alpaca, get_data_beaver, get_data_chatbot, get_data_harmless, get_data_helpful]:
        tr, ev, te = func(tokenizer)
        train_dataset.append(tr)
        eval_dataset.append(ev.select(range(32)))
        test_dataset.append(te)
    train_dataset = concatenate_datasets(train_dataset)
    eval_dataset = concatenate_datasets(eval_dataset)
    test_dataset = concatenate_datasets(test_dataset)
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")
    return train_dataset.shuffle(), eval_dataset, test_dataset


MAPPING = {
    "helpful": {"func": get_data_helpful, "mistral": (45, 2160), "baseline": (51, 2000)},
    "harmless": {"func": get_data_harmless, "mistral": (65, 1040), "baseline": (64, 1840)},
    "chatbot": {"func": get_data_chatbot, "mistral": (52, 1440), "baseline": (53, 1360)},
    "beaver": {"func": get_data_beaver, "mistral": (56, 2240), "baseline": (57, 2320)},
    "alpaca": {"func": get_data_alpaca, "mistral": (49, 2160), "baseline": (50, 1600)},
    "all": {"func": get_data_all, "mistral": (73, 3900), "baseline": (74, 4200)}
}


def inference(script_args, model, tokenizer, ds):
    generation_kwargs = {
        "top_p": 0.8,
        "temperature": 0.5,
        "do_sample": True,
        "max_new_tokens": script_args.max_new_tokens,
        # "max_length": script_args.max_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    for item in ds:
        result = model.generate(item["input_ids"].unsqueeze(0), **generation_kwargs)[0]
        print(result)
        print(tokenizer.decode(result))
        raise NotImplementedError


def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    script_args.rm_adapter = script_args.rm_adapter.format(*MAPPING[script_args.dataset]["baseline" if script_args.baseline else "mistral"])

    m_name = "phi-2" if script_args.phi else "mistral" 
    if script_args.phi:
        script_args.model_name = "/root/model/phi-2"
        script_args.rm_adapter = script_args.rm_adapter.replace("mistral", "phi-2")
        
    
    if script_args.test:
        script_args.logging_dir = script_args.logging_dir.format("test")
        script_args.output_dir = script_args.output_dir.format("test")
    else:
        if accelerator.is_main_process:
            script_args.logging_dir = script_args.logging_dir.format(m_name)
            script_args.output_dir = script_args.output_dir.format(m_name)
            if script_args.checkpoint:
                suffix = f"_Exp{script_args.checkpoint.split('-')[0]}"
                script_args.logging_dir += suffix
                suffix_tb = get_suffix_tb(script_args.logging_dir)
                script_args.logging_dir += suffix_tb
                config_name = "config" + suffix_tb + ".txt"
            else:
                suffix = get_suffix(script_args.output_dir)
                script_args.logging_dir += suffix
                config_name = "config.txt"
            script_args.output_dir += suffix
            print("logging dir:", script_args.logging_dir)
            Path(script_args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(script_args.output_dir) / config_name, "w") as f:
                f.write(json.dumps(dataclasses.asdict(script_args), indent=4))

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        # bias="none",
        task_type="CAUSAL_LM",
        # target_modules= r".*mlp\.fc.*" if script_args.phi else r".*mlp\..*proj"
    )
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    if not (Path(script_args.rm_adapter) / "adapter_model.bin").exists():
    # if True:
        param = load_file(Path(script_args.rm_adapter.format()) / "adapter_model.safetensors")
        del param["base_model.model.score.bias"]
        torch.save(param, Path(script_args.rm_adapter) / "adapter_model.bin")

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name,
        device_map={"": "xpu:0"} if is_xpu_available() else {"": device},
        peft_config=lora_config,
        quantization_config=nf4_config,
        reward_adapter=script_args.rm_adapter,
        use_safetensors=script_args.use_safetensors,
        # trust_remote_code=True,
    )
    
    if script_args.checkpoint:
        Eid, Cid = script_args.checkpoint.split('-')
        param = load_file(f"/root/exp-modeling/output/checkpoint/RLHF/mistral_Exp{Eid}/{'best' if Cid=='best' else 'checkpoint-'+Cid}/adapter_model.safetensors")
        # for n, v in model.named_parameters():
        #     print(n, end="  ")
        # print('='*40)
        # for n in param.keys():
        #     print(n, end='  ')
        model.load_state_dict({'pretrained_model.'+k.replace(".weight", ".default.weight"): v for k,v in param.items()}, strict=False)

    # model._no_split_modules += ["base_model.model.score"]
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset = create_and_prepare_dataset(tokenizer)
    dataset, eval_dataset, test_dataset = MAPPING[script_args.dataset]["func"](tokenizer)

    if script_args.inference:
        inference(script_args, model, tokenizer, test_dataset)
        return

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])


    config = PPOConfig(
        model_name=script_args.model_name,
        log_with=script_args.log_with,
        learning_rate=script_args.lr,
        batch_size=script_args.bs,
        mini_batch_size=script_args.mini_bs,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        seed=script_args.seed,
        use_score_scaling=script_args.use_score_scaling,
        use_score_norm=script_args.use_score_norm,
        score_clip=script_args.score_clip,
        project_kwargs={"logging_dir": script_args.logging_dir,}
    )

    ppo_trainer = PPOFullTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=collator,
    )

    script_args.eval_step = int(script_args.total_episodes / script_args.bs / ppo_trainer.accelerator.num_processes / script_args.eval_num)
    print("eval step:", script_args.eval_step)

    generation_kwargs = {
        "top_p": 0.8,
        "temperature": 0.5,
        "length_penalty": 1.3,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "max_new_tokens": script_args.max_new_tokens,
        # "max_length": script_args.max_length,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    saved = False
    best_eval_rewards = float("-inf")
    best_path = Path(script_args.output_dir) / "best"
    current_step = 0
    if script_args.checkpoint:
        print("load from existing checkpoint")
        Cid = script_args.checkpoint.split('-')[1]
        assert Cid != "best"
        if accelerator.is_main_process and best_path.exists():
            rm_tree(best_path)
        current_step = int(Cid)
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.current_step = current_step

    total_step = int(script_args.total_episodes / ppo_trainer.accelerator.num_processes / script_args.bs)
    while ppo_trainer.current_step < total_step:
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
            print(current_step)
            question_tensors = batch["input_ids"]
            if ppo_trainer.current_step >= total_step:
                break
            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                batch_size=2,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            print(batch["response"])
            # Compute reward score  
            # should not start with '<s> '!!
            texts = [q[4:] + r for q, r in zip(batch["query"], batch["response"])]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.model.device)
            raw_rewards = ppo_trainer.model.compute_reward_score(**inputs)
            # print(raw_rewards)
            # rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token
            rewards = [raw_rewards[i, -1, 0] for i in range(len(raw_rewards))]

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            torch.cuda.empty_cache()
            current_step += 1

            # eval
            # [add KL and other metrics]
            if current_step % script_args.eval_step == 0:
                import torch.distributed as dist
                if ppo_trainer.is_distributed:
                    dist.barrier()
                pbar = tqdm(ppo_trainer.eval_dataloader, total=len(ppo_trainer.eval_dataloader))
                all_rewards = []
                all_texts = []
                for ev_batch in pbar:
                    ev_q_tensors = ev_batch["input_ids"]
                    with torch.no_grad():
                        ev_response_tensors = ppo_trainer.generate(ev_q_tensors, return_prompt=False, batch_size=2, **generation_kwargs)
                        ev_batch["response"] = tokenizer.batch_decode(ev_response_tensors, skip_special_tokens=True)
                        ev_texts = [q[4:] + r for q, r in zip(ev_batch["query"], ev_batch["response"])]
                        all_texts.extend(ev_texts)
                        ev_inputs = tokenizer(ev_texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.model.device)
                        ev_raw_rewards = ppo_trainer.model.compute_reward_score(**ev_inputs)
                        ev_rewards = [ev_raw_rewards[i, -1, 0].float() for i in range(len(ev_raw_rewards))]
                        all_rewards.extend(ev_rewards)
                    if accelerator.is_main_process:
                        pbar.set_description(f"reward: {all_rewards[-1].item():.3f}")
                all_rewards = torch.tensor(all_rewards).to(ppo_trainer.current_device)
                if ppo_trainer.is_distributed:
                    dist.barrier()
                    dist.all_reduce(all_rewards, op=dist.ReduceOp.SUM)
                    # all_output_text = [None for i in range(ppo_trainer.accelerator.num_processes)]
                    # dist.all_gather_object(all_output_text, all_texts)
                if accelerator.is_main_process:
                    logs = {}
                    all_rewards = all_rewards.mean().cpu().item() / ppo_trainer.accelerator.num_processes
                    logs["env/eval_reward_mean"] = all_rewards
                    ppo_trainer.accelerator.log(logs, step=current_step)
                    save_path = Path(script_args.output_dir) / f"checkpoint-{current_step}"
                    if save_path.exists():
                        rm_tree(save_path)
                    ppo_trainer.save_pretrained(save_path)
                    with open(save_path / "result_half.json", 'w') as f:
                        json.dump(all_texts, f, indent=4)
                    if not saved:
                        ppo_trainer.save_pretrained(best_path)
                        saved = True
                    elif all_rewards >= best_eval_rewards:
                        best_eval_rewards = all_rewards
                        rm_tree(best_path)
                        ppo_trainer.save_pretrained(best_path)
                    

                

    ###########################
    # save models!!
    # metrics


if __name__ == "__main__":
    train()