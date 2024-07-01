from dataclasses import dataclass, field
from typing import Optional

from trl import RewardConfig


@dataclass
class MoRMConfig:
    method: str = "softmax"
    cls_loss_ratio: float = 0.8
    num_expert: int = 2
    pretrain_ratio: float = 0.2
    em_step: Optional[int] = None
    em_ratio: Optional[float] = None
    tau: float = 10.0
    gamma: float = 0.999


@dataclass
class DispatchConfig:
    cuda_list: Optional[str] = None
    memory: Optional[str] = None


@dataclass
class ScriptArguments:
    use_checkpoint: bool = True             ################################
    checkpoint_path: str = "/root/exp-modeling/output/checkpoint/DPO/phi-2_Exp{}/checkpoint-{}/"
    exp_id: int = 2                         ####################
    checkpoint_step: int = 1000
    set_pad_to_eos_token: bool = True            ####################
    dataset_name: str = "alpaca-human-0-sum-unique"   ############################
    max_dataset_length: int = 99999                 ##########################
    output_dir: str = "/root/exp-modeling/model/RM/{}"
    model_name: str = "/root/model/phi-2"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    save_steps: int = 999999
    not_save_model: bool = True             ###########################
    bias: bool = False                      ############################
    dispatch: DispatchConfig = field(
        default_factory=lambda: DispatchConfig()
    )
    MoRM: bool = False
    use_em: bool = False
    
    morm_config: MoRMConfig = field(
        default_factory=lambda: MoRMConfig()
    )
    local_rank: int = 0  # for deepspeed
    test: bool = False
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir="output/checkpoint/{}",
            per_device_train_batch_size=1,
            num_train_epochs=3,             #############################
            evaluation_strategy="steps",
            gradient_accumulation_steps=16, #############################
            gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1e-5,             #############################
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_dir="/root/exp-modeling/tensorboard/RM/{}",
            adam_epsilon=1e-5,  # 1e-8 is too small and will lead to "nan"
            logging_steps=2,
            max_length=512,
            max_grad_norm=1.,
            seed=42,
            save_steps=99999, # not saving in this way 
            fp16=False,
            eval_steps=0.02
        )
    )