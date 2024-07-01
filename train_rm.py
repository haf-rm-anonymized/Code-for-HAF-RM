import argparse
import json
import random


UNREACHABLE_LIST = ["unreachable", "test_mode", "param", "get_args"]
ARGUMENTS = []


class Arg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def param(*arguments):
    def decorator(func):
        def F(*args, **kwargs):
            global ARGUMENTS
            ARGUMENTS = arguments
            return func(*args, **kwargs)
        return F
    return decorator


def unreachable(func):
    UNREACHABLE_LIST.append(func.__name__)
    return func


def test_mode(func):
    def F():
        print("Test start.")
        func()
        print("Test passed!")
    return F


def get_args():
    parser = argparse.ArgumentParser()
    for arg in ARGUMENTS:
        parser.add_argument(*arg.args, **arg.kwargs)
    args = parser.parse_args()
    return args

##########################################################################################################

@param(
    Arg("--inverse-rate", "-i", type=float, default=0.0),
    Arg("--calculate_method", "-c", type=str, choices=["mean", "sum"], default="mean")
)
def generate_data_for_training():
    # get logit average
    args = get_args()
    inverse_ratio = args.inverse_rate
    logit_avg = []
    with open("qa_status/phi-2_alpaca_human_pref_.jsonl") as f:
        for line in f:
            item = json.loads(line)
            w = item["r"]
            l = item["l"]
            if args.calculate_method == "mean":
                logit_avg.append((sum(w)/len(w), sum(l)/len(l)))
            else:
                logit_avg.append((sum(w), sum(l)))

    # get QA data
    with open("/root/dataset/alpaca_farm/alpaca_human_preference.json") as f:
        qa_data = json.load(f)
    assert len(logit_avg) == len(qa_data)

    # combine them
    # don't forget that logit_avg = [(win logit, lose logit),...], not [(1,2),...]
    final_data = []
    for (wp, lp), item in zip(logit_avg, qa_data):
        q = item["instruction"] + '\n' + item["input"]
        wa = item[f'output_{item["preference"]}']
        la = item[f'output_{3-item["preference"]}']
        flag = 0
        if random.random() < inverse_ratio:
            wa, la = la, wa
            flag += 1
        if wp >= lp:
            final_data.append({
                "input": q,
                "win": wa,
                "lose": la,
                "consensus": flag%2 == 0,
            })
        else:
            final_data.append({
                "input": q,
                "win": la,
                "lose": wa,
                "consensus": flag%2 == 1
            })
    if args.calculate_method == "mean":
        with open(f"data/phi_2-alpaca_human_pref-I{int(100*inverse_ratio)}.json", "w") as f:
            json.dump(final_data, f, indent=4)
    else:
        with open(f"data/phi_2-alpaca_human_pref-I{int(100*inverse_ratio)}-sum.json", "w") as f:
            json.dump(final_data, f, indent=4)


def dpo():
    from src.train_dpo_tmp import train_dpo
    train_dpo()

def evaluate():
    from src.eval_haf import train_dpo
    train_dpo()


def lora():
    from src.train_dpo_lora import train_dpo
    train_dpo()


def rlhf():
    from src.train_ppo import train
    train()


def zero_init():
    from src.train_zero_init import train_dpo
    train_dpo()


def all_data():
    from src.train_all import train_dpo
    train_dpo()


def all_lora():
    from src.train_all_lora import train_dpo
    train_dpo()


def dpo_and_gen_long():
    from pathlib import Path
    from src.train_dpo_tmp import train_dpo
    from src.sample_result import sample_sample_single

    def get_suffix(p):
        idx = 1
        while Path(p + f"_Exp{idx}").exists():
            idx += 1
        return idx

    basedir = "/root/exp-modeling/output/checkpoint/NO_REF/phi-2"
    idx = get_suffix(basedir)
    print(f"test EXP{idx}")
    STEP = 12480
    train_dpo()
    sample_sample_single(exp=idx, step=STEP)


@param(Arg("--exp", "-e", type=int), Arg("--step", "-s", type=int, default=1440), Arg("--sample", action="store_true"))
def gen():
    from src.sample_result import sample_beam_single, sample_sample_single
    args = get_args()
    if args.sample:
        sample_sample_single(exp=args.exp, step=args.step)
    else:
        sample_beam_single(exp=args.exp, step=args.step)


@param(
    Arg("--logdir", "-l", type=str, default=""), 
    Arg("--dpo", action="store_true")
)
def look_up():
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorboard.backend.event_processing import event_accumulator
    args = get_args()
    base = "tensorboard/DPO/" if args.dpo else "tensorboard/RM/" 
    logdir = base + args.logdir
    result = glob.glob(logdir)
    print(glob.glob(logdir))
    flag = "y"
    if len(result) > 1:
        flag = input("output?")
    if flag.lower() in ["y", "yes"]:
        Converge = None
        Agree = None
        Disagree = None
        for res in result:
            f = glob.glob(res+"/*")
            name = res.rsplit('/', 1)[1]
            ea = event_accumulator.EventAccumulator(f[0])
            ea.Reload()
            step, acc = [i for i in zip(*[(i.step, i.value) for i in ea.scalars.Items('eval/acc')])]
            agreed_acc = [i.value for i in ea.scalars.Items('eval/agreed_acc')]
            disagreed_acc = [i.value for i in ea.scalars.Items('eval/disagreed_acc')]
            if Converge is None:
                Converge = np.array(acc)
                Agree = np.array(agreed_acc)
                Disagree = np.array(disagreed_acc)
            else:
                Converge += np.array(acc)
                Agree = np.array(agreed_acc)
                Disagree = np.array(disagreed_acc)
            plt.figure()
            plt.plot(step, acc, color="#00F5FF", label="total")
            plt.plot(step, agreed_acc, color="#00FF00", label="agree")
            plt.plot(step, disagreed_acc, color="#FF4500", label="disagree")
            plt.legend()
            plt.savefig("img/"+name+".jpg")
        if Converge is not None:
            L = len(Converge) // 2
            Converge /= len(result)
            Agree /= len(result)
            Disagree /= len(result)
            print("min max converge agree disagree")
            print(np.min(Converge), np.max(Converge), end=" ")
            tmp1 = np.mean(Converge[L:])
            tmp2 = np.mean(Converge[L//2:])
            tmp3 = np.mean(Converge[L//6:])
            print(tmp1, tmp2, end=" ")
            print(np.mean(Agree[L//2:]), np.mean(Disagree[L//2:]))


# detect_influence_dpo_and_rm
@param(
    Arg("--logdir", "-l", type=str, default=""),
    Arg("--dif-eps", "-d", type=float, default=0.2)
)
def detect():
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorboard.backend.event_processing import event_accumulator
    args = get_args()
    base = "tensorboard/DPO/"
    logdir = base + args.logdir
    result = glob.glob(logdir)
    print(glob.glob(logdir))

    for res in result:
        f = glob.glob(res+"/*")
        ea = event_accumulator.EventAccumulator(f[0])
        ea.Reload()
        alternate_steps = min([i.step for i in ea.scalars.Items('eval/loss_state') if i.value == 0])
        eval_steps = ea.scalars.Items('eval/loss_state')[0].step
        assert alternate_steps % eval_steps == 0
        change_step = alternate_steps // eval_steps
        abnormal = []
        for k in [
            'eval/loss/prob',
            'eval/loss/reward', 
            "eval/rewards/true_accuracies_total", 
            "eval/rewards/true_accuracies_t0",
            "eval/rewards/true_accuracies_t1",  
            "eval/rewards/true_margin",
            "eval/rewards/variance/true_margins",
        ]:
            try:
                loss = [i.value for i in ea.scalars.Items(k)]
                loss = [loss[0]] + loss + (change_step - len(loss) % change_step) * [loss[-1]]
                s = np.array(loss)
                dif = (s[1:] - s[:-1])
                for ab_idx in abnormal:
                    dif[ab_idx] = 0
                if 'loss' in k:
                    ab_idx = np.abs(dif) > args.dif_eps
                    if ab_idx.any():
                        dif[ab_idx] = 0
                        abnormal.append(ab_idx)
                dif = dif.reshape((-1, change_step))
                print('==', k, '==')
                print("no reward loss:", np.sum(dif[::2]))
                print("no DPO loss: ", np.sum(dif[1::2]))
            except Exception as e:
                print('=== no key:', k, '===')
        

#########################################################################################
#   TEST

@test_mode
def test_dist_model_loc():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mpath = "/root/model/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    for n, v in model.named_parameters():
        print(n, v.device)



@test_mode
@param(
    Arg("--model_path", "-m", type=str, default="/root/exp-modeling/output/checkpoint/DPO/phi-2_Exp2/checkpoint-1000")
)
def test_chat():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    args = get_args()
    mpath = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    p = "Instruct: Rewrite the given statement in the negative form\nHe always smiles\nOutput:"
    q = tokenizer([p], return_tensors="pt").to(model.device)
    output = model.generate(**q, return_dict_in_generate=True, max_length=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=50256, do_sample=False, num_beams=4)
    output = tokenizer.batch_decode(output.sequences)[0]
    print(output)


@test_mode
@param(
    Arg("--init-from-config", "-l", action="store_true")
)
def test_pretrained_rm():
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    from src.modeling_phi import PhiRMWithCaulsalLM
    init_from_config = get_args().init_from_config
    mpath = "/root/model/phi-2"
    if init_from_config:
        config = AutoConfig.from_pretrained(mpath, trust_remote_code=True)
        # model = PhiRMWithCaulsalLM(config)
        # just initialized, pretrained params are not loaded!
        config.auto_map["AutoModel"] = "modeling_phi.PhiPRMWithCausalLM"
        model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
        print(model.device)
        p = "Instruct: Rewrite the given statement in the negative form\nHe always smiles\nOutput:"
        q = tokenizer([p], return_tensors="pt").to(model.device)
        output = model.generate(**q, return_dict_in_generate=True, max_length=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=50256, do_sample=True, repetition_penalty=1.2)
        output = tokenizer.batch_decode(output.sequences)[0]
        print(output)
        print('=')
    else:
        tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        model = PhiRMWithCaulsalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="cuda")
        # model = AutoModel.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda", empty_init=False)
        for n, v in model.named_parameters():
            print(n)
        print(model.device)
        if model.device == "cpu":
            model = model.to("cuda")
        p = "Instruct: Rewrite the given statement in the negative form\nHe always smiles\nOutput:"
        q = tokenizer([p], return_tensors="pt").to(model.device)
        output = model.generate(**q, return_dict_in_generate=True, max_length=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=50256, do_sample=True, repetition_penalty=1.2)
        output = tokenizer.batch_decode(output.sequences)[0]
        print(output)
        print('=')
        q = tokenizer([output[:-1]], return_tensors="pt").to(model.device)
        r, output = model.calculate(**q)
        print(r, output.logits.shape)


@test_mode
def test_hidden_state():
    import torch
    from transformers import AutoModelForCausalLM
    I1 = torch.tensor([43993, 25, 644, 460, 1312, 1234, 287, 2130, 338, 47259, 9294, 284, 33401, 606, 30, 198, 26410, 25, 1867, 466, 345, 1612, 30, 198, 50256, 50256], device="cuda", dtype=torch.long)
    A1 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], device="cuda")
    # L1 = torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1867, 466, 345, 1612, 30, 198, 50256, 50256]], device="cuda", dtype=torch.long)
    
    I2 = torch.tensor([43993, 25, 644, 460, 1312, 1234, 287, 2130, 338, 47259, 9294, 284, 33401, 606, 30, 198, 26410, 25, 1867, 466, 345, 1612, 30, 198, 50256], device="cuda", dtype=torch.long)
    A2 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device="cuda")
    # L2 = torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1867, 466, 345, 1612, 30, 198, 50256]], device="cuda", dtype=torch.long)
    mpath = "/root/model/phi-2"
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    # print(model(I1, attention_mask=A1, labels=L1).logits)
    # print(model(I2, attention_mask=A2, labels=L2).logits)
    print(model(I1, attention_mask=A1).logits)
    print(model(I2, attention_mask=A2).logits)


@test_mode
def test_rm_with_dpo():
    import torch
    from transformers import AutoTokenizer, AutoModel
    mpath = "/root/exp-modeling/output/checkpoint/DPO/phi-2_Exp30/checkpoint-1500"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModel.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    p = "Instruct: {}\nOutput: {}"
    I = []
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an immense amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an immense amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land."))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an immense amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land.\n"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an immense amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land.\n<|endoftext|>"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an immense amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land.<|endoftext|>"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an immense amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land<|endoftext|>"))
    # rejected
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an enormous amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land.\n<|endoftext|>"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an enormous amount of energy, and they can only stay airborne for a limited amount of time before they become exhausted and must land.<|endoftext|>"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Flying requires the user to expend an enormous amount of energy, and they can only stay airborne for limited amount of time before they become exhausted and must land.\n<|endoftext|>"))
    # others
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Network latency refers to the time it takes for data to travel from one point to another in a network.\n<|endoftext|>"))
    I.append(p.format("Make up a limitation for a given superpower.\nFlight", "Network latency refers to the time delay between the sending and receiving of data packets over a network.\n<|endoftext|>"))

    I.append(p.format("What's your name?\n", "I'm Donald Trump.\n<|endoftext|>"))
    I.append(p.format("What's your name?\n", "It's not polite to ask someone's name in such a direct way.\n<|endoftext|>"))
    I.append(p.format("What's your name?\n", "Network latency refers to the time it takes for data to travel from one point to another in a network.\n<|endoftext|>"))
    

    INPUT = [
        tokenizer([item], return_tensors="pt").to(model.device) for item in I
    ]
    for item in INPUT:
        r, output = model.calculate(**item)
        print(r[0].item(), end=" ")
        print(tokenizer.decode(torch.argmax(output.logits[:, -1], dim=-1)))


@test_mode
def test_mistral_tok():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/model/Mistral-7B-v02')
    mes = [
        {"role": "user", "content": "A A A A"},
        {"role": "assistant", "content": "A A A A"},
    ]
    # print(tokenizer.apply_chat_template(mes, return_tensors="pt"))
    print(tokenizer('<s>[INST] A A A A [/INST]\nA A A A<\s>', add_special_tokens=False))
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    print(tokenizer.pad_token_id)
    print(tokenizer('', add_special_tokens=False))
    print(tokenizer('[INST]', add_special_tokens=False))


@test_mode
def test_mistral():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, PeftModel
    tokenizer = AutoTokenizer.from_pretrained('/root/model/Mistral-7B-v02')
    
    model = AutoModel.from_pretrained('/root/model/Mistral-7B-v02', torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=None,
        target_modules=r".*mlp\..*proj"
    )

    model = PeftModel(model, peft_config, 'ex')
    s = '<s>[INST] Are you kidding me? [/INST]'
    t = tokenizer(s, return_tensors="pt").to("cuda")
    o = model.generate(**t, max_length=100)
    print(tokenizer.decode(o[0]))



@test_mode
def test_peft():
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModel

    mpath = "/root/model/phi-2"
    m = AutoModel.from_pretrained(mpath, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=None,
        target_modules=r".*mlp\.fc.*"
    )

    m = PeftModel(m, peft_config, 'ex')
    m.base_model.model.rm_head.requires_grad_(True)
    for k, v in m.named_parameters():
        print(k, v.requires_grad)
    with m.disable_adapter():
        for k, v in m.named_parameters():
            print(k, v.requires_grad)
    print('=== ')
    for k, v in m.named_parameters():
        print(k, v.requires_grad)



@test_mode
def test_grad_require():
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModel

    mpath = "/root/model/phi-2"
    m = AutoModel.from_pretrained(mpath, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=None,
        target_modules=r".*mlp\.fc.*"
    )
    for k, v in m.named_parameters():
        print(k, v.requires_grad)

    m = PeftModel(m, peft_config, 'ex')
    m.add_adapter(peft_config, adapter_name="extra_lora")

    from types import MethodType

    def wrapper(func):
        def new_enable_adapters(self, *args, **kwargs):
            func(*args, **kwargs)
            self.rm_head.ln.requires_grad_(True)
            self.rm_head.linear.requires_grad_(True)
        return new_enable_adapters
    m.enable_adapters = MethodType(wrapper(m.enable_adapters), m)

    # def new_enable_func_wrapper(func):
    #     def new_func(self, *args, **kwargs):
    #         self.func(*args)
    # m.ena
    for k, v in m.named_parameters():
        print(k, v.requires_grad)
    # peft 0.6.0
    m.disable_adapters()
    for k, v in m.named_parameters():
        print(k, v.requires_grad)
    print('==')
    m.enable_adapters()
    for k, v in m.named_parameters():
        print(k, v.requires_grad)
    print('==')
    with m.disable_adapters():
        for k, v in m.named_parameters():
            print(k, v.requires_grad)
    print('==')
    for k, v in m.named_parameters():
        print(k, v.requires_grad)
    print('==')
    


@test_mode
def change_method():
    from types import MethodType
    class A:
        b = 1
        c = 1
        def f1(self, x):
            print(1)
            self.c += 1
            return 2+x
        
    def wrapper(func):
        def new_func(self, *args, **kwargs):
            func(*args, **kwargs)
            self.b = 2
            print(2)
            return 100
        return new_func
    a = A()
    print('=', a.c)
    a.f1(3)
    print('=', a.c)
    a.f1 = MethodType(wrapper(a.f1), a)
    a.f1(3)
    print('=', a.c)



@test_mode
def get_name():
    import torch
    from transformers import AutoModel

    mpath = "/root/model/phi-2"
    m = AutoModel.from_pretrained(mpath, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    a = torch.tensor([1.]*2560, device="cuda", dtype=torch.float16)
    for k, _ in m.named_parameters():
        print(k)
    print(m.rm_head.ln(a))


if __name__ == "__main__":
    import sys
    import inspect
    module = __import__(__name__) 
    all_funcs = [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)] 
    if len(sys.argv) > 1:
        func = sys.argv[1]
        if func in UNREACHABLE_LIST:
            raise ValueError(func)
        if func in all_funcs:
            sys.argv.pop(1)
            eval(func)()
        else:
            raise KeyError
    else:
        raise KeyError
    
