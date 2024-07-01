from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random


class Sampler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def test_batch_sample(self, q, n=4):
        p = f"Instruct: {q}\nOutput:"
        length = len(p)
        p = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        results = []
        # for _ in range(2*n):
        result2 = self.model.generate(**p, 
                                    return_dict_in_generate=True, 
                                    max_length=512,
                                    eos_token_id=self.tokenizer.eos_token_id, 
                                    pad_token_id=50256,  # eos_token_id
                                    top_p=0.85,
                                    temperature=random.uniform(0.5, 1.2),
                                    do_sample=True,
                                    num_return_sequences=n,
                                    )
        print(type(result2))
        result2 = self.tokenizer.batch_decode(result2.sequences)
        print(result2)
        #     if result2.endswith("<|endoftext|>"):
        #         result2 = result2[:-13].strip()
        #     else:
        #         result2 = result2.strip()
        #     if result2 not in results:
        #         results.append(result2)
        #     if len(results) == n:
        #         break
        # return results 

    def sample_n_topk(self, q, n=4):
        p = f"Instruct: {q}\nOutput:"
        length = len(p)
        p = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        results = []
        try:
            for _ in range(2*n):
                result2 = self.model.generate(**p, 
                                            return_dict_in_generate=True, 
                                            max_length=512,
                                            eos_token_id=self.tokenizer.eos_token_id, 
                                            pad_token_id=50256,  # eos_token_id
                                            top_p=0.85,
                                            temperature=random.uniform(0.5, 1.2),
                                            do_sample=True,
                                            )

                result2 = self.tokenizer.batch_decode(result2.sequences)[0][length:]
                if result2.endswith("<|endoftext|>"):
                    result2 = result2[:-13].strip()
                else:
                    result2 = result2.strip()
                if result2 not in results:
                    results.append(result2)
                if len(results) == n:
                    break
            return results 
        except Exception as e:
            print(e)
            return results

    def sample_sentence_beam_and_topk(self, q):
        p = f"Instruct: {q}\nOutput:"
        length = len(p)
        p = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        try:
            result1 = self.model.generate(**p, 
                                        return_dict_in_generate=True, 
                                        max_length=512,
                                        eos_token_id=self.tokenizer.eos_token_id, 
                                        pad_token_id=50256,
                                        num_beams=4
                                        )
            result2 = self.model.generate(**p, 
                                        return_dict_in_generate=True, 
                                        max_length=512,
                                        eos_token_id=self.tokenizer.eos_token_id, 
                                        pad_token_id=50256,  # eos_token_id
                                        top_p=0.85,
                                        temperature=1.2,
                                        do_sample=True,
                                        )
            result1 = self.tokenizer.batch_decode(result1.sequences)[0][length:]
            result2 = self.tokenizer.batch_decode(result2.sequences)[0][length:]
            for _ in range(3):
                if result2 == result1:
                    result2 = self.model.generate(**p, 
                                                return_dict_in_generate=True, 
                                                max_length=512,
                                                eos_token_id=self.tokenizer.eos_token_id, 
                                                pad_token_id=50256,  # eos_token_id
                                                top_p=0.85,
                                                temperature=2.,
                                                do_sample=True,
                                                )
                    result2 = self.tokenizer.batch_decode(result2.sequences)[0][length:]
                else: break
            if result1.endswith("<|endoftext|>"):
                result1 = result1[:-13]
            if result2.endswith("<|endoftext|>"):
                result2 = result2[:-13]
            return result1.strip(), result2.strip()
        except Exception as e:
            return None, None

    def sample_sentence_beam(self, q):
        p = f"Instruct: {q}\nOutput:"
        length = len(p)
        p = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        try:
            result1 = self.model.generate(**p, 
                                        return_dict_in_generate=True, 
                                        max_length=512,
                                        eos_token_id=self.tokenizer.eos_token_id, 
                                        pad_token_id=50256,
                                        num_beams=4
                                        )
            result1 = self.tokenizer.batch_decode(result1.sequences)[0][length:]
            if result1.endswith("<|endoftext|>"):
                result1 = result1[:-13]
            return result1.strip()
        except Exception as e:
            return None
    
    def sample_sentence_sample(self, q):
        p = f"Instruct: {q}\nOutput:"
        length = len(p)
        p = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        try:
            result1 = self.model.generate(**p, 
                                        return_dict_in_generate=True, 
                                        max_length=512,
                                        eos_token_id=self.tokenizer.eos_token_id, 
                                        pad_token_id=50256,
                                        num_beams=1,
                                        do_sample=True,
                                        repetition_penalty=1.2,
                                        )
            result1 = self.tokenizer.batch_decode(result1.sequences)[0][length:]
            if result1.endswith("<|endoftext|>"):
                result1 = result1[:-13]
            return result1.strip()
        except Exception as e:
            return None
    

def sample_anthropic_human_pref_phi_2(RANK=0, exp=2, data_name="alpaca"):
    import json
    from tqdm import tqdm
    mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-500"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)

    # with open("/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json") as f:
    dpath = {
        "chatbot-arena": "chatbot_arena",
        "alpaca": "alpaca_human_pref_phi_2_sample"
    }[data_name]
    with open(f"/root/exp-modeling/data/{dpath}.json") as f:
        data = json.load(f)
    for item in tqdm(data):
        result1, result2 = s.sample_sentence_beam_and_topk(item["input"])
        if result1 is not None:
            d = {"input": item["input"], "win": result1, "lose": result2}
            with open(f"/root/exp-modeling/generation/tmp_{data_name}_"+str(RANK)+".jsonl", "a") as f:
                json.dump(d, f)
                f.write('\n')


def sample_beam(RANK=0, exp=4, data_name="beaver", step=500):
    import json
    from tqdm import tqdm
    # mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-1500"
    mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-{step}"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)

    # with open("/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json") as f:
    dpath = {
        "chatbot-arena": "chatbot_arena",
        "alpaca": "alpaca_human_pref_phi_2_sample",
        "beaver": "beaver_eval",
        "helpful": "helpful-test",
    }[data_name]
    with open(f"/root/exp-modeling/data/{dpath}.json") as f:
        data = json.load(f)
    for item in tqdm(data):
        result1 = s.sample_sentence_beam(item["input"])
        if result1 is not None:
            if "response" in item:
                d = {"input": item["input"], "win": result1, "lose": item["response"]}
            else:
                d = {"input": item["input"], "win": result1, "lose": result1}
            with open(f"/root/exp-modeling/generation/tmp_E{exp}S{step}_{data_name}_"+str(RANK)+".jsonl", "a") as f:
                json.dump(d, f)
                f.write('\n')


def sample_beam_single(exp=4, step=500):
    import json
    from pathlib import Path
    from tqdm import tqdm
    # mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-1500"
    mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-{step}"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)

    # with open("/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json") as f:
    with open(f"/root/exp-modeling/data/total_eval_gen.json") as f:
        data = json.load(f)
    assert not Path(f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/c{step}.txt").exists()
    for idx, item in tqdm(enumerate(data), total=600):
        result1 = s.sample_sentence_beam(item["input"])
        i = idx % 200
        if result1 is not None:
            with open(f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/c{step}.jsonl", "a") as f:
                d = {"input": item["input"], "response": result1, "idx": item["from"]+f'-{i}'}
                json.dump(d, f)
                f.write('\n')


def sample_sample_single(exp=4, step=500):
    import json
    from pathlib import Path
    from tqdm import tqdm
    # mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-1500"
    mpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/checkpoint-{step}"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)

    # with open("/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json") as f:
    with open(f"/root/exp-modeling/data/total_eval_gen.json") as f:
        data = json.load(f)
    assert not Path(f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/c{step}.txt").exists()
    for idx, item in tqdm(enumerate(data), total=600):
        result1 = s.sample_sentence_sample(item["input"])
        i = idx % 200
        if result1 is not None:
            with open(f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/c{step}.jsonl", "a") as f:
                d = {"input": item["input"], "response": result1, "idx": item["from"]+f'-{i}'}
                json.dump(d, f)
                f.write('\n')


from datasets import load_dataset, concatenate_datasets

def get_data_beaver():
    KEY = "safer_response_id"
    
    eval_dataset = load_dataset("json", data_files={"eval": "/root/exp-modeling/data/BeaverTails_eval.jsonl"}, split="eval").filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"])
    reform = lambda examples: {
        "prompt": examples["prompt"],
    }
    return eval_dataset.map(reform).shuffle(seed=0).select(range(1000, 2000))

def get_data_alpaca():
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json"}, split="train")
    tl = len(dataset)
    train_dataset = dataset.select(range(int(0.1*tl), tl))
    reform = lambda examples: {
        "prompt": examples["input"],
    }
    return train_dataset.map(reform).shuffle(seed=0).select(range(len(train_dataset)-1000, len(train_dataset)))

def get_hh(label="helpful"):
    if label == "helpful":
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/helpful-train.json"}, split="train")
    if label == "harmless":
        train_dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/harmless-train.json"}, split="train")
    reform = lambda examples: {
        "prompt": examples["input"],
    }
    return train_dataset.map(reform).shuffle(seed=0).select(range(len(train_dataset)-1000, len(train_dataset)))

def get_data_arena():
    dataset = load_dataset("json", data_files={"train": "/root/exp-modeling/data/chatbot_arena.json"}, split="train")
    tl = len(dataset)
    train_dataset = dataset.select(range(int(0.05*tl), tl))
    reform = lambda examples: {
        "prompt": examples["input"],
    }
    return train_dataset.map(reform).shuffle(seed=0).select(range(len(train_dataset)-1000, len(train_dataset)))


def sample_output():
    import argparse
    import json
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="")
    args = parser.parse_args()
    mpath = f"/root/model/phi-2"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)
    with open(f"/root/exp-modeling/data/sampled_data/{args.data}.jsonl", "w") as f:
        if args.data == "beaver":
            ds = get_data_beaver()
        elif args.data == "alpaca":
            ds = get_data_alpaca()
        elif args.data == "helpful":
            ds = get_hh("helpful")
        elif args.data == "harmless":
            ds = get_hh("harmless")
        elif args.data == "chatbot":
            ds = get_data_arena()
        else:
            raise KeyError
        for item in tqdm(ds, total=1000):
            result = s.sample_n_topk(item["prompt"])
            json.dump({"input": item["prompt"], **{f'result_{i+1}': d for i, d in enumerate(result)}}, f)
            f.write('\n')


def test():
    mpath = f"/root/model/phi-2"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)
    s.test_batch_sample("hello, who are you")


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--rank", type=int, default=0)
    # parser.add_argument("--exp", type=int, default=2)
    # parser.add_argument("--dname", type=str, default="chatbot-arena")
    # parser.add_argument("--step", type=int, default=500)
    # args = parser.parse_args()
    # # sample_anthropic_human_pref_phi_2(RANK=args.rank, exp=args.exp, data_name=args.dname)
    # # sample_beam(RANK=args.rank, exp=args.exp, data_name=args.dname, step=args.step)
    # sample_beam_single(exp=args.exp, step=args.step)
    # # combine(total=args.rank)

    # sample_output()
    test()