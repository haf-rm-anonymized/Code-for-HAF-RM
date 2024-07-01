import random
from datasets import load_dataset


__all__ = ["load_and_preprocess"]


def get_config(args):
    dataset_config = DATASET_CONFIG[args.dataset_name]
    dataset_config["remove_columns"] = dataset_config.get("remove_columns", ["input", "win", "lose"])
    dataset_config["preprocess_func"] = dataset_config.get("preprocess_func", preprocess_alpaca_ref)
    dataset_config["path_eval"] = dataset_config.get("path_eval", None)
    for k, v in TOKENIZER_CONFIG.items():
        if k in args.model_name:
            return dataset_config, v
    raise RuntimeError("Unable to find tokenize func.")


def load_and_preprocess(args, tokenizer, split_from_train_ratio=None, shuffle_seed=1):
    dataset_config, tokenize_config = get_config(args)

    train_dataset = load_dataset("json", data_files={"train": dataset_config["path_train"]}, split="train")
    train_dataset = train_dataset.map(
        dataset_config["preprocess_func"],
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "config": {"train_eval": "train", **tokenize_config}},
        remove_columns=dataset_config["remove_columns"]
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["chosen_input_ids"][0]) <= args.reward_config.max_length
        and len(x["rejected_input_ids"][0]) <= args.reward_config.max_length
    ).shuffle(seed=shuffle_seed)

    eval_dataset = None
    if dataset_config["path_eval"] is not None:
        eval_dataset = load_dataset("json", data_files={"eval": dataset_config["path_eval"]}, split="eval")
        eval_dataset = eval_dataset.map(
            dataset_config["preprocess_func"],
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "config": {"train_eval": "eval", **tokenize_config}},
            remove_columns=dataset_config["remove_columns"]
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["chosen_input_ids"][0]) <= args.reward_config.max_length
            and len(x["rejected_input_ids"][0]) <= args.reward_config.max_length
        ).shuffle(seed=shuffle_seed)
    else:
        if split_from_train_ratio is not None:
            total = len(train_dataset)
            assert split_from_train_ratio < 0.5
            thre = int(total * split_from_train_ratio)
            eval_dataset = train_dataset.select(range(thre))
            train_dataset = train_dataset.select(range(thre, total))
    if "extra_map" in dataset_config and dataset_config["extra_map"]:
        train_dataset = train_dataset.map(dataset_config["extra_map"])
    if "extra_filter" in dataset_config and dataset_config["extra_filter"]:
        train_dataset = train_dataset.filter(dataset_config["extra_filter"])
    if len(train_dataset) > args.max_dataset_length:
        train_dataset = train_dataset.select(range(args.max_dataset_length))
    print('===')
    print("total length:", len(train_dataset))
    return train_dataset, eval_dataset

######################################################################
#                                                                    #
#                      Preprocess Function                           #
#                                                                    #
######################################################################
def Template_Preprocess_Function(examples, tokenizer, config):       #
    """                                                              #
    TEMPLATE function of data preprocessing                          #
    Note that tokenize function is in *config*                       #
    """                                                              #
    new_examples = {                                                 #
        "chosen_input_ids": [],                                      #
        "chosen_attention_mask": [],                                 #
        "rejected_input_ids": [],                                    #
        "rejected_attention_mask": []                                #
    }                                                                #
    func = config["tokenize_func"]  # get tokenize function          #
    for var1, var2 in zip(examples["key1"], examples["key2"]):       #
        # update new_examples                                        #
        question = None                                              #
        answer = None                                                #
        _ = func(question, answer, tokenizer)                        #
        ...                                                          #
    return new_examples                                              #
######################################################################


def preprocess_alpaca_ref(examples, tokenizer, config):
    new_examples = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": []
    }
    func = config["tokenize_func"]
    if "consensus" not in examples:
        for q, w, l in zip(examples["input"], examples["win"], examples["lose"]):
            qw = func(q, w, tokenizer)
            ql = func(q, l, tokenizer)
            new_examples["chosen_input_ids"].append(qw["input_ids"])
            new_examples["rejected_input_ids"].append(ql["input_ids"])
            new_examples["chosen_attention_mask"].append(qw["attention_mask"])
            new_examples["rejected_attention_mask"].append(ql["attention_mask"])
        return new_examples
    else:
        new_examples["consensus"] = []
        for q, w, l, con in zip(examples["input"], examples["win"], examples["lose"], examples["consensus"]):
            qw = func(q, w, tokenizer)
            ql = func(q, l, tokenizer)
            new_examples["chosen_input_ids"].append(qw["input_ids"])
            new_examples["rejected_input_ids"].append(ql["input_ids"])
            new_examples["chosen_attention_mask"].append(qw["attention_mask"])
            new_examples["rejected_attention_mask"].append(ql["attention_mask"])
            new_examples["consensus"].append(1 if con else 0)
        return new_examples
    

def preprocess_wrapper_alpaca_ref_with_inverse(inv=0, reverse_eval=True):
    def func(examples, tokenizer, config):
        new_examples = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": []
        }
        func = config["tokenize_func"]
        if "consensus" not in examples:
            for q, w, l in zip(examples["input"], examples["win"], examples["lose"]):
                if random.random() < inv and (config["train_eval"] == "train" or reverse_eval):
                    w, l = l, w
                qw = func(q, w, tokenizer)
                ql = func(q, l, tokenizer)
                new_examples["chosen_input_ids"].append(qw["input_ids"])
                new_examples["rejected_input_ids"].append(ql["input_ids"])
                new_examples["chosen_attention_mask"].append(qw["attention_mask"])
                new_examples["rejected_attention_mask"].append(ql["attention_mask"])
            return new_examples
        else:
            new_examples["consensus"] = []
            for q, w, l, con in zip(examples["input"], examples["win"], examples["lose"], examples["consensus"]):
                if random.random() < inv and (config["train_eval"] == "train" or reverse_eval):
                    w, l = l, w
                qw = func(q, w, tokenizer)
                ql = func(q, l, tokenizer)
                new_examples["chosen_input_ids"].append(qw["input_ids"])
                new_examples["rejected_input_ids"].append(ql["input_ids"])
                new_examples["chosen_attention_mask"].append(qw["attention_mask"])
                new_examples["rejected_attention_mask"].append(ql["attention_mask"])
                new_examples["consensus"].append(1 if con else 0)
            return new_examples
    return func


######################################################################
#                                                                    #
#                        Tokenize Function                           #
#                                                                    #
######################################################################
def Template_Tokenize_Function(q, a, tokenizer):                     #
    """                                                              #
    TEMPLATE function of tokenization                                #
    input: question, answer, tokenizer                               #
    output: {"input_ids": ..., "attention_mask": ...}                #
    """                                                              #
    return tokenizer([q+a], return_tensors="pt")                     #
######################################################################


def tokenize_phi_2(q, a, tokenizer):
    completion = f"Instruct: {q}\nOutput: {a}"
    return tokenizer([completion], padding="max_length", max_length=512, return_tensors="pt")


######################################################################
#                                                                    #
#                        Extra filter                                #
#                                                                    #
######################################################################
def consensus_filter(x):
    return x["consensus"] == 1


def disagreement_filter(x):
    return x["consensus"] == 0


def mix_filter_wrapper(inv=None):
    def filter(x):
        return x["consensus"] == 1 or random.random() < inv
    return filter


def reverse_mix_filter_wrapper(inv=None):
    def filter(x):
        return x["consensus"] == 0 or random.random() < inv
    return filter


def random_drop_wrapper(inv=None):
    def filter(x):
        return random.random() < inv
    return filter


######################################################################
#                                                                    #
#                              Config                                #
#                                                                    #
######################################################################
"""
- alpaca-human-gt-sum-{k} drop (100-k)% disagreed data, agree : disagree = 48:(k*0.52)
- alpaca-human-gt-sum-u{k} drop (100-k)% agreed data, agree : disagree = (0.48*k):52
- alpaca-phi-dist-{k} reverse (k)% label

config: {
    "path_train": ...,
    "path_eval": None,
    "preprocess_func": preprocess_alpaca_ref,
    "remove_columns": ["input", "win", "lose"],
}
"""
DATASET_CONFIG = {
    "dpo": {
        "path_train": "/root/exp-modeling/data/dpo_sample_train.json",
        "path_eval": "/root/exp-modeling/data/dpo_sample_eval.json"
    },
    "dpo-inv": {
        "path_train": "/root/exp-modeling/data/dpo_sample_train.json",
        "path_eval": "/root/exp-modeling/data/dpo_sample_eval.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=1)
    },
    "dpo-10": {
        "path_train": "/root/exp-modeling/data/dpo_sample_train.json",
        "path_eval": "/root/exp-modeling/data/dpo_sample_eval.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.1, reverse_eval=False)
    },
    "dpo-30": {
        "path_train": "/root/exp-modeling/data/dpo_sample_train.json",
        "path_eval": "/root/exp-modeling/data/dpo_sample_eval.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.3, reverse_eval=False)
    },
    "dpo-50": {
        "path_train": "/root/exp-modeling/data/dpo_sample_train.json",
        "path_eval": "/root/exp-modeling/data/dpo_sample_eval.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.5, reverse_eval=False)
    },
    "dpo-70": {
        "path_train": "/root/exp-modeling/data/dpo_sample_train.json",
        "path_eval": "/root/exp-modeling/data/dpo_sample_eval.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.7, reverse_eval=False)
    },
    "alpaca-human-0": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json",
    },
    "alpaca-human-0-sum": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I0-sum.json",
    },
    "alpaca-human-0-sum-unique": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I0-sum.json",
        "extra_filter": disagreement_filter,
    },
    "alpaca-human-15": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I15.json",
    },
    "alpaca-human-30": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I30.json",
    },
    "alpaca-human-50": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I50.json",
    },
    "alpaca-human-gt": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt.json",
    },
    "alpaca-human-gt-sum": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
    },
    "alpaca-human-gt-sum-10": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": mix_filter_wrapper(0.1),
    },
    "alpaca-human-gt-sum-30": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": mix_filter_wrapper(0.3),
    },
    "alpaca-human-gt-sum-50": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": mix_filter_wrapper(0.5),
    },
    "alpaca-human-gt-sum-70": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": mix_filter_wrapper(0.7),
    },
    "alpaca-human-gt-sum-consensus": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": consensus_filter,
    },
    "alpaca-human-gt-sum-unique": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": disagreement_filter,
    },
    "alpaca-human-gt-sum-u10": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": reverse_mix_filter_wrapper(0.1),
    },
    "alpaca-human-gt-sum-u30": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": reverse_mix_filter_wrapper(0.3),
    },
    "alpaca-human-gt-sum-u50": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": reverse_mix_filter_wrapper(0.5),
    },
    "alpaca-human-gt-sum-u70": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt-sum.json",
        "extra_filter": reverse_mix_filter_wrapper(0.7),
    },
    "alpaca-phi-dist": {
        "path_train": "/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json",
    },
    "alpaca-phi-dist-5": {
        "path_train": "/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.05),
    },
    "alpaca-phi-dist-10": {
        "path_train": "/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.1),
    },
    "alpaca-phi-dist-30": {
        "path_train": "/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.3),
    },
    "alpaca-phi-dist-50": {
        "path_train": "/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json",
        "preprocess_func": preprocess_wrapper_alpaca_ref_with_inverse(inv=0.5),
    },
    "chatbot-arena": {
        "path_train": "/root/exp-modeling/data/chatbot_arena.json",
    },
    "harmless": {
        "path_train": "/root/exp-modeling/data/harmless-train.json",
        "path_eval": "/root/exp-modeling/data/harmless-test.json",
    },
    "helpful": {
        "path_train": "/root/exp-modeling/data/helpful-train.json",
        "path_eval": "/root/exp-modeling/data/helpful-test.json",
    },
    "anthropic-hh": {
        "path_train": [
            "/root/exp-modeling/data/helpful-train.json",
            "/root/exp-modeling/data/harmless-train.json",
        ],
        "path_eval": [
            "/root/exp-modeling/data/harmless-test.json",
            "/root/exp-modeling/data/helpful-test.json",
        ],
    },
}

TOKENIZER_CONFIG = {
    "phi-2": {
        "tokenize_func": tokenize_phi_2,
    },
}

#############################################################################
# test
def test():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/model/phi-2", trust_remote_code=True)
    dataset_config = DATASET_CONFIG["alpaca-human-0"]
    tokenize_config = TOKENIZER_CONFIG["phi-2"]
    train_dataset = load_dataset("json", data_files={"train": dataset_config["path_train"]}, split="train")
    train_dataset = train_dataset.map(
        dataset_config["preprocess_func"],
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "config": tokenize_config},
        remove_columns=dataset_config["remove_columns"]
    )
    print(train_dataset)
    print(train_dataset[0])
    m = 0
    for item in train_dataset:
        if len(item["chosen_input_ids"][0]) > m:
            m = len(item["chosen_input_ids"][0])
        if len(item["rejected_input_ids"][0]) > m:
            m = len(item["rejected_input_ids"][0])
    print("max_tokens_num:", m)  # 619
    train_dataset = train_dataset.filter(
        lambda x: len(x["chosen_input_ids"][0]) <= 512
        and len(x["rejected_input_ids"][0]) <= 512
    )
    print(train_dataset)
    train_dataset = train_dataset.select(range(1000))
    print(train_dataset)


def test1():
    import json
    t = 0
    c = 0
    d = 0
    m1 = 0
    m2 = 0
    tie = 0
    with open("/root/dataset/chatbot_arena/train.jsonl") as f:
        for line in f:
            t += 1
            item = json.loads(line)
            if len(item["conversation_a"]) == 2 and len(item["conversation_b"]) == 2:
                c += 1
                a = item["conversation_a"]
                b = item["conversation_b"]
                if a[0]["content"].strip() != b[0]["content"].strip():
                    m1 += 1
                if not(
                    a[0]["role"] == "user"
                    and b[0]["role"] == "user"
                    and a[1]["role"] == "assistant"
                    and b[1]["role"] == "assistant"
                ):
                    m2 += 1
                if item["winner"] == "tie":
                    tie += 1
            if len(item["conversation_a"]) <= 4 and len(item["conversation_b"]) <= 4:
                d += 1
    print(t, d, c, m1, m2, tie)


def test_load():
    dataset_config = DATASET_CONFIG["anthropic-hh"]
    train_dataset = load_dataset("json", data_files={"train": dataset_config["path_train"]}, split="train")
    print(len(train_dataset))


def test_load_and_preprocess():
    from transformers import AutoTokenizer
    class R:
        max_length=512
    class TMP:
        model_name="/root/model/phi-2"
        dataset_name="alpaca-human-gt-sum"
        reward_config=R()

    args = TMP()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side="left"
    tokenizer.add_special_tokens({'pad_token': ' '})
    td, ed = load_and_preprocess(args, tokenizer, split_from_train_ratio=0.1)
    ta, tda = 0, 0
    ea, eda = 0, 0
    for item in td:
        if item["consensus"]:
            ta += 1
        else:
            tda += 1
    for item in ed:
        if item["consensus"]:
            ea += 1
        else:
            eda += 1
    print(ta, tda)
    print(ea, eda)

#################################################################################
# data preprocessing
def get_chatbot_arena_single_round():
    import json
    data = []
    with open("/root/dataset/chatbot_arena/train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            if len(item["conversation_a"]) == 2 and len(item["conversation_b"]) == 2 and "tie" not in item["winner"]:
                if item["winner"] == "model_a":
                    win = "conversation_a"
                    lose = "conversation_b"
                elif item["winner"] == "model_b":
                    win = "conversation_b"
                    lose = "conversation_a"
                else:
                    raise KeyError(item["winner"])
                data.append({
                    "input": item["conversation_a"][0]["content"].strip(),
                    "win": item[win][1]["content"].strip(),
                    "lose": item[lose][1]["content"].strip(),
                })
    print(len(data))
    with open("/root/exp-modeling/data/chatbot_arena.json", "w") as f:
        json.dump(data, f, indent=4)


def get_anthropic_hh():
    import json
    harmpath = "/root/dataset/anthropic-rlhf/harmless-base/"
    helppath = "/root/dataset/anthropic-rlhf/helpful-base/"
    for p in [harmpath, helppath]:
        for d in ["train.jsonl", "test.jsonl"]:
            HA = []
            with open(p+d) as f:
                for line in f:
                    item = json.loads(line)
                    cs = item["chosen"].split("\n\nHuman: ")
                    rs = item["rejected"].split("\n\nHuman: ")
                    if len(cs) == 2 and len(rs) == 2:
                        cs = cs[1].split("\n\nAssistant: ")
                        rs = rs[1].split("\n\nAssistant: ")
                        assert cs[0].strip() == rs[0].strip()
                        HA.append({
                            "input": cs[0].strip(),
                            "win": cs[1].strip(),
                            "lose": rs[1].strip()
                        })
            print(len(HA))
            with open(p.replace("dataset/anthropic-rlhf/", "exp-modeling/data/")[:-5]+d[:-1], "w") as f:
                json.dump(HA, f, indent=4)


def rectify_phi_2_dist():
    import json
    with open("/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json") as f:
        primo_data = json.load(f)
    data = []
    with open("/root/exp-modeling/qa_status/alpaca_human_pref_phi-2_eos_sample.jsonl") as f:
        inv = 0
        prob = []
        for l in f.readlines():
            prob.append(json.loads(l))
    for item, pitem in zip(primo_data, prob):
        if item["win"] == item["lose"]:
            continue
        if item["win"] == "" or item["lose"] == "":
            continue
        w = sum(pitem["r"])
        l = sum(pitem["l"])
        if w > l:
            data.append({
                **item,
            })
        else:
            inv += 1
            data.append({
                "input": item["input"],
                "win": item["lose"],
                "lose": item["win"],
            })
    print(inv / len(data))
    print(len(data))
    with open("/root/exp-modeling/data/alpaca_sample_with_eos.json", "w") as f:
        json.dump(data, f, indent=4)


def prepare_for_hidden_state(split="train", suffix=""):
    """
    {
        "id": idx,
        "r": r["w"]["prob"],
        "l": r["l"]["prob"],
        "rr": r["w"]["rank"],
        "lr": r["l"]["rank"],
        "hidden_state_w": r["w"]["hidden_state"],
        "hidden_state_l": r["l"]["hidden_state"],
    }
    """
    import json
    import torch
    for data_type in ["harmless", "helpful"]:
        with open(f"/root/exp-modeling/data/{data_type}-{split}.json".replace("eval", "test")) as f:
            primo_data = json.load(f)
        data = []
        prob = torch.load(f"/root/exp-modeling/qa_status/mimic_dpo_{data_type}{suffix}-{split}.pt")
        inv = 0
        for item, pitem in zip(primo_data, prob):
            if item["win"] == item["lose"]:
                continue
            if item["win"] == "" or item["lose"] == "":
                continue
            w = sum(pitem["r"])
            l = sum(pitem["l"])
            if w > l:
                data.append({
                    **item,
                    "hidden_state_win": pitem["hidden_state_w"],
                    "hidden_state_lose": pitem["hidden_state_l"],
                })
            else:
                inv += 1
                data.append({
                    "input": item["input"],
                    "win": item["lose"],
                    "lose": item["win"],
                    "hidden_state_win": pitem["hidden_state_l"],
                    "hidden_state_lose": pitem["hidden_state_w"],
                })
        print(inv / len(data))
        print(len(data))
        torch.save(data, f"/root/exp-modeling/data/{data_type}_hidden_state{suffix}-{split}.pt")


def prepare_for_hidden_state_ca(key="prob", suffix=""):
    """
    {
        "id": idx,
        "r": r["w"]["prob"],
        "l": r["l"]["prob"],
        "rr": r["w"]["rank"],
        "lr": r["l"]["rank"],
        "hidden_state_w": r["w"]["hidden_state"],
        "hidden_state_l": r["l"]["hidden_state"],
    }
    """
    import json
    import torch
    with open(f"/root/exp-modeling/data/chatbot_arena.json") as f:
        primo_data = json.load(f)
    data = []
    prob = torch.load(f"/root/exp-modeling/qa_status/mimic_dpo_chatbot_arena{suffix}.pt")
    inv = 0
    for item, pitem in zip(primo_data, prob):
        if item["win"] == item["lose"]:
            continue
        if item["win"] == "" or item["lose"] == "":
            continue
        if key == "prob":
            w = sum(pitem["r"])
            l = sum(pitem["l"])
        elif key == "reward":
            w, l = pitem["r"], pitem["l"]
        else:
            raise KeyError
        if w > l:
            data.append({
                **item,
                "hidden_state_win": pitem["hidden_state_w"],
                "hidden_state_lose": pitem["hidden_state_l"],
            })
        else:
            inv += 1
            data.append({
                "input": item["input"],
                "win": item["lose"],
                "lose": item["win"],
                "hidden_state_win": pitem["hidden_state_l"],
                "hidden_state_lose": pitem["hidden_state_w"],
            })
    print(inv / len(data))
    print(len(data))
    torch.save(data, f"/root/exp-modeling/data/chatbot_arena_hidden_state{suffix}.pt")


if __name__ == "__main__":
    # test()
    # test1()
    # test_load()
    # test_load_and_preprocess()
    # rectify_phi_2_dist()
    # get_chatbot_arena_single_round()
    # get_anthropic_hh()
    # prepare_for_hidden_state(split="train", suffix="_0_5")
    # prepare_for_hidden_state(split="eval", suffix="_0_5")
    prepare_for_hidden_state_ca(suffix="_0_5")

