import json
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def wait_to_be_implemented(func):
    print(f"{func.__name__} has not been implemented yet")
    return func


def wait_to_be_improved(s):
    print("This method has some structural or implementing defects that should be rectified.")
    def F(func):
        return func
    return F


class DistributionProbe:
    def __init__(self, model, tokenizer, max_len=410, max_bs=50, func=None, seed=0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len  # of tokens
        self.max_bs = max_bs  # for model batched forward
        self.use_model = kwargs.get("use_model", "chatglm2")
        self.with_eos = kwargs.get("eos", False)
        self.mimic_dpo = kwargs.get("mimic_dpo", False)
        self.return_hidden_state = kwargs.get("return_hidden_state", False)
        self.store_pt = kwargs.get("store_pt", False)
        if self.with_eos:
            print("with eos: \n<|endoftext|>")

    def get_prob(self, idx, mat, return_log=True):
        """
        idx: [bs,] gt token's index for each query
        mat: [bs, vocab_size] predicted logits for every token 
        """
        mat = F.softmax(mat, dim=-1)
        prob = mat.gather(1, idx.to(torch.int64).unsqueeze(1))
        rank = (mat >= prob).sum(1)
        if return_log:
            return torch.clamp(torch.log(prob.squeeze(1)), min=-20)
        return prob.squeeze(1)
    
    def get_prob(self, idx, mat, return_log=True):
        mat = F.softmax(mat, dim=-1)
        prob = mat.gather(1, idx.to(torch.int64).unsqueeze(1))
        if return_log:
            return torch.clamp(torch.log(prob.squeeze(1)), min=-20)
        return prob.squeeze(1)
    
    @wait_to_be_implemented
    def stream_probing(self, fpath, return_topk=4):
        """
        [fpath] can either be the json file path or the loaded data.
        [return_topk] can either be bool(False means ) or int
        """
        if type(fpath) is list:
            data = fpath
        else:
            with open(fpath) as f:
                data = json.load(f)

    @wait_to_be_implemented
    def get_topk_id_and_prob(self, q, a, k=4, return_log=True):
        pass

    def tmp_label_data(self, data, save_prefix="alpaca_human_pref_"):
        pbar = tqdm(enumerate(data), total=len(data))
        DATA = []
        for idx, item in pbar:
            r = self._response_pair_loop(item)
            pbar.set_description(f'win:{r["w"]["prob_avg"].item():.2f}  lose:{r["l"]["prob_avg"].item():.2f}')
            with open(f"data/perturbed_data/{save_prefix}.jsonl", "a") as f:
                json.dump({
                    "id": idx,
                    "win": r["w"]["prob"].tolist(),
                    "lose": r["l"]["prob"].tolist(),
                }, f)
                f.write('\n')

    @wait_to_be_improved("""\
    Put the tokenizing procedure out of this function, \
    create a new class(Tokenizer) to support different tokenizing strategy.
    """)
    def probing_data(self, fpath, probe_single=False, save_prefix="alpaca_human_pref_"):
        if type(fpath) is list:
            data = fpath
        else:
            with open(fpath) as f:
                data = json.load(f)
        pbar = tqdm(enumerate(data), total=len(data))
        DATA = []
        if probe_single:
            for idx, item in pbar:
                r = self._response_single_loop(item)
                pbar.set_description(f'win:{r["status"]["prob_avg"].item():.2f}')
                with open(f"qa_status/{save_prefix}.jsonl", "a") as f:
                    json.dump({
                        "id": idx,
                        "prob": r["status"]["prob"].tolist(),
                        "rank": r["status"]["rank"].tolist(),
                    }, f)
                    f.write('\n')
        else:
            for idx, item in pbar:
                r = self._response_pair_loop(item)
                pbar.set_description(f'win:{r["w"]["prob_avg"].item():.2f}  lose:{r["l"]["prob_avg"].item():.2f}')
                if self.store_pt:
                    DATA.append({
                        "id": idx,
                        "r": r["w"]["prob"],
                        "l": r["l"]["prob"],
                        "rr": r["w"]["rank"],
                        "lr": r["l"]["rank"],
                        "hidden_state_w": r["w"]["hidden_state"],
                        "hidden_state_l": r["l"]["hidden_state"],
                    })
                else:
                    with open(f"qa_status/{save_prefix}.jsonl", "a") as f:
                        if self.return_hidden_state:
                            json.dump({
                                "id": idx,
                                "r": r["w"]["prob"].tolist(),
                                "l": r["l"]["prob"].tolist(),
                                "rr": r["w"]["rank"].tolist(),
                                "lr": r["l"]["rank"].tolist(),
                                "hidden_state_w": r["w"]["hidden_state"].tolist(),
                                "hidden_state_l": r["l"]["hidden_state"].tolist()
                            }, f)
                        else:
                            json.dump({
                                "id": idx,
                                "r": r["w"]["prob"].tolist(),
                                "l": r["l"]["prob"].tolist(),
                                "rr": r["w"]["rank"].tolist(),
                                "lr": r["l"]["rank"].tolist(),
                            }, f)
                        f.write('\n')
        if self.store_pt:
            torch.save(DATA, f"qa_status/{save_prefix}.pt")

    def _response_single_loop(self, item):
        Q = (item["instruction"] + '\n' +item["input"]).strip()
        A = item["output"]
        # status = {k: v for k, v in zip(["prob", "rank"], self._batch_loop(Q, A))}
        status = {k: v for k, v in zip(["prob", "rank"], self._smart_loop(Q, A))}
        status["prob_avg"] = torch.mean(status["prob"])
        status["output"] = A
        return {
            "status": status,
            "Q": Q,
        }

    def _response_pair_loop(self, item, func=None):
        # data preprocessing, should be implemented in a new function.
        if func is not None:
            Q, Awin, Alose = func(item)
        elif "preference" in item.keys():
            Q = item["instruction"] + '\n' + item["input"]
            prefer = item["preference"]
            assert prefer in [1, 2]
            Awin = item[f"output_{prefer}"]
            Alose = item[f"output_{3-prefer}"]
        else:
            Q = item["input"]
            Awin = item["win"]
            Alose = item["lose"]

        # status_win = {k: v for k, v in zip(["prob", "rank"], self._batch_loop(Q, Awin))}
        status_win = {"prob": self._smart_loop(Q, Awin)}
        status_win["prob_avg"] = torch.mean(status_win["prob"])
        status_win["output"] = Awin

        # status_lose = {k: v for k, v in zip(["prob", "rank"], self._batch_loop(Q, Alose))}
        status_lose = {"prob": self._smart_loop(Q, Alose)}
        status_lose["prob_avg"] = torch.mean(status_lose["prob"])
        status_lose["output"] = Alose

        return {
            "w": status_win,
            "l": status_lose,
            "greedy": status_greedy,
            "sample": status_sample,
            "Q": Q,
        }
    
    def _smart_loop(self, q, a, return_rank=True):
        pin = "Instruct: {}\nOutput:".format(q)
        pcompletion = "Instruct: {}\nOutput: {}".format(q, a)
        if self.with_eos:
            eos_token = "\n<|endoftext|><|endoftext|>" if self.mimic_dpo else "\n<|endoftext|>"
            pcompletion = pcompletion + eos_token
        q_seq = self.tokenizer([pin], return_tensors="pt").to("cuda")
        qa_seq = self.tokenizer([pcompletion], return_tensors="pt").to("cuda")
        if self.mimic_dpo:
            qa_seq["attention_mask"][:, -2] = 0

        with torch.no_grad():
            start = len(q_seq.input_ids[0]) - 1
            Mat = self.model(**qa_seq, return_dict=True, output_hidden_states=True)
            mat = Mat.logits[0, start:-1, :]  # [1, seq, n]
            gt = qa_seq.input_ids[0, start+1:]
            prob = self.get_prob(gt, mat, return_log=True)
        return prob.cpu()


def probe_dist_single_gpu_single_data(**kwargs):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    mpath = kwargs.get("mpath", "/root/chatglm2-6b")
    dataset_path = kwargs.get("dataset_path", "/root/dataset/alpaca_farm/alpaca_human_preference.json")
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    for v in model.parameters():
        R = torch.rand(*v.data.shape) * 1.8 + 0.1
        v.data = R.to("cuda")*v.data
    dp = DistributionProbe(model, tokenizer, **kwargs)
    dp.probing_data(dataset_path, save_prefix=kwargs.get("save_prefix", "tmp"))


if __name__ == "__main__":
    # # init()
    # # test_sort_consuming()
    # # test_tokenize_length()
    # # test_chatglm3()
    # # test_phi_2()
    # # probe_dist_single_gpu(
    # #     mimic_dpo=True,
    # #     dataset_path="/root/exp-modeling/data/helpful-train.json",
    # #     mpath="/root/model/phi-2",
    # #     save_prefix="mimic_dpo_helpful_0_5-train",
    # #     use_model="phi-2",
    # #     return_hidden_state=True,
    # #     store_pt=True,
    # # )
    # ratio = -0.2
    # suffix = f"_R0_{int(ratio*10)}"
    # from src.data import prepare_for_hidden_state_ca
    # probe_dist_single_gpu_single_data_RM(
    #     mimic_dpo=True,
    #     dataset_path="/root/exp-modeling/data/chatbot_arena.json",
    #     mpath="/root/model/phi-2",
    #     save_prefix=f"mimic_dpo_chatbot_arena{suffix}",
    #     use_model="phi-2",
    #     return_hidden_state=True,
    #     store_pt=True,
    #     ratio=ratio
    # )
    # prepare_for_hidden_state_ca(key="reward", suffix=suffix)
    # # probe_dist(
    # #     max_bs=100,
    # #     cuda_list="0,1,2",
    # #     memory="6GiB",
    # #     set_host_low_loading=True,
    # #     max_len=410,
    # #     mpath="/root/chatglm2-6b",
    # #     save_prefix="glm2_alpaca_human_pref_2k_",
    # #     use_model="chatglm2"
    # # )
    # # save_linear_weight_and_ln()

    base = 0.2
    save_linear_weight_and_ln_with_disturb(base)
    # check_param_correctness()

