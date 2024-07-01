import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--exp", "-e", type=str, required=True, help="23-14: Exp23 checkpoint-14")
parser.add_argument("--eb", type=str, required=True, help="baseline")
parser.add_argument("--name", "-n", type=str, required=True, help="dataset name")
args = parser.parse_args()

exp, cp = args.exp.split('-')
expb, cpb = args.eb.split('-')
m_path = "/root/exp-modeling/output/checkpoint/RLHF_RM/phi-2_Exp{}/checkpoint-{}"

tokenizer = AutoTokenizer.from_pretrained("/root/model/phi-2", trust_remote_code=True)
model = AutoModel.from_pretrained(m_path.format(exp, cp), trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
model_base = AutoModel.from_pretrained(m_path.format(expb, cpb), trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

ds_path = "/root/exp-modeling/data/sampled_data/{}.jsonl".format(args.name)
ds = load_dataset("json", data_files={"train": ds_path}, split="train")
ds = ds.filter(lambda x:"result_4" in x)
print(len(ds))
QA = "Instruct: {}\nOutput: {}\n<|endoftext|>"
C, c = 0, 0
with open(f"/root/exp-modeling/output/infer_result/{args.name}.jsonl", "w") as f:
    pbar = tqdm(ds, total=len(ds))
    for item in pbar:
        C += 1
        haf_idx, base_idx = 0, 0
        haf_max, base_max = float("-inf"), float("-inf")
        score_haf = []
        score_base = []
        with torch.no_grad():
            for i in range(1, 5):
                seq = tokenizer(QA.format(item["input"], item[f"result_{i}"]), return_tensors="pt").to("cuda")
                if len(seq.input_ids[0]) > 512:
                    break
                r_haf = model.calculate(**seq)[0][0].item()  # r, forward_output = model.calculate()
                score_haf.append(r_haf)
                if r_haf > haf_max:
                    haf_max, haf_idx = r_haf, i
                r_base = model_base.calculate(**seq)[0][0].item()
                score_base.append(r_base)
                if r_base > base_max:
                    base_max, base_idx = r_base, i
            else:
                # token_leng <= 512
                json.dump({
                    "input": item["input"],
                    "haf": item[f"result_{haf_idx}"],
                    "hafRM_hafPick": score_haf[haf_idx-1],
                    "hafRM_basePick": score_haf[base_idx-1],
                    "base": item[f"result_{base_idx}"],
                    "baseRM_hafPick": score_base[haf_idx-1],
                    "baseRM_basePick": score_base[base_idx-1],
                }, f)
                f.write('\n')  
                c += 1
                pbar.set_description(f"{c} / {C}")
