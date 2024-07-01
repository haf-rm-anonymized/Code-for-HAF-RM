import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--exp", "-e", type=str, required=True, help="23-14: Exp23 checkpoint-14")
parser.add_argument("--eb", type=str, required=True, help="baseline")
parser.add_argument("--name", "-n", type=str, required=True, help="dataset name")
parser.add_argument("--base", action="store_true")
parser.add_argument("--new", action="store_true")
args = parser.parse_args()

exp, cp = args.exp.split('-')
expb, cpb = args.eb.split('-')
mistral_path = "/root/model/Mistral-7B-v02"
lora_path = "/root/exp-modeling/output/checkpoint/HAF-lora/mistral_Exp{}/checkpoint-{}"

tokenizer = AutoTokenizer.from_pretrained(mistral_path, trust_remote_code=True)
model = AutoModel.from_pretrained(mistral_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
# model_base = AutoModel.from_pretrained(mistral_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

if not args.base:
    config = PeftConfig.from_pretrained(lora_path.format(exp, cp))
    model = PeftModel.from_pretrained(model, lora_path.format(exp, cp), device_map="auto")
else:
    config = PeftConfig.from_pretrained(lora_path.format(expb, cpb))
    model = PeftModel.from_pretrained(model, lora_path.format(expb, cpb), device_map="auto")
# model_base = PeftModel.from_pretrained(model_base, lora_path.format(eexpb, cpb), device_map="auto")

####################
# from peft import get_peft_model, LoraConfig
# peft_config = LoraConfig(
#     r=64,
#     lora_alpha=16,
#     bias="none",
#     task_type="SEQ_CLS",
#     # target_modules=r".*mlp\.fc.*"
#     target_modules=r".*mlp\..*proj"
# )
# model = get_peft_model(model, peft_config)
####################

ds_path = "/root/exp-modeling/data/sampled_data_mistral/{}.jsonl".format(args.name)
ds = load_dataset("json", data_files={"train": ds_path}, split="train")
ds = ds.filter(lambda x:"result_8" in x)
print(len(ds))
QA = "[INST] {} [/INST]{}</s>"
C, c = 0, 0
TOTAL_RESULT = []
KEY = "base" if args.base else "haf"
with open(f"/root/exp-modeling/output/infer_result/mistral_tmp/{args.name}_mistral{'_base' if args.base else ''}{'_new' if args.new else ''}.jsonl", "w") as f:
    pbar = tqdm(ds, total=len(ds))
    for item in pbar:
        C += 1
        haf_idx, base_idx = 0, 0
        haf_max, base_max = float("-inf"), float("-inf")
        score_haf = []
        score_base = []
        Flag = True
        with torch.no_grad():
            for i in range(1, 9):
                seq = tokenizer(QA.format(item["input"], item[f"result_{i}"]), return_tensors="pt").to("cuda")
                if len(seq.input_ids[0]) > 520:
                    Flag = False
                    continue
                r_haf = model.calculate(**seq, return_dict=True, output_hidden_states=True)[0][0].item()  # r, forward_output = model.calculate()
                score_haf.append(r_haf)
                if r_haf > haf_max:
                    haf_max, haf_idx = r_haf, i
            if Flag:
                c += 1
            if haf_max == float("-inf"):
                continue
            json.dump({
                "input": item["input"],
                KEY: item[f"result_{haf_idx}"],
                **{f"result_{i+1}": {"response": item[f"result_{i+1}"], "score_"+KEY: score_haf[i]} for i in range(len(score_haf))}
            }, f)
            f.write('\n')  
            pbar.set_description(f"{c} / {C}")
