import json
import re
from collections import defaultdict
from glob import glob
from pathlib import Path


# dpath = "output/checkpoint/simple/"
# model_name = "deberta"
CONFIG = defaultdict(lambda: defaultdict(dict))
# {data_path: {label_for_data: {idx: [freeze_ratio, seed]}}}
# files = glob(dpath+model_name+"_Exp*")
# for file in files:
#     idx = re.findall(r"_Exp(\d+)/?$", file)[0]
#     with open(Path(file) / "config.txt") as f:
#         config = json.load(f)
#     CONFIG[config["data_path"]][config["label_for_data"]][idx] = (config["freeze_ratio"], config["seed"])
# print(CONFIG)

dpath = "output/checkpoint/DPO/"
model_name="phi-2"
for idx in range(155, 189):
    file = dpath+model_name+f"_Exp{idx}"
    with open(Path(file) / "config.txt") as f:
        config = json.load(f)
    CONFIG[config["data_path"]][config["label_for_data"]][idx] = (config["dpo_ratio"], config["loss_type"], config["seed"] if 'seed' in config else 0)
for K, ITEM in CONFIG.items():
    print(K)
    for key, item in ITEM.items():
        print('  '+key)
        for IDX, it in item.items():
            print('    '+str(IDX), end=' ')
            print(it)