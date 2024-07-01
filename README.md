# Code-for-HAF-RM
Code for HAF-RM (anonymous)

There are some absolute paths in the codes, which only work with the same code structure as below:
```
/root/
|--- exp-modeling/
|--- |--- train_rm.py
|--- |--- data/
|--- |--- |--- harmless-train.json
|--- |--- |--- ......
|--- |--- |--- sampled_data/
|--- |--- |--- sampled_data_mistral/
|--- |--- output/
|--- |--- |--- checkpoint/
|--- |--- |--- eval_result/
|--- |--- |--- gpt_judge/
|--- |--- |--- infer_result/
|--- |--- generation/
|--- |--- tensorboard/
|--- |--- src/
|--- |--- |--- train_dpo_lora.py
|--- |--- |--- ......
|--- |--- script/
|--- |--- |--- ......
|--- model/
|--- |--- Mistral-7B-v02
|--- |--- phi-2
|--- trl/
