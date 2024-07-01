import json
import time
import random
from tqdm import tqdm

from openai import OpenAI


I = 0
# model="gpt-4-turbo-2024-04-09"
model = "gpt-3.5-turbo-0125"
base=""
key="YOUR API KEY"

client = OpenAI(api_key=key, base_url=base)
retry = 3
PROMPT = """Select the output (a) or (b) that best matches the given instruction. Choose your preferred \
output, which can be subjective. Your answer should ONLY contain: Output (a) or Output (b). 
Here's an example:

 # Example:
 ## Instruction:
 Give a description of the following job: "ophthalmologist"

 ## Output (a):
 An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to \
read letters from a chart.

 ## Output (b):
 An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye \
diseases and conditions.

 ## Which is best, Output (a) or Output (b)?
 Output (b)

 Here the answer is Output (b) because it provides a comprehensive and accurate description of \
the job of an ophthalmologist. In contrast, output (a) is more of a joke.

 # Task:
 Now is the real task, do not explain your answer, just say Output (a) or Output (b).

 ## Instruction:
 {question}

 ## Output (a):
 {v1}

 ## Output (b):
 {v2}
 
 ## Which is best, Output (a) or Output (b)?"""

def bi_judge(question="", cpl="", base=""):
    global I
    I += 1
    print(I)
    INFO = {"input": question, "haf": cpl, "base": base}
    if cpl == base or (not cpl and not base):
        return "TIE", "", "TIE", "", INFO
    elif not cpl:
        return "BASELINE", "", "BASELINE", "", INFO
    elif not base:
        return "HAF", "", "HAF", "", INFO
    if random.random() < 0.5:
        reverse = False
        v1, v2 = (cpl, base) if reverse else (base, cpl)
        
        prompt = PROMPT.format(question=question, v1=v1, v2=v2)
        retry_interval = 1
        for _ in range(retry):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": " You are a helpful instruction-following assistant that prints the best model by selecting the best outputs for a given instruction."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=32,
                    temperature=0
                )
            
                res = response.choices[0].message.content
                processed = ''.join(res.lower().split())

                # if reverse  a=haf  b=baseline
                if processed.startswith("output(a)") or processed.startswith("outputa"):
                    choice = "HAF" if reverse else "BASELINE"
                    return choice, res, choice, res, INFO
                elif processed.startswith("output(b)") or processed.startswith("outputb"):
                    choice = "BASELINE" if reverse else "HAF"
                    return choice, res, choice, res, INFO
                else:
                    return "", res, "", res, INFO
            except Exception as e:
                print(e)
                retry_interval *= 2
                time.sleep(retry_interval)
        else:
            return "FAIL", "", "FAIL", "", INFO
    else:
        reverse = True
        v1, v2 = (cpl, base) if reverse else (base, cpl)
        prompt = PROMPT.format(question=question, v1=v1, v2=v2)
        retry_interval = 1
        for _ in range(retry):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": " You are a helpful instruction-following assistant that prints the best model by selecting the best outputs for a given instruction."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=32,
                    temperature=0
                )
            
                rres = response.choices[0].message.content
                processed = ''.join(rres.lower().split())

                # if reverse  a=haf  b=baseline
                if processed.startswith("output(a)") or processed.startswith("outputa"):
                    return "HAF" if reverse else "BASELINE", rres, "HAF" if reverse else "BASELINE", rres, INFO
                elif processed.startswith("output(b)") or processed.startswith("outputb"):
                    return "BASELINE" if reverse else "HAF", rres, "BASELINE" if reverse else "HAF", rres, INFO
                else:
                    return "", rres, "", rres, INFO
            except Exception as e:
                print(e)
                retry_interval *= 2
                time.sleep(retry_interval)
        return "FAIL", "", "FAIL", "", INFO


PROMPT_FOR_RANKING_4 = """I want you to create a leaderboard of different models. \
To do so, I will give you the instructions (prompts) given to the models, and the responses of four models. \
Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{{
    "instruction": {question},
}}

Here are the outputs of the models:
[
    {{
        "model": "model_1",
        "answer": {output_1}
    }},
    {{
        "model": "model_2",
        "answer": {output_2}
    }},
    {{
        "model": "model_3",
        "answer": {output_3}
    }},
    {{
        "model": "model_4",
        "answer": {output_4}
    }}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{"model": "model_1", "rank": <model-rank>}},
    {{"model": "model_2", "rank": <model-rank>}},
    {{"model": "model_3", "rank": <model-rank>}},
    {{"model": "model_4", "rank": <model-rank>}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""


def multi_ranking(question="", responses=[], model="gpt-3.5-turbo-0125"):
    global I
    I += 1
    print(I)
    retry_interval = 1
    assert '3.5' in model, 'gpt-4 is too expensive.'
    idx_mapping = [i for i in range(4)]
    random.shuffle(idx_mapping)
    label2idx_mapping = {j: i for i, j in enumerate(idx_mapping)}

    prompt = PROMPT_FOR_RANKING_4.format(question=question, **{f"output_{j+1}": responses[idx_mapping[j]] for j in range(4)})
    rres = ''
    for _ in range(retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant, that ranks models by the quality of their answers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=128,
                temperature=0
            )
        
            rres = response.choices[0].message.content
        except Exception as e:
            print(e)
            retry_interval *= 2
            time.sleep(retry_interval)
            continue
        try: 
            processed = json.loads(rres)
        except:
            try:
                processed = json.loads('['+rres+']')
            except:
                time.sleep(retry_interval)
                continue
        try:
            if type(processed) is dict and len(processed.keys()) == 1:
                    for _, v in processed.items():
                        data = v
                    processed = data
            label_rank = [0, 0, 0, 0]
            for k in processed:
                assert k["model"] in ["model_1", "model_2", "model_3", "model_4"]
                label_rank[int(k["model"][-1])-1] = int(k["rank"])
            for i in label_rank:
                assert i in [1,2,3,4]
            rank = [label_rank[label2idx_mapping[i]] for i in range(4)]
            return question, responses, rank, rres
        except Exception as e:
            print("=== error ===")
            time.sleep(retry_interval)
            continue
    return question, responses, [], rres


def analyze_rank(question, responses, rank, info, f):
    if rank == []:
        log = {f"response_{i+1}": {"text": responses[i], 'rank': None} for i in range(4)}
        log.update({"info": info, "question": question})
        json.dump(log, f)
        f.write('\n')
    else:
        log = {f"response_{i+1}": {"text": responses[i], 'rank': rank[i]} for i in range(4)}
        log.update({"info": info, "question": question})
        json.dump(log, f)
        f.write('\n')


def multi_thread_rank(ds="alpaca", thread=10, start=0, end=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=thread) as executor, open(f"/root/exp-modeling/data/sampled_data/{ds}.jsonl") as f:
        data = [json.loads(line) for line in f]
        data = [item for item in data if "result_4" in item.keys()]
        if end is None:
            data = data[start:]
        else:
            data = data[start:end]
        futures = []
        for item in data:
            futures.append(executor.submit(multi_ranking, item["input"], [item[f"result_{i+1}"] for i in range(4)]))
    
    pbar = tqdm(as_completed(futures), total=len(futures))
    with open(f"/root/exp-modeling/output/gpt_judge/rank_sampled/{ds}.jsonl", "a") as fw:
        for job in pbar:
            results = job.result() 
            analyze_rank(*results, fw)


def multi_thread_rank_mistral(ds="alpaca", thread=10, start=0, end=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # with ThreadPoolExecutor(max_workers=thread) as executor, open(f"/root/exp-modeling/data/sampled_data_mistral/{ds}.jsonl") as f:
    with ThreadPoolExecutor(max_workers=thread) as executor, open(f"/root/exp-modeling/output/gpt_judge/rank_sampled/tmp/{ds}.jsonl") as f:
        data = [json.loads(line) for line in f]
        data = [item for item in data if "result_4" in item.keys()]
        # data = [item for item in data if "result_8" in item.keys()]
        if end is None:
            data = data[start:]
        else:
            data = data[start:end]
        futures = []
        for item in data:
            # futures.append(executor.submit(multi_ranking, item["input"], [item[f"result_{i}"] for i in range(5, 9)]))
            futures.append(executor.submit(multi_ranking, item["input"], [item[f"result_{i}"] for i in range(1, 5)]))
    
    pbar = tqdm(as_completed(futures), total=len(futures))
    # with open(f"/root/exp-modeling/output/gpt_judge/rank_sampled/tmp/{ds}_59.jsonl", "a") as fw:
    with open(f"/root/exp-modeling/output/gpt_judge/rank_sampled/{ds}_mistral.jsonl", "a") as fw:
        for job in pbar:
            results = job.result() 
            analyze_rank(*results, fw)


def multi_rank_test():
    with open("/root/exp-modeling/data/sampled_data/alpaca.jsonl") as f:
        data = [json.loads(line) for line in f]
    for j in range(3):
        print("======")
        print(j)
        item = data[j]
        print(multi_ranking(item["input"], [item[f"result_{i+1}"] for i in range(4)]))


class Judge:
    def __init__(self, base="", key="", model="gpt-4-turbo-2024-04-09", retry=4):
        self.base = base
        self.key = key
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base)
        self.retry = retry
        self.PROMPT = """Select the output (a) or (b) that best matches the given instruction. Choose your preferred \
output, which can be subjective. Your answer should ONLY contain: Output (a) or Output (b). 
Here's an example:

 # Example:
 ## Instruction:
 Give a description of the following job: "ophthalmologist"

 ## Output (a):
 An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to \
read letters from a chart.

 ## Output (b):
 An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye \
diseases and conditions.

 ## Which is best, Output (a) or Output (b)?
 Output (b)

 Here the answer is Output (b) because it provides a comprehensive and accurate description of \
the job of an ophthalmologist. In contrast, output (a) is more of a joke.

 # Task:
 Now is the real task, do not explain your answer, just say Output (a) or Output (b).

 ## Instruction:
 {question}

 ## Output (a):
 {v1}

 ## Output (b):
 {v2}
 
 ## Which is best, Output (a) or Output (b)?"""

    def gpt_judge(self, question="", cpl="", base="", reverse=False):
        v1, v2 = (cpl, base) if reverse else (base, cpl)
        prompt = f"""Select the output (a) or (b) that best matches the given instruction. Choose your preferred \
output, which can be subjective. Your answer should ONLY contain: Output (a) or Output (b). 
Here's an example:

 # Example:
 ## Instruction:
 Give a description of the following job: "ophthalmologist"

 ## Output (a):
 An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to \
read letters from a chart.

 ## Output (b):
 An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye \
diseases and conditions.

 ## Which is best, Output (a) or Output (b)?
 Output (b)

 Here the answer is Output (b) because it provides a comprehensive and accurate description of \
the job of an ophthalmologist. In contrast, output (a) is more of a joke.

 # Task:
 Now is the real task, do not explain your answer, just say Output (a) or Output (b).

 ## Instruction:
 {question}

 ## Output (a):
 {v1}

 ## Output (b):
 {v2}
 
 ## Which is best, Output (a) or Output (b)?"""
        retry_interval = 1
        for _ in range(self.retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": " You are a helpful instruction-following assistant that prints the best model by selecting the best outputs for a given instruction."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=32,
                    temperature=0
                )
            
                res = response.choices[0].message.content
                processed = ''.join(res.lower().split())

                # if reverse  a=haf  b=baseline
                if processed.startswith("output(a)") or processed.startswith("outputa"):
                    return "HAF" if reverse else "BASELINE", res
                elif processed.startswith("output(b)") or processed.startswith("outputb"):
                    return "BASELINE" if reverse else "HAF", res
                else:
                    return "", res
            except Exception as e:
                print(e)
                retry_interval *= 2
                time.sleep(retry_interval)
        return "FAIL", ""
    
    def bi_judge(self, question="", cpl="", base=""):
        INFO = {"input": question, "haf": cpl, "base": base}
        if cpl == base or (not cpl and not base):
            return "TIE", "", "TIE", "", INFO
        elif not cpl:
            return "BASELINE", "", "BASELINE", "", INFO
        elif not base:
            return "HAF", "", "HAF", "", INFO
        reverse = False
        v1, v2 = (cpl, base) if reverse else (base, cpl)
        
        
        prompt = self.PROMPT.format(question=question, v1=v1, v2=v2)
        retry_interval = 1
        for _ in range(self.retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": " You are a helpful instruction-following assistant that prints the best model by selecting the best outputs for a given instruction."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=32,
                    temperature=0
                )
            
                res = response.choices[0].message.content
                processed = ''.join(res.lower().split())

                # if reverse  a=haf  b=baseline
                if processed.startswith("output(a)") or processed.startswith("outputa"):
                    choice = "HAF" if reverse else "BASELINE"
                elif processed.startswith("output(b)") or processed.startswith("outputb"):
                    choice = "BASELINE" if reverse else "HAF"
                else:
                    choice = ""
                break
            except Exception as e:
                print(e)
                retry_interval *= 2
                time.sleep(retry_interval)
        else:
            choice, res = "FAIL", ""
        time.sleep(1)
        reverse = True
        v1, v2 = (cpl, base) if reverse else (base, cpl)
        prompt = self.PROMPT.format(question=question, v1=v1, v2=v2)
        retry_interval = 1
        for _ in range(self.retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": " You are a helpful instruction-following assistant that prints the best model by selecting the best outputs for a given instruction."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=32,
                    temperature=0
                )
            
                rres = response.choices[0].message.content
                processed = ''.join(rres.lower().split())

                # if reverse  a=haf  b=baseline
                if processed.startswith("output(a)") or processed.startswith("outputa"):
                    return choice, res, "HAF" if reverse else "BASELINE", rres, INFO
                elif processed.startswith("output(b)") or processed.startswith("outputb"):
                    return choice, res, "BASELINE" if reverse else "HAF", rres, INFO
                else:
                    return choice, res, "", rres, INFO
            except Exception as e:
                print(e)
                retry_interval *= 2
                time.sleep(retry_interval)
        return choice, res, "FAIL", "", INFO
        

def GPT_get_result(exp, exp_base, step):
    from pathlib import Path
    fpath = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/c{step}.jsonl"
    fpath_base = f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp_base}/c{step}.jsonl"

    rm = Judge(base="", key="YOUR API KEY")
    Start = 0
    if Path(f"/root/exp-modeling/generation/R_GPT_E{exp}_{exp_base}S{step}.jsonl").exists():
        with open(f"/root/exp-modeling/generation/R_GPT_E{exp}_{exp_base}S{step}.jsonl", "r") as f:
            Start = len(f.readlines())
    with open(fpath) as f, open(fpath_base) as fb, open(f"/root/exp-modeling/generation/R_GPT_E{exp}_{exp_base}S{step}.jsonl", "a") as fout:
        tt = 600
        pbar = tqdm(zip(range(tt), f, fb), total=tt)
        WIN_INFO = {
            "alpaca": {"win": 0, "lose": 0, "tie": 0, "total": 0},
            "beaver": {"win": 0, "lose": 0, "tie": 0, "total": 0},
            "chatbot": {"win": 0, "lose": 0, "tie": 0, "total": 0},
        }
        St = 0
        Sw = 0
        for i, l, lb in pbar:
            if i >= Start:
                item = json.loads(l)
                itembase = json.loads(lb)
                assert item["input"] == itembase["input"]
                if item["response"] == "" and itembase["response"] == "":
                    q1 = q2 = "TIE"
                elif item["response"] == "":
                    q1 = q2 = "BASELINE"
                elif itembase["response"] == "":
                    q1 = q2 = "CPL"
                elif item["response"] == itembase["response"]:
                    q1 = q2 = "TIE"
                else:
                    q1 = rm.gpt_judge(question=item["input"], cpl=item["response"], base=itembase["response"], reverse=True)
                    q2 = rm.gpt_judge(question=item["input"], cpl=item["response"], base=itembase["response"], reverse=False)

                DS = item["idx"].split('-')[0]
                if q1 is None or q2 is None:
                    winner = "ERROR"
                elif q1 not in ["CPL", "BASELINE"]:
                    winner = q1
                elif q2 not in ["CPL", "BASELINE"]:
                    winner = q2
                else:
                    if q1 != q2:
                        winner = "TIE"
                    else:
                        winner = q1
                json.dump({
                    "winner": winner,
                    "from": DS
                }, fout)
                fout.write('\n')

                if winner in ["CPL", "BASELINE", "TIE"]:
                    if winner == "CPL":
                        Sw += 1
                    St += 1
                    key = {"CPL": "win", "BASELINE": "lose", "TIE": "tie"}[winner]
                    WIN_INFO[DS][key] += 1
                    WIN_INFO[DS]["total"] += 1
                    pbar.set_description(f"last: {winner} total: {Sw}/{St}")
                else:
                    pbar.set_description(f"last: Error! total: {Sw}/{St}")
    for ds in ["alpaca", "chatbot", "beaver"]:
        if WIN_INFO[ds]['total'] > 0:
            print(f"{ds}:")
            for K in ["win", "tie", "lose"]:
                print(f" {K}: {WIN_INFO[ds][K]/WIN_INFO[ds]['total']:.3f} ", end=' ')
            print()
    return True


def get_result(exp, exp_base, step, no_tie, verbose=True):
    def pprint(*args, verbose=True, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    WIN_INFO = {
        "alpaca": {"win": 0, "lose": 0, "tie": 0, "total": 0},
        "beaver": {"win": 0, "lose": 0, "tie": 0, "total": 0},
        "chatbot": {"win": 0, "lose": 0, "tie": 0, "total": 0},
    }
    with open(f"/root/exp-modeling/output/checkpoint/NO_REF/phi-2_Exp{exp}/config.txt") as f:
        config = json.load(f)
        IND_key = config["data_path"].split("_")[0]
    with open(f"/root/exp-modeling/generation/R_GPT_E{exp}_{exp_base}S{step}.jsonl") as f:
        for line in f:
            item = json.loads(line)
            if item["winner"] == "CPL":
                WIN_INFO[item["from"]]["win"] += 1
                WIN_INFO[item["from"]]["total"] += 1
            elif item["winner"] == "BASELINE":
                WIN_INFO[item["from"]]["lose"] += 1
                WIN_INFO[item["from"]]["total"] += 1
            elif item["winner"] == "TIE" and not no_tie:
                WIN_INFO[item["from"]]["tie"] += 1
                WIN_INFO[item["from"]]["total"] += 1
    IND = {"win": 0, "lose": 0, "tie": 0, "total": 0}
    OOD = {"win": 0, "lose": 0, "tie": 0, "total": 0}
    for ds in ["alpaca", "chatbot", "beaver"]:
        if WIN_INFO[ds]['total'] > 0:
            pprint(f"{ds}:", verbose=verbose)
            LIST = ["win", "lose"] if no_tie else ["win", "tie", "lose"] 
            for K in LIST:
                pprint(f" {K}: {WIN_INFO[ds][K]/WIN_INFO[ds]['total']:.3f}/{WIN_INFO[ds][K]} ", end=' ', verbose=verbose)
                if ds == IND_key:
                    IND[K] = WIN_INFO[ds][K]
                    IND["total"] += WIN_INFO[ds][K]
                else:
                    OOD[K] += WIN_INFO[ds][K]
                    OOD["total"] += WIN_INFO[ds][K]
            pprint(f" total: {WIN_INFO[ds]['total']}", verbose=verbose)
    return IND, OOD


def get_all_result():
    import glob

    files = glob.glob("/root/exp-modeling/generation/R_GPT*")
    for file in files:
        tmp1, tmp2 = file.rsplit('_', 2)[1:]
        exp = tmp1[1:]
        exp_base = tmp2.split('S')[0]
        step = tmp2.split('S')[1][:-6]
        try:
            IND, OOD = get_result(exp, exp_base, step, True, verbose=False)
        except:
            continue
        print(file)
        print('IND: ', end='')
        LIST = ["win", "lose"]
        for K in LIST:
            print(f" {K}: {IND[K]/IND['total']:.3f}", end=' ')
        print()
        print('OOD: ', end='')
        for K in LIST:
            print(f" {K}: {OOD[K]/OOD['total']:.3f}", end=' ')
        print()
                 

def analyze(choice, res, rchoice, rres, item, fw):
    if choice in ["HAF", "BASELINE"] and rchoice in ["HAF", "BASELINE"]:
        if choice == rchoice:
            json.dump({
                "input": item["input"],
                # "haf": item["haf"],
                # "base": item["base"],
                "judge": choice,
                "base_first": res,
                "haf_first":rres,
            }, fw)
            fw.write('\n')
        else:
            json.dump({
                "input": item["input"],
                # "haf": item["haf"],
                # "base": item["base"],
                "judge": 'TIE',
                "base_first": res,
                "haf_first":rres,
            }, fw)
            fw.write('\n')
    
    elif choice in ["HAF", "BASELINE"]:
        if rchoice == "":
            json.dump({
                "input": item["input"],
                # "haf": item["haf"],
                # "base": item["base"],
                "judge": choice+'-haf_first_illegal',
                "base_first": res,
                "haf_first":rres,
            }, fw)
            fw.write('\n')
        else:
            json.dump({
                "input": item["input"],
                # "haf": item["haf"],
                # "base": item["base"],
                "judge": 'ERROR',
                "base_first": res,
                "haf_first": rchoice,
            }, fw)
            fw.write('\n')
        
    elif rchoice in ["HAF", "BASELINE"]:
        if choice == "":
            json.dump({
                "input": item["input"],
                # "haf": item["haf"],
                # "base": item["base"],
                "judge": rchoice+'-base_first_illegal',
                "base_first": res,
                "haf_first":rres,
            }, fw)
            fw.write('\n')
        else:
            json.dump({
                "input": item["input"],
                # "haf": item["haf"],
                # "base": item["base"],
                "judge": 'ERROR',
                "base_first": choice,
                "haf_first": rres,
            }, fw)
            fw.write('\n')
    # both error
    elif choice and rchoice:
        json.dump({
            "input": item["input"],
            # "haf": item["haf"],
            # "base": item["base"],
            "judge": 'ERROR',
            "base_first": choice,
            "haf_first": rchoice,
        }, fw)
        fw.write('\n')
    elif not choice and not rchoice:
        json.dump({
            "input": item["input"],
            # "haf": item["haf"],
            # "base": item["base"],
            "judge": 'ILLEGAL',
            "base_first": res,
            "haf_first": rres,
        }, fw)
        fw.write('\n')
    else:
        json.dump({
            "input": item["input"],
            # "haf": item["haf"],
            # "base": item["base"],
            "judge": 'ILLEGAL-ERROR',
            "base_first": choice,
            "haf_first": rchoice,
        }, fw)
        fw.write('\n')


def multi_thread(ds="alpaca", thread=3, start=0, end=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=thread) as executor, open(f"output/infer_result/{ds}_mistral.jsonl") as f:
    # with ThreadPoolExecutor(max_workers=thread) as executor, open(f"output/infer_result/rlhf_infer/{ds}.jsonl") as f:
        data = [json.loads(line) for line in f]
        if end is None:
            data = data[start:]
        else:
            data = data[start:end]
        futures = []
        for item in data:
            # print(item["haf"])
            futures.append(executor.submit(bi_judge, item["input"], item["haf"], item["base"]))
    
    pbar = tqdm(as_completed(futures), total=len(futures))
    # with open(f"output/gpt_judge/{ds}_rlhf_mistral.jsonl", "a") as fw:
    with open(f"output/gpt_judge/{ds}_bon_mistral.jsonl", "a") as fw:
        for job in pbar:
            results = job.result() 
            analyze(*results, fw)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp", type=int, default=1)
    # parser.add_argument("--exp_base", type=int, default=2)
    # parser.add_argument("--step", type=int, default=500)
    # parser.add_argument("--read-only", action="store_true")
    # parser.add_argument("--drop-tie", action="store_true")
    # parser.add_argument("--tmp", action="store_true")
    # parser.add_argument("--get-all", action="store_true")
    # args = parser.parse_args()
    # if args.get_all:
    #     get_all_result()
    # elif args.tmp:
    #     tmp_get_two_result()
    # else:
    #     if args.read_only:
    #         get_result(exp=args.exp, exp_base=args.exp_base, step=args.step, no_tie=args.drop_tie)
    #     else:
    #         GPT_get_result(exp=args.exp, exp_base=args.exp_base, step=args.step)


    for name in ["alpaca", "beaver", "chatbot", "helpful", "harmless"]:
        multi_thread(ds=name, end=300, thread=10)
    for name in ["alpaca", "beaver", "chatbot", "helpful", "harmless"]:
        multi_thread(ds=name, start=300, end=600, thread=10)
    for name in ["alpaca", "beaver", "chatbot", "helpful", "harmless"]:
        multi_thread(ds=name, start=600, end=-1, thread=10)

    # for name in ["alpaca", "chatbot", "helpful"]:
    #     multi_thread(ds=name, end=300)
    # for name in ["alpaca", "chatbot", "helpful"]:
    #     multi_thread(ds=name, start=300, end=600)
    # for name in ["alpaca", "chatbot", "helpful"]:
    #     multi_thread(ds=name, start=600, end=-1)

    # for name in ["helpful"]:
    #     multi_thread(ds=name, end=300)
    # for name in ["helpful"]:
    #     multi_thread(ds=name, start=300, end=600)
    # for name in ["helpful"]:
    #     multi_thread(ds=name, start=600, end=-1)

    # for name in ["alpaca", "beaver", "chatbot", "helpful", "harmless"]:
    #     multi_thread_rank_mistral(ds=name, end=300)
    # for name in ["alpaca", "beaver", "chatbot", "helpful", "harmless"]:
    #     multi_thread_rank_mistral(ds=name, start=300, end=600)
    # for name in ["alpaca", "beaver", "chatbot", "helpful", "harmless"]:
    #     multi_thread_rank_mistral(ds=name, start=600, end=-1)
