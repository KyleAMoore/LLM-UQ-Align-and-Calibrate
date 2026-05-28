import json
import pandas as pd
import numpy as np


path = r"C:\Users\colew\Downloads\School\Research\calibration_cont_Flairs 2026\Cambridge Multiple-Choice Questions Reading Dataset\Cambridge Multiple-Choice Questions Reading Dataset.jsonl"


# Open the file in read mode ('r')
with open(path, 'r', encoding='utf-8', errors='ignore') as file:
    # Load file content into a variable named 'data'
    data = [json.loads(line) for line in file]

df = pd.DataFrame(columns = ["Question", "Answers", "Answer_Ratios", "Correct_Answer"])

for line in data:
    num = 1
    for question in line.get("questions"):
        ques = line.get("text") + " " + line.get("questions").get(str(num)).get("text")
        correct = line.get("questions").get(str(num)).get("answer")
        
        options = line.get("questions").get(str(num)).get("options")
        answers = []
        ratios = []
        for option in ['a', 'b', 'c', 'd']:
            answers.append(options.get(option).get("text"))
            ratios.append(options.get(option).get("fac"))
    
        df.loc[len(df)] = {'Question': ques, 'Answers': answers, 'Answer_Ratios': ratios, 'Correct_Answer': correct.upper()}
        num += 1

df_camchoice_new = df[~df['Answer_Ratios'].apply(lambda x: any(pd.isna(v) for v in x))]
df_camchoice_new.to_parquet("CamChoice.parquet")
