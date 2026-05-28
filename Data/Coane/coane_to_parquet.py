import json
import pandas as pd
import random


A1 = r"C:\Users\colew\Downloads\School\Research\calibration_cont_Flairs 2026\Coane_dataset\table_A1.csv"
A2 = r"C:\Users\colew\Downloads\School\Research\calibration_cont_Flairs 2026\Coane_dataset\table_A2.csv"

df = pd.DataFrame(columns = ["Question", "Answers", "Answer_Ratios", "Correct_Answer", "Correct_Answer_Text", "FR_Correct_Human", "MC_Correct_Human", "FR_Response_Time", "MC_Response_Time"])
df1 = pd.read_csv(A1)
df2 = pd.read_csv(A2)

answers = []
ratios = []
correct = []
for index, row in df2.iterrows():
    o1 = row["CORRECT ANSWER"]
    o2 = row["MC FOILS  (% of times selected as response) – Foil 1"]
    o3 = row["MC FOILS  (% of times selected as response) – Foil 2"]
    o4 = row["MC FOILS  (% of times selected as response) – Foil 3"]
    options = [o1, o2, o3, o4]
    random.shuffle(options)

    A = 65
    choices = []
    percents = []
    
    for option in options:
        if option == row["CORRECT ANSWER"]:
            correct.append(chr(A))
        A += 1

        #parse the answer choices from the percentages and make two different lists, but keep the ordering the same
        choices.append(option.split('(')[0].strip())
        try:
            percents.append(round((float(option.split('(')[1].split('%')[0]) / 100), 3))
        except IndexError:
            percents.append(row['Correct'])

    answers.append(choices)
    ratios.append(percents)

df["Answers"] = answers
df["Answer_Ratios"] = ratios
df["Correct_Answer"] = correct
df["Question"] = df1["General Knowledge Question"]
df["Correct_Answer_Text"] = df2["CORRECT ANSWER"]
df["FR_Correct_Human"] = df1["CR (proportion of responses) – Correct"]
df["MC_Correct_Human"] = df2["Correct"]
df["FR_Response_Time"] = df1["CR (response times) – Correct"]
df["MC_Response_Time"] = df2["MC  (response times) – Correct"]

df.to_csv("test.csv", index=False)

df.to_parquet("Coane.parquet")