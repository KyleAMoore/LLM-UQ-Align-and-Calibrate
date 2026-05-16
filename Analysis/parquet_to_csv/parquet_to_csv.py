import pandas as pd

def flatten(lol):
    return [item for sublist in lol for item in sublist]

model_families = [
    ['Llama_2_7B', 'Llama_2_7B_Chat', 'Llama_2_13B', 'Llama_2_13B_Chat'],
    ['Llama_3.2_1B', 'Llama_3.2_1B_Ins', 'Llama_3.2_3B', 'Llama_3.2_3B_Ins', 'Llama_3.1_8B', 'Llama_3.1_8B_Ins'],
    ['Mistral_0.1', 'Mistral_0.1_Ins', 'Mistral_0.3', 'Mistral_0.3_Ins'],
    ['Falcon3_1B', 'Falcon3_1B_Ins', 'Falcon3_3B', 'Falcon3_3B_Ins', 'Falcon3_7B',  'Falcon3_7B_Ins', 'Falcon3_10B', 'Falcon3_10B_Ins'],
    ['Gemma_2B', 'Gemma_2B_Ins', 'Gemma_7B', 'Gemma_7B_Ins'],
    ['Gemma2_2B', 'Gemma2_2B_Ins', 'Gemma2_9B', 'Gemma2_9B_Ins'],
]
model_names = flatten(model_families)
for model_name in model_names:
    tmp_df = pd.read_parquet(f'../Results/coane/coane_{model_name}.parquet')
    new_df = tmp_df[['Question', 'Correct_Answer_Text', 'fr_model_answer']].copy()
    new_df['fr_correct?'] = ''
    new_df.to_csv(f'../coane_csvs/coane_{model_name}.csv')