import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)

mismatch_dfs = []
for fn in Path("./Results/coane").glob("*.parquet"):
    if fn.stem == 'coane_data': continue

    trg_df = pd.read_parquet(fn)
    src_df = pd.read_parquet(f'./coane_grading/{fn.name}')

    trg_ans_col = 'fr_model_answer'
    src_ans_col = 'fr_model_answer'

    src_corr_col = 'final_ruling'
    trg_corr_col = 'fr_correct?'

    # sanity check:
    common_idx = src_df.index.intersection(trg_df.index)

    src_vals = src_df.loc[common_idx, src_ans_col]
    trg_vals = trg_df.loc[common_idx, trg_ans_col]

    mismatches = common_idx[src_vals.values != trg_vals.values]
    if len(mismatches) > 0:
        print(fn.name)
        print(f"\tMatching rows: {len(common_idx) - len(mismatches)} / {len(common_idx)}")

        mismatch_dfs.append(pd.DataFrame({
            "model": fn.stem,
            "src": src_df.loc[mismatches, src_ans_col],
            "trg": trg_df.loc[mismatches, trg_ans_col]
        }))
    else:
        trg_df[trg_corr_col] = src_df[src_corr_col]
        trg_df.to_parquet(fn)

if len(mismatch_dfs) > 0:
    print('Mismatch found in:')
    for md in mismatch_dfs:
        print('\t'+md['model'][0])
else:
    print('Successfully inserted correctness in all dataframes')