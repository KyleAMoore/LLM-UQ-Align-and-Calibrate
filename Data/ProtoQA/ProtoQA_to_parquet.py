import json
import numpy as np
import pandas as pd

# ── 1. Load the JSONL ───────────────────────────────────────────────
rows = []
with open('proto_qa_train.jsonl', 'r') as f:
    for line in f:
        rows.append(json.loads(line))

# ── 2. Parse each record ────────────────────────────────────────────
def parse_record(record):
    question = record['question']['original'].strip()
    total_responses = record['num']['answers']

    clusters = record['answers']['clusters']

    # Sort clusters by count descending (most popular first)
    sorted_clusters = sorted(
        clusters.values(),
        key=lambda c: c['count'],
        reverse=True
    )

    # Take the first phrasing in each cluster as canonical; strip whitespace
    answers = np.array([c['answers'][0].strip() for c in sorted_clusters], dtype=object)

    # Compute proportions in decimal form, consistent with reference parquets
    answer_ratios = np.array(
        [round(c['count'] / total_responses, 4) for c in sorted_clusters]
    )

    return {
        'Question':      question,
        'Answers':       answers,
        'Answer_Ratios': answer_ratios,
        'Response_Count': total_responses
    }

parsed = [parse_record(r) for r in rows]
df = pd.DataFrame(parsed)

# Remove questions with anomalous answer choice counts.
df = df.loc[df['Answers'].apply(lambda x: 1 < len(x) < 10)]

df.to_parquet('protoqa_data.parquet', index=False)
print(f"Saved {len(df)} rows to train_familyfeud.parquet")
