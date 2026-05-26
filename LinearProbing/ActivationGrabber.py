#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[4]:


import os
import csv
import string
import random
import itertools
import gc
from time import time
import pickle
import math
from pathlib import Path
import numpy as np
import ast
import re

import pyarrow
import pyarrow.parquet as pq
import pandas as pd
import transformers as tf
from tqdm import tqdm
import torch
from huggingface_hub import scan_cache_dir, whoami, get_token
from huggingface_hub import login
import datasets


# In[5]:


print(whoami())

base_dir = "/work/projects/bs-wdward43/wdward43/"

my_token = get_token()


# # Data Cleaning

# In[6]:


def find_unicode(df):
    non_ascii = set()
    for cell in df['Question']:
        non_ascii.update(re.findall(r'[^\x00-\xFF]', cell))

    for i, cell in enumerate(df['Answers']):
        for s in cell:
            try:
                non_ascii.update(re.findall(r'[^\x00-\xFF]', s))
            except Exception as e:
                print(s, i)
                raise e
    return non_ascii

def unicode_clean(df):
    '''
        Replaces known unicode characters with ascii equivalents.
    '''
    df_copy = df.copy()
    unicode_to_ascii_map = {
        '–' : '-',
        '—' : '-',
        '’' : '\'',
        '”' : '\"',
        '“' : '\"',
        '…' : '...',
        '‘' : '\'',
    }
    def clean_str(s):
        return ''.join([unicode_to_ascii_map[c] if c in unicode_to_ascii_map else c for c in s ])

    df_copy['Question'] = df_copy['Question'].apply(lambda x: clean_str(x))
    df_copy['Answers'] = df_copy['Answers'].apply(lambda x: [clean_str(a) for a in x])

    return df_copy


# # Model List

# In[7]:


# [(mod_path, mod_name, mod_type, mod_size)]
mistral_models = [
    ('mistralai/Mistral-7B-v0.1',          'Mistral_0.1',      'base',  7   ),
    ('mistralai/Mistral-7B-v0.3',          'Mistral_0.3',      'base',  7   ),
    ('mistralai/Mistral-7B-Instruct-v0.1', 'Mistral_0.1_Ins',  'ins',   7   ),
    ('mistralai/Mistral-7B-Instruct-v0.3', 'Mistral_0.3_Ins',  'ins',   7   ),
]

llama_2_models = [
    ('meta-llama/Llama-2-7b-hf',           'Llama_2_7B',       'base',  7   ),
    ('meta-llama/Llama-2-13b-hf',          'Llama_2_13B',      'base', 13   ),
    ('meta-llama/Llama-2-7b-chat-hf',      'Llama_2_7B_Chat',  'ins' ,  7   ),
    ('meta-llama/Llama-2-13b-chat-hf',     'Llama_2_13B_Chat', 'ins' , 13   ),
]

llama_3_models = [
    ('meta-llama/Llama-3.1-8B',            'Llama_3.1_8B',     'base',  8   ),
    ('meta-llama/Llama-3.1-8B-Instruct',   'Llama_3.1_8B_Ins', 'ins',   8   ),
    ('meta-llama/Llama-3.2-1B',            'Llama_3.2_1B',     'base',  1   ),
    ('meta-llama/Llama-3.2-1B-Instruct',   'Llama_3.2_1B_Ins', 'ins',   1   ),
    ('meta-llama/Llama-3.2-3B',            'Llama_3.2_3B',     'base',  3   ),
    ('meta-llama/Llama-3.2-3B-Instruct',   'Llama_3.2_3B_Ins', 'ins',   3   ),
]

gemma_models = [
    ('google/gemma-2b',                    'Gemma_2B',         'base',  2   ),
    ('google/gemma-2b-it',                 'Gemma_2B_Ins',     'ins',   2   ),
    ('google/gemma-7b',                    'Gemma_7B',         'base',  7   ),
    ('google/gemma-7b-it',                 'Gemma_7B_Ins',     'ins',   7   ),
]

gemma_2_models = [
    ('google/gemma-2-2b',                  'Gemma2_2B',        'base',  2   ),
    ('google/gemma-2-2b-it',               'Gemma2_2B_Ins',    'ins',   2   ),
    ('google/gemma-2-9b',                  'Gemma2_9B',        'base',  9   ),
    ('google/gemma-2-9b-it',               'Gemma2_9B_Ins',    'ins',   9   ),
]

falcon_models = [
    ('tiiuae/falcon3-1B-Base',             'Falcon3_1B',       'base',  1   ),
    ('tiiuae/falcon3-1B-Instruct',         'Falcon3_1B_Ins',   'ins',   1   ),
    ('tiiuae/falcon3-3B-Base',             'Falcon3_3B',       'base',  3   ),
    ('tiiuae/falcon3-3B-Instruct',         'Falcon3_3B_Ins',   'ins',   3   ),
    ('tiiuae/falcon3-7B-Base',             'Falcon3_7B',       'base',  7   ),
    ('tiiuae/falcon3-7B-Instruct',         'Falcon3_7B_Ins',   'ins',   7   ),
    ('tiiuae/falcon3-10B-Base',            'Falcon3_10B',      'base', 10   ),
    ('tiiuae/falcon3-10B-Instruct',        'Falcon3_10B_Ins',  'ins',  10   ),
]

pythia_models = [
    ('EleutherAI/pythia-70m',              'Pythia_70M',       'base',  0.07),
    ('EleutherAI/pythia-160m',             'Pythia_160M',      'base',  0.16),
    ('EleutherAI/pythia-410m',             'Pythia_410M',      'base',  0.41),
    ('EleutherAI/pythia-1b',               'Pythia_1B',        'base',  1   ),
    ('EleutherAI/pythia-1.4b',             'Pythia_1.4B',      'base',  1.4 ),
    ('EleutherAI/pythia-2.8b',             'Pythia_2.8B',      'base',  2.8 ),
    ('EleutherAI/pythia-6.9b',             'Pythia_6.9B',      'base',  6.9 ),
    ('EleutherAI/pythia-12b',              'Pythia_12B',       'base', 12   ),
]
# above families + all_models below to reflect

all_models = (
    # mistral_models +
    # llama_2_models + 
    #llama_3_models +
    #gemma_models + gemma_2_models +
    falcon_models 
    #+ pythia_models
)

sorted_models = list(sorted(all_models, key=lambda x: x[3]))


# In[8]:


updated_all_models = [
#    (mod_path, mod_name, mod_type, mod_size)
#    for mod_path, mod_name, mod_type, mod_size in all_models
#    if "pythia" not in mod_path.lower()
#    and "pythia" not in mod_name.lower()
]

print("Original model count:", len(all_models))
print("Updated model count:", len(updated_all_models))

for model in updated_all_models:
    print(model)


# # Model Auxiliary Functions

# In[9]:


def load_model(model_name, device='cuda', token=None, verbose=False):
    config = tf.AutoConfig.from_pretrained(model_name, token=my_token)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name, token=my_token, use_fast=True)

    tokenizer.pad_token = tokenizer.eos_token

    try:
        transformer = tf.AutoModelForCausalLM.from_pretrained(model_name,
                                                              config=config,
                                                              token=my_token,
                                                              low_cpu_mem_usage=True,
                                                              device_map="auto")
        if verbose:
            print(f'Successfully loaded model: {model_name}')
    except Exception as e:
        if verbose:
            print(f'[ERROR] Failed to load model: {model_name}')
        raise e

    return transformer, tokenizer

def clear_hf_cache():
    hf_cache_info = scan_cache_dir()
    hashes = set()

    for repo in hf_cache_info.repos:
        for revision in repo.revisions:
            hashes.add(revision.commit_hash)

    hf_cache_info.delete_revisions(*hashes).execute()

def find_token_budget(model, tokenizer, sample_texts, min_budget=1024, safety_margin=0.5):
    """
    Empirically determine a safe token budget via binary search.

    Generates synthetic batches of increasing size, runs forward passes,
    and binary searches for the largest budget that doesn't OOM.

    safety_margin: fraction of the found maximum to use (default 0.8 = 80%)
    """
    # Use the median length of real inputs as a representative sequence length,
    # so our probe batches reflect actual padding behaviour
    print('Calculating token budget...')
    lengths = sorted(
        len(tokenizer(t, add_special_tokens=False).input_ids)
        for t in sample_texts
    )
    median_len = lengths[len(lengths) // 2]
    max_len = lengths[-1]

    def probe(token_budget):
        """
        Try a forward pass with a batch that fills the token budget.
        Uses the median sequence length so padding mirrors real workloads.
        """
        batch_size = max(1, token_budget // median_len)
        seq_len = token_budget // batch_size  # actual padded length

        # Build a synthetic batch of the right shape directly as token ids,
        # bypassing the tokenizer entirely for speed
        input_ids = torch.full(
            (batch_size, seq_len),
            fill_value=tokenizer.pad_token_id,
            dtype=torch.long,
            device=model.device
        )
        attention_mask = torch.ones_like(input_ids)

        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()  # ensure the kernel actually ran
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            # Always free memory before the next probe, including on OOM
            del input_ids, attention_mask
            torch.cuda.empty_cache()

    best = min_budget

    if not probe(best):
        raise RuntimeError("Even minimum token budget OOM'd")

    print(f'...Current best: {best}')
    while probe(best * 2):
        best *= 2
        print(f'...Current best: {best}')

    budget = int(best * safety_margin)
    print(f"Max viable token budget: {best} | Using {budget} (safety margin: {safety_margin})")
    print(f"  (median prompt length: {median_len}, max prompt length: {max_len})")
    return budget


# # Data Auxiliary Functions

# In[10]:


def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)

def load_df(path):
    return pd.read_parquet(path)

# Reusable bracket metadata — single source of truth shared with find_bracket_token_span
BRACKET_TYPES = {
    'curly':  ('{', '}'),
    'square': ('[', ']'),
    'paren':  ('(', ')'),
}

def _resolve_bracket(bracket) -> tuple[str, str]:
    """Return (open, close) for a bracket name or a custom (open, close) tuple."""
    if isinstance(bracket, str):
        if bracket not in BRACKET_TYPES:
            raise ValueError(f"Unknown bracket type '{bracket}'. "
                             f"Choose from {list(BRACKET_TYPES)} or pass a (open, close) tuple.")
        return BRACKET_TYPES[bracket]
    return tuple(bracket)

def _make_icl_examples(bracket='curly') -> list[dict]:
    open_b, close_b = _resolve_bracket(bracket)
    return [
        {
            "question":    "What is the chemical symbol for gold?",
            "fr_answer":   f"The chemical symbol for gold is {open_b}Au{close_b}.",
            "mcqa_choices": ["Ag", "Mo", "Au", "Gd"],
            "mcqa_answer":  "C",
        },
        {
            "question":    "In what year did the First World War end?",
            "fr_answer":   f"{open_b}1918{close_b}.",
            "mcqa_choices": ["1939", "1918", "1914", "1945"],
            "mcqa_answer":  "B",
        },
    ]

def _make_fr_system_message(bracket='curly') -> str:
    open_b, close_b = _resolve_bracket(bracket)
    return (
        'Answer each question concisely. '
        f'Wrap the key part of your answer in {open_b}{close_b} brackets. '
        'Do not explain or elaborate beyond the bracketed answer.'
    )

def _make_fr_answer_prompt(bracket='curly') -> str:
    open_b, close_b = _resolve_bracket(bracket)
    return f'Answer in as few words as possible, wrapping the key part in {open_b}{close_b} brackets:'

def _format_fr_block(question: str, bracket='curly') -> str:
    """Format a question into a plain-text FR block (no answer choices)."""
    return f"{question}\n{_make_fr_answer_prompt(bracket)}"

def _format_mcqa_block(question: str, choices: list[str]) -> str:
    """Format a question and choices into a plain-text MCQA block (no answer line)."""
    lines = [f"Question: {question}"]
    for i, choice in enumerate(choices):
        lines.append(f"{string.ascii_uppercase[i]}. {choice}")

    lines.append("Answer:")
    return "\n".join(lines)

def _make_fr_icl_completion_block(bracket='curly') -> str:
    examples = _make_icl_examples(bracket)
    blocks = []
    for ex in examples:
        blocks.append(
            f"{ex['question']}\n"
            f"{_make_fr_answer_prompt(bracket)} {ex['fr_answer']}"
        )
    return "\n\n".join(blocks) + "\n\n"

def _make_mcqa_icl_completion_block(bracket='curly') -> str:
    examples = _make_icl_examples(bracket)
    blocks = []
    for ex in examples:
        question_block = _format_mcqa_block(ex["question"], ex["mcqa_choices"])
        blocks.append(f"{question_block} {ex['mcqa_answer']}")
    return "\n\n".join(blocks) + "\n\n"

def _supports_system_role(tokenizer) -> bool:
    """Return True if the tokenizer's chat template accepts a 'system' role."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return True
    except Exception:
        return False

def format_questions(
    df: pd.DataFrame,
    tokenizer,
    use_chat_template: bool,
    exp_type: str,
    bracket='curly',
) -> pd.DataFrame:
    icl_examples = _make_icl_examples(bracket)

    def format_completion(row, mcqa) -> str:
        if mcqa:
            icl_block      = _make_mcqa_icl_completion_block(bracket)
            question_block = icl_block + _format_mcqa_block(row["Question"], row["Answers"])
        else:
            open_b, _ = _resolve_bracket(bracket)
            icl_block      = _make_fr_icl_completion_block(bracket)
            question_block = icl_block + _format_fr_block(row["Question"], bracket) + ' ' + open_b

        if tokenizer.bos_token is not None:
            return tokenizer.bos_token + question_block
        else:
            return question_block

    def format_chat(row, mcqa) -> str:
        messages = []
        has_system = _supports_system_role(tokenizer)

        if mcqa:
            system_content = "Answer each multiple choice question with only the letter of the correct answer."
            if has_system:
                messages.append({"role": "system", "content": system_content})
            for ex in icl_examples:
                user_content = _format_mcqa_block(ex["question"], ex["mcqa_choices"])
                if not has_system and not messages:  # prepend to first user turn
                    user_content = system_content + "\n\n" + user_content
                messages.append({"role": "user",      "content": user_content})
                messages.append({"role": "assistant", "content": ex["mcqa_answer"]})
            question_block = _format_mcqa_block(row["Question"], row["Answers"])
        else:
            system_content = _make_fr_system_message(bracket)
            if has_system:
                messages.append({"role": "system", "content": system_content})
            for ex in icl_examples:
                user_content = ex["question"]
                if not has_system and not messages:  # prepend to first user turn
                    user_content = system_content + "\n\n" + user_content
                messages.append({"role": "user",      "content": user_content})
                messages.append({"role": "assistant", "content": ex["fr_answer"]})
            question_block = _format_fr_block(row["Question"], bracket)

        messages.append({"role": "user", "content": question_block})

        open_b, _ = _resolve_bracket(bracket)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ) + open_b

    base_format_fn = format_chat if use_chat_template else format_completion

    tmp_df = df.copy()
    if exp_type in ['mcqa', 'both']:
        mcqa_format_fn = lambda x: base_format_fn(x, mcqa=True)
        tmp_df["formatted_question_mcqa"] = df.apply(mcqa_format_fn, axis=1)
    if exp_type in ['fr', 'both']:
        fr_format_fn = lambda x: base_format_fn(x, mcqa=False)
        tmp_df["formatted_question_fr"]   = df.apply(fr_format_fn, axis=1)

    return tmp_df


# # Label Variation Logic

# In[11]:


LABEL_VARIATION_TEMPLATES = [
    "{}",
    "{}.",
    "{} ",
    "{})",
    "{},",
    "{}:",
    " {}",
    "({})",
    "{}。",   # fullwidth period (multilingual models)
]

def _expand_templates(letter, templates):
    variants = []
    for t in templates:
        variants.append(t.format(letter))
        lower = t.format(letter.lower())
        if lower not in variants:
            variants.append(lower)
    return variants

def get_letter_token_map(tokenizer, max_k=26):
    """
    Returns a dict mapping each uppercase letter to a list of token IDs —
    one per single-token variation. Variations that tokenize to more than
    one token for the current tokenizer are silently excluded.
    """
    letters = list(string.ascii_uppercase[:max_k])
    token_map = {}

    for letter in letters:
        candidates = _expand_templates(letter, LABEL_VARIATION_TEMPLATES)
        valid_ids = []
        seen = set()
        for variant in candidates:
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] not in seen:
                valid_ids.append(ids[0])
                seen.add(ids[0])
        if not valid_ids:
            raise ValueError(f"No single-token variation found for letter '{letter}'")
        token_map[letter] = valid_ids  # list of token IDs

    return token_map


# # Experiment Code

# ## Auxiliary Functions

# In[12]:


def next_token_logits(model, tokenizer, ctx):
    tokenized = tokenizer(
        ctx,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**tokenized)

    return outputs.logits[:, -1, :]

def make_token_budget_batches(texts, tokenizer, token_budget):
    lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]

    batches = []
    i = 0
    while i < len(texts):
        base_length = lengths[i]
        batch = [i]
        j = i + 1
        while j < len(texts):
            if lengths[j] != base_length:
                break
            if base_length * (len(batch) + 1) > token_budget:
                break
            batch.append(j)
            j += 1
        batches.append((i, i + len(batch)))
        i += len(batch)
    return batches

def find_bracket_token_span(generated_ids, tokenizer,
                            bracket_precedence=('curly', 'square', 'paren')):
    """
    For each sequence in a batch, find the [start, end) token index range
    corresponding to the content before the first closing bracket, trying
    bracket types in precedence order.

    Because the opening bracket is primed into the input, we only search for
    the closing bracket in the generated tokens. The span always starts at
    token 0 and ends at the token before the closing bracket.

    Returns a list of (start_tok_idx, end_tok_idx) or None per sequence.
    """
    compiled_patterns = []
    for bracket in bracket_precedence:
        _, close_b = _resolve_bracket(bracket)
        pattern = re.compile(re.escape(close_b))
        compiled_patterns.append((close_b, pattern))

    spans = []
    for seq_ids in generated_ids:
        ids_list = seq_ids.tolist()

        # Build per-token character ranges
        char_so_far = 0
        token_char_ranges = []
        for tok_id in ids_list:
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
            token_char_ranges.append((char_so_far, char_so_far + len(tok_str)))
            char_so_far += len(tok_str)

        full_text = tokenizer.decode(ids_list, skip_special_tokens=True)

        span = None
        for close_b, pattern in compiled_patterns:
            m = pattern.search(full_text)
            if m is None:
                continue

            close_char = m.start()

            # Find the last token that ends before the closing bracket
            tok_end = None
            for t_idx, (c_start, c_end) in enumerate(token_char_ranges):
                if c_start < close_char:
                    tok_end = t_idx + 1
                else:
                    break

            if tok_end is not None and tok_end > 0:
                span = (0, tok_end)
                break

        spans.append(span)

    return spans


# ## Metrics Computation

# In[13]:


def compute_entropy(log_probs) -> torch.Tensor:
    probs = torch.exp(log_probs)
    contributions = probs * log_probs
    contributions = torch.nan_to_num(contributions, nan=0.0)
    return -torch.sum(contributions, dim=-1)

def compute_pdf_metrics(log_probs,
                        cloze_token_ids,
                        top_k_values=[5, 10, 25, 50, 100],
                        top_p_values=[0.95, 0.9, 0.75, 0.5],
                        normalize=True):
    """
    cloze_token_ids: list (batch) of list (answer choices) of list (token ID variations).
    e.g. cloze_token_ids[i][j] = [tok_id_for_"A", tok_id_for_"a", tok_id_for_"A.", ...]
    """
    batch_size = log_probs.shape[0]
    results = {}

    # --- Full entropy ---
    vocab_size = log_probs.shape[-1]
    results['mc_total-ent'] = [
        min(1.0, e / (math.log(vocab_size) if normalize else 1))
        for e in compute_entropy(log_probs).tolist()
    ]

    # --- Sort once descending for all top-k and top-p metrics ---
    sorted_log_probs, _ = torch.sort(log_probs, dim=-1, descending=True)
    sorted_probs = torch.exp(sorted_log_probs)

    # --- Top-1 vocab log-prob ---
    results['mc_top-1-prob'] = sorted_log_probs[:, 0].tolist()

    # --- Top-k entropy ---
    for k in top_k_values:
        topk_lp = sorted_log_probs[:, :k]
        topk_lp = topk_lp - torch.logsumexp(topk_lp, dim=-1, keepdim=True)
        results[f'mc_top-k-ent-{k}'] = [
            min(1.0, e / (math.log(k) if normalize else 1))
            for e in compute_entropy(topk_lp).tolist()
        ]

    # --- Top-p entropy and token counts ---
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    for p in top_p_values:
        mask = (cumprobs - sorted_probs) < p

        token_counts = mask.sum(dim=-1)
        results[f'mc_top-p-len-{p}'] = token_counts.tolist()

        masked_lp = sorted_log_probs.masked_fill(~mask, float('-inf'))
        masked_lp = masked_lp - torch.logsumexp(masked_lp, dim=-1, keepdim=True)
        raw = compute_entropy(masked_lp).tolist()
        results[f'mc_top-p-ent-{p}'] = [
            0.0 if n == 1 else
            min(1.0, e / (math.log(n) if normalize else 1))
            for e, n in zip(raw, token_counts.tolist())
        ]

    # --- Per-row cloze metrics ---
    cloze_entropies = []
    top_1_masses = []
    cloze_margins = []
    cloze_proportions = []
    cloze_probs = []

    for i in range(batch_size):
        choice_groups = cloze_token_ids[i]
        n = len(choice_groups)

        summed_log_probs = []
        for variation_ids in choice_groups:
            ids_tensor = torch.tensor(variation_ids, device=log_probs.device)
            lse = torch.logsumexp(log_probs[i, ids_tensor], dim=0)
            summed_log_probs.append(lse)

        lp = torch.stack(summed_log_probs)  # (num_choices,) in log space
        cloze_probs.append(lp.tolist())

        # Cloze margin (log-space: log(p1 - p2))
        sorted_lp = torch.sort(lp, descending=True).values

        # Highest cloze log-prob
        top_1_mass = sorted_lp[0].item()
        top_1_masses.append(top_1_mass)

        # Cloze entropy
        lp_norm = lp - torch.logsumexp(lp, dim=0)
        raw_entropy = compute_entropy(lp_norm.unsqueeze(0)).item()
        normalised = min(1.0, raw_entropy / (math.log(n) if normalize else 1)) if n > 1 else 0.0
        cloze_entropies.append(normalised)

    results['mc_choice-ent'] = cloze_entropies
    results['mc_choice-prob'] = top_1_masses
    results['mc_choice-pdf'] = cloze_probs

    return results

def compute_sequence_metrics(log_probs,
                             gen_mask,
                             bracket_mask=None,
                             top_k_values=[5, 10, 25, 50, 100],
                             top_p_values=[0.95, 0.9, 0.75, 0.5],
                             normalize=True):
    """
    log_probs:  (batch_size, gen_steps, vocab_size) — log-softmax over vocab at each step
    gen_mask:   (batch_size, gen_steps) — True for real tokens, False for post-EOS padding
    bracket_mask:  (batch_size, gen_steps) — True for tokens inside {...}
                   If None, bracketed metrics are filled with None/NaN.
    """
    batch_size, gen_steps, vocab_size = log_probs.shape
    results = {}

    # --- Per-step full-vocab entropy: (batch_size, gen_steps) ---
    step_entropies = compute_entropy(log_probs)  # works on last dim → (batch_size, gen_steps)

    # Mask and average
    masked_ent = step_entropies * gen_mask
    seq_lengths = gen_mask.sum(dim=-1).float()  # (batch_size,)
    norm_denom = math.log(vocab_size) if normalize else 1.0

    results['fr_total-ent'] = (masked_ent.sum(dim=-1) / seq_lengths / norm_denom).tolist()

    # --- First-token entropy ---
    first_step_ent = step_entropies[:, 0]  # (batch_size,)
    results['fr_first-token-ent'] = (first_step_ent / norm_denom).tolist()

    # --- Sort once for top-k and top-p ---
    sorted_log_probs, _ = torch.sort(log_probs, dim=-1, descending=True)
    sorted_probs = torch.exp(sorted_log_probs)

    # --- Top-k entropy: average over sequence ---
    for k in top_k_values:
        topk_lp = sorted_log_probs[:, :, :k]  # (batch_size, gen_steps, k)
        topk_lp = topk_lp - torch.logsumexp(topk_lp, dim=-1, keepdim=True)
        step_topk_ent = compute_entropy(topk_lp)  # (batch_size, gen_steps)

        masked_topk = step_topk_ent * gen_mask
        k_norm = math.log(k) if normalize else 1.0
        results[f'fr_top-k-ent-{k}'] = (masked_topk.sum(dim=-1) / seq_lengths / k_norm).tolist()

    # --- Top-p entropy and token counts ---
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    for p in top_p_values:
        mask = (cumprobs - sorted_probs) < p  # (batch_size, gen_steps, vocab_size)

        # Per-step token counts in the p-nucleus
        step_token_counts = mask.sum(dim=-1)  # (batch_size, gen_steps)

        # Average top-p length over real steps
        masked_counts = step_token_counts.float() * gen_mask
        results[f'fr_top-p-len-{p}'] = (masked_counts.sum(dim=-1) / seq_lengths).tolist()

        # Per-step top-p entropy (raw, unnormalized)
        masked_lp = sorted_log_probs.masked_fill(~mask, float('-inf'))
        masked_lp = masked_lp - torch.logsumexp(masked_lp, dim=-1, keepdim=True)
        step_topp_ent = compute_entropy(masked_lp)  # (batch_size, gen_steps)

        # Zero out steps with nucleus size 1 (entropy is 0) and post-EOS steps
        step_topp_ent = step_topp_ent * (step_token_counts > 1).float() * gen_mask

        if normalize:
            # Sum of log(n_t) over real steps where n_t > 1
            # Steps with n_t = 1 contribute 0 entropy and 0 max entropy, so exclude from denominator
            step_log_n = torch.log(step_token_counts.float().clamp(min=1))
            step_log_n = step_log_n * (step_token_counts > 1).float() * gen_mask
            sum_log_n = step_log_n.sum(dim=-1)  # (batch_size,)

            # Guard against sequences where all steps have nucleus size 1
            safe_denom = sum_log_n.clamp(min=1e-10)
            results[f'fr_top-p-ent-{p}'] = (step_topp_ent.sum(dim=-1) / safe_denom).tolist()
        else:
            results[f'fr_top-p-ent-{p}'] = step_topp_ent.sum(dim=-1).tolist()

    # --- Perplexity: exp(mean negative log-likelihood of chosen tokens) ---
    # The chosen token at each step is the argmax (greedy), so its log-prob is the max
    chosen_log_probs = log_probs.max(dim=-1).values  # (batch_size, gen_steps)
    masked_nll = -chosen_log_probs * gen_mask
    mean_nll = masked_nll.sum(dim=-1) / seq_lengths  # (batch_size,)
    results['fr_ppl'] = torch.exp(mean_nll).tolist()

    # --- Bracketed-subsequence metrics (bfr_ prefix) ---
    if bracket_mask is not None:
        bfr_results = compute_sequence_metrics(
            log_probs,
            bracket_mask,
            bracket_mask=None,   # no further nesting
            top_k_values=top_k_values,
            top_p_values=top_p_values,
            normalize=normalize,
        )
        # Rename keys from fr_ to bfr_
        for k, v in bfr_results.items():
            if k.startswith('bfr_'):
                continue  # already renamed in the recursive call's else-branch; skip
            results[k.replace('fr_', 'bfr_', 1)] = v
    else:
        # No bracket info — fill with NaN placeholders
        nan_row = [float('nan')] * batch_size
        for k in list(results.keys()):
            if k.startswith('fr_') and k != 'fr_model_answer' and k != 'fr_gen_length':
                results[k.replace('fr_', 'bfr_', 1)] = nan_row

    return results


# ## Experiment Body

# In[14]:


def run_mcqa_batch(batch_df,
                   model,
                   tokenizer,
                   letter_groups,
                   normalize=True):
    input_col = 'formatted_question_mcqa'

    answers = batch_df["Answers"].tolist()
    batch = batch_df[input_col].tolist()

    logits = next_token_logits(model, tokenizer, batch)
    log_probs = torch.log_softmax(logits, dim=-1)

    cloze_token_ids = []
    batch_cloze_preds = []
    for i, ans_list in enumerate(answers):
        k = len(ans_list)
        labels = string.ascii_uppercase[:k]
        groups = letter_groups[k - 1]  # list of lists of token IDs

        # Predict: pick the choice with highest summed probability
        summed = []
        for variation_ids in groups:
            ids_t = torch.tensor(variation_ids, device=log_probs.device)
            summed.append(torch.logsumexp(log_probs[i, ids_t], dim=0))
        best = torch.argmax(torch.stack(summed)).item()
        batch_cloze_preds.append(labels[best])
        cloze_token_ids.append(groups)

    batch_metrics = compute_pdf_metrics(
        log_probs,
        cloze_token_ids,
        normalize=normalize,
    )
    batch_metrics['cloze_pred'] = batch_cloze_preds

    return batch_metrics

def run_fr_batch(batch_df,
                 model,
                 tokenizer,
                 normalize=True,
                 max_new_tokens=150,
                 bracket_precedence=('curly', 'square', 'paren')):
    input_col = 'formatted_question_fr'
    batch = batch_df[input_col].tolist()

    tokenized = tokenizer(
        batch,
        return_tensors="pt",
        padding=False,
        add_special_tokens=False,
    ).to(model.device)

    input_len = tokenized.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # outputs.scores is a tuple of (batch_size, vocab_size) tensors, one per generated step
    # Stack into (gen_steps, batch_size, vocab_size) then permute to (batch_size, gen_steps, vocab_size)
    all_scores = torch.stack(outputs.scores, dim=0)  # (gen_steps, batch_size, vocab_size)
    all_logits = all_scores.permute(1, 0, 2)          # (batch_size, gen_steps, vocab_size)

    # Build per-sequence generation mask: True for real tokens, False for post-EOS padding
    generated_ids = outputs.sequences[:, input_len:]   # (batch_size, gen_steps)
    gen_steps = generated_ids.shape[1]
    batch_size = generated_ids.shape[0]

    close_b = _resolve_bracket(bracket_precedence[0])[1]
    close_token_ids = set(
        ids[0] for ids in [
            tokenizer(close_b, add_special_tokens=False).input_ids,
            tokenizer(' ' + close_b, add_special_tokens=False).input_ids,
        ]
        if len(ids) == 1
    )

    eos_id = tokenizer.eos_token_id
    gen_mask = torch.ones(batch_size, gen_steps, dtype=torch.bool, device=generated_ids.device)
    hit_cap = []

    for i in range(batch_size):
        seq = generated_ids[i].tolist()
        # Find whichever comes first: EOS or closing bracket
        cutoff = None
        for j, tok in enumerate(seq):
            if tok == eos_id:
                cutoff = j      # mask from j+1 onward, include EOS
                hit_cap.append(False)
                break
            if tok in close_token_ids:
                cutoff = j      # mask from j+1 onward, include closing bracket
                hit_cap.append(False)
                break
        else:
            hit_cap.append(True)

        if cutoff is not None:
            gen_mask[i, cutoff + 1:] = False

    # Compute log probs: (batch_size, gen_steps, vocab_size)
    log_probs = torch.log_softmax(all_logits, dim=-1)

    # Decode generated text (only real tokens)
    fr_answers = []
    gen_lengths = gen_mask.sum(dim=-1).tolist()
    for i in range(batch_size):
        real_ids = generated_ids[i, :int(gen_lengths[i])]
        fr_answers.append(tokenizer.decode(real_ids, skip_special_tokens=True))

    # Decode generated ids and locate bracketed spans
    bracket_spans = find_bracket_token_span(generated_ids, tokenizer,
                                            bracket_precedence=bracket_precedence)

    bracket_mask = torch.zeros(batch_size, gen_steps, dtype=torch.bool,
                               device=generated_ids.device)
    has_any_bracket = False
    bracket_found    = []
    bracket_inferred = []

    for i, span in enumerate(bracket_spans):
        if span is not None:
            tok_start, tok_end = span
            tok_end = min(tok_end, gen_steps)
            if tok_start < tok_end:
                bracket_mask[i, tok_start:tok_end] = True
                has_any_bracket = True
            bracket_found.append(True)
            bracket_inferred.append(False)
        else:
            # Post-hoc fallback: treat the full real sequence as the bracketed span
            real_len = int(gen_lengths[i])
            if real_len > 0:
                bracket_mask[i, :real_len] = True
                has_any_bracket = True
            bracket_found.append(False)
            bracket_inferred.append(True)

    bracket_mask_arg = bracket_mask if has_any_bracket else None

    batch_metrics = compute_sequence_metrics(
        log_probs,
        gen_mask,
        bracket_mask=bracket_mask_arg,
        normalize=normalize,
    )
    batch_metrics['fr_model_answer']    = fr_answers
    batch_metrics['fr_gen_length']      = gen_lengths
    batch_metrics['fr_hit_cap']         = hit_cap
    batch_metrics['fr_bracket_found']   = bracket_found
    batch_metrics['fr_bracket_inferred'] = bracket_inferred

    return batch_metrics

def _run_single_pass(df,
                     tokenizer,
                     token_budget,
                     sort_col,
                     process_batch_fn,
                     min_token_budget=1024):
    """
    Shared scaffolding: sort by token length, batch, iterate with OOM
    recovery, collect metrics, restore original row order.

    process_batch_fn(batch_df) -> dict of {col_name: list_of_values}
        Must return one value per row in batch_df for every key.
    """
    work_df = df.copy()

    # Sort by prompt length and track original positions
    work_df = work_df.assign(
        _len=work_df[sort_col].apply(
            lambda t: len(tokenizer(t, add_special_tokens=False).input_ids)
        )
    ).sort_values('_len').reset_index(drop=False)

    original_positions = work_df['index'].values
    work_df = work_df.drop(columns=['_len', 'index'])

    texts = work_df[sort_col].tolist()

    metrics_buffer = []
    current_budget = token_budget
    pending = make_token_budget_batches(texts, tokenizer, current_budget)

    with tqdm(total=len(texts)) as pbar:
        while pending:
            start, end = pending.pop(0)
            batch_df = work_df.iloc[start:end]

            # Verify exact length equality (required to avoid padding)
            batch_lengths = [
                len(tokenizer(text, add_special_tokens=False).input_ids)
                for text in batch_df[sort_col].tolist()
            ]
            assert len(set(batch_lengths)) == 1, \
                f"Batch has mismatched lengths: {batch_lengths}"

            try:
                batch_result = process_batch_fn(batch_df)
                metrics_buffer.append(batch_result)
                pbar.update(end - start)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

                new_budget = current_budget // 2
                if new_budget < min_token_budget:
                    raise RuntimeError(
                        f'\033[91mERROR: Out of Memory at minimum budget '
                        f'({min_token_budget}) on rows {start}-{end}\033[0m'
                    )

                print(f'\n\033[93mWARN: OOM on rows {start}-{end}. '
                      f'Halving budget: {current_budget} → {new_budget}\033[0m')
                current_budget = new_budget

                remaining_texts = texts[start:]
                new_batches = make_token_budget_batches(
                    remaining_texts, tokenizer, current_budget
                )
                pending = [(start + s, start + e) for s, e in new_batches]

    # Flatten batch-level metrics into column-level lists
    all_metrics = {
        col: [val for batch in metrics_buffer for val in batch[col]]
        for col in metrics_buffer[0]
    }

    for col, vals in all_metrics.items():
        work_df[col] = vals

    # Restore original row order
    work_df['_original_position'] = original_positions
    work_df = work_df.sort_values('_original_position') \
                      .drop(columns=['_original_position']) \
                      .reset_index(drop=True)

    # Verification
    if not all(work_df[sort_col].values == df[sort_col].values[:len(work_df)]):
        print("⚠️  WARNING: Index restoration may have failed!")
        print(f"Original length: {len(df)}, Current length: {len(work_df)}")

    assert len(work_df) == len(df), \
        f"Row count mismatch: {len(work_df)} != {len(df)}"

    return work_df


def run_mcqa_exp(df, model, tokenizer, token_budget,
                 min_token_budget=1024, normalize=True):

    letter_map = get_letter_token_map(tokenizer)
    letter_groups = []
    for k in range(1, 27):
        groups = [letter_map[c] for c in string.ascii_uppercase[:k]]
        letter_groups.append(groups)

    def process_batch(batch_df):
        return run_mcqa_batch(
            batch_df, model, tokenizer, letter_groups, normalize=normalize,
        )

    result_df = _run_single_pass(
        df, tokenizer, token_budget,
        sort_col='formatted_question_mcqa',
        process_batch_fn=process_batch,
        min_token_budget=min_token_budget,
    )
    return result_df


def run_fr_exp(df,
               model,
               tokenizer,
               token_budget,
               max_new_tokens=150,
               bracket_precedence=('curly', 'square', 'paren'),
               min_token_budget=1024,
               normalize=True):

    def process_batch(batch_df):
        return run_fr_batch(
            batch_df, model, tokenizer,
            normalize=normalize,
            max_new_tokens=max_new_tokens,
            bracket_precedence=bracket_precedence,
        )

    result_df = _run_single_pass(
        df, tokenizer, token_budget,
        sort_col='formatted_question_fr',
        process_batch_fn=process_batch,
        min_token_budget=min_token_budget,
    )
    return result_df


def run_exp(df, model, tokenizer, token_budget, exp_type,
            min_token_budget=1024, normalize=True,
            max_new_tokens=150, bracket_precedence=('curly', 'square', 'paren')):
    assert exp_type in ['mcqa', 'fr', 'both']

    time_start = time()
    common_kwargs = dict(
        model=model, tokenizer=tokenizer, token_budget=token_budget,
        min_token_budget=min_token_budget, normalize=normalize,
    )
    fr_kwargs = dict(max_new_tokens=max_new_tokens, bracket_precedence=bracket_precedence)

    if exp_type == 'mcqa':
        result_df = run_mcqa_exp(df, **common_kwargs)

    elif exp_type == 'fr':
        result_df = run_fr_exp(df, **common_kwargs, **fr_kwargs)

    elif exp_type == 'both':
        mcqa_df = run_mcqa_exp(df, **common_kwargs)
        fr_df   = run_fr_exp(df, **common_kwargs, **fr_kwargs)

        # Merge: keep all columns from mcqa_df, add only the new fr_ columns
        fr_new_cols = [c for c in fr_df.columns if c not in mcqa_df.columns]
        result_df = mcqa_df.copy()
        for col in fr_new_cols:
            result_df[col] = fr_df[col].values

    result_df['exp_time'] = time() - time_start
    return result_df


# # Main Loop

# ## Meta-parameters

# In[ ]:


# Change this to 'cpu' if trying to run on a local non-cuda-enabled machine. This is not recommended.
dev = 'cuda'

dyn_token_budget = False
default_token_budget = 16384  # Ignored if dyn_token_budget = True.
min_token_budget = 1024
max_new_tokens = 100   # generous cap; sequences hitting this are flagged via fr_hit_cap
bracket        = 'curly'
bracket_precedence = ('curly', 'square', 'paren')

# EDIT LOCATIONS HERE AS NECESSARY
base_dir = '/work/projects/bs-wdward43/wdward43/' #Change as necessary
output_filename_format = '{}/{}/{}_{}.parquet'
def format_output_file_name(exp_label, model_name):
    return output_filename_format.format(base_dir, exp_label, exp_label, model_name)


# DATASET BUILDER

# In[ ]:


activation_output_directory = base_dir + "activation_outputs/"

def build_activation_datasets(
    df,
    model,
    tokenizer,
    model_name,
    output_directory,
    text_col = "formatted_question_mcqa",
    target_cols = ("human-ent", "MC_Human_RT_Correct", "MC_Human_RT_Combined"),
    batch_size = 1,
    max_length = None,
    save_targets = True
):
    # This function saves one parquet file per transformer layer.
    # Each file = one layer
    # Each row. = one COANE question
    # Each dim column = final input-token activation dimension

    model_folder = os.path.join(output_directory, model_name)
    os.makedirs(model_folder, exist_ok=True)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size


    dim_cols = [f"dim_{i}" for i in range(hidden_size)]
    writers = {}

    try:
        for start in tqdm(range(0, len(df), batch_size), desc=f"Extracting {model_name}"):

            end = min(start+batch_size, len(df))
            batch_df = df.iloc[start:end].copy()

            texts = batch_df[text_col].tolist()

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False
            )

            seq_len = inputs["input_ids"].shape[1]

            if max_length is not None and seq_len > max_length:
                raise ValueError(
                f"Input length {seq_len} exceeds max_length = {max_length}. "
                "Stopping instead of truncating because truncating would change activations."
                )


            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states = True,
                    use_cache = False
                )

            hidden_states = outputs.hidden_states

            # hidden_states[0] = embedding output
            # hidden_states[1] = after transformer layer 0
            # hidden_states[2] = after transformer layer 1

            transformer_hidden_states = hidden_states[1:]

            attention_mask = inputs["attention_mask"]
            final_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(texts), device = model.device)

            metadata = pd.DataFrame({
                "question_id" : batch_df.index.to_numpy()
            })

            if save_targets:
                for col in target_cols:
                    if col in batch_df.columns:
                        metadata[col] = batch_df[col].to_numpy()

            for layer_idx, layer_hidden in enumerate(transformer_hidden_states):
                final_token_acts = layer_hidden[batch_indices, final_token_indices, :]

                acts_np = final_token_acts.float().cpu().numpy()
                acts_df = pd.DataFrame(acts_np, columns = dim_cols)

                out_df = pd.concat(
                    [metadata.reset_index(drop=True), acts_df], 
                    axis=1
                )

                table = pyarrow.Table.from_pandas(out_df, preserve_index=False)

                if layer_idx not in writers:
                    layer_path = os.path.join(
                        model_folder,
                        f"{model_name}-layer-{layer_idx}.parquet"
                    )

                    writers[layer_idx] = pq.ParquetWriter(
                        layer_path,
                        table.schema
                    )

                writers[layer_idx].write_table(table)

            del outputs, hidden_states, transformer_hidden_states
            torch.cuda.empty_cache()

    finally:
        for writer in writers.values():
            writer.close()

    print(f"Saved activation datasets to: {model_folder}")


# In[ ]:


updated_all_models = [

    (mod_path, mod_name, mod_type, mod_size)
    for mod_path, mod_name, mod_type, mod_size in all_models
    if "pythia" not in mod_path.lower()
    and "pythia" not in mod_name.lower()

]


coane_df_init = load_df(base_dir + 'coane_data.parquet')
coane_df_init = unicode_clean(coane_df_init)


#
#
# This is the part that runs to build activation dataset. Left earlier code for fear of breaking something.
#
#

for mod_path, mod_name, mod_type, mod_size in updated_all_models:
    
    print(f"Starting activation extraction on model {mod_name}")
    random.seed(42)

    exp_model, exp_tknzr = load_model(mod_path, device = dev, token = my_token)

    coane_df = format_questions(
        coane_df_init.copy(),
        exp_tknzr,
        mod_type == 'ins',
        exp_type ='mcqa',
        bracket=bracket
    )

    exp_model.eval()

    try:

        build_activation_datasets(
            df = coane_df,
            model = exp_model,
            tokenizer = exp_tknzr,
            model_name = mod_name,
            output_directory = activation_output_directory,
            text_col = "formatted_question_mcqa",
            target_cols = (
                "human-ent",
                "MC_Human_RT_Correct",
                "MC_Human_RT_Combined"
            ),
            batch_size=1,
            max_length=default_token_budget,
            save_targets=True
        )

        print(f"\033[92mSuccess: Finished activations - {mod_name}\033[0m")

    except Exception as e:
        print(f"\033[91mERROR: FAILED MODEL - {mod_name}\033[0m")
        raise e

    finally:
        if "exp_model" in locals():
            del exp_model

        try:
            torch.cuda.empty_cache()
            clear_hf_cache()

        except:
            pass

        gc.collect()


# ## Actual main loop

# - [X] Try to add an additional step to model free response that encourages bracketing the answers (e.g. {Abraham Lincoln}).
#   - [X]  If doing so, take two versions of every fr uncertainty metric. One should be the full sequence metric and the other the sequence metric calculated exclusively over the tokens within the brackets.
#     - [X]  Possibly remove/substantially increase the max_generation limit so this limit does not poison the full sequence metrics
#     - [ ]  How to handle (if necessary) multiple disjoint brackets?
#   - [X]  Consider adding few shot ICL (for both ins and base) for bracketed free response. Will hopefully have the side benefit of shortening response length and overall speeding up experiment.
#     - [X] Also added to MCQA. ICL examples are shared between both FR and MCQA, with appropriate formatting based on both question type and model type.
# 
# 
# 
# Overall experiment plan:
# 
# * DS1/2: ProtoQA/CamChoice
#   * MCQA only, all non-sequential metrics
#   * cloze-based correctness
# 
# * DS3: Coane
#   * MCQA
#     * non-sequential metrics
#     * cloze-based correctness
#   * FR
#     * sequential metrics
#       * full sequence
#       * bracketed subsequence
#     * semi-auto correctness
#       * auto: check for exact match with bracketed subsequence
#       * manual inspection likely necessary for all non-exact-matches
# 
# 
# Analysis:
# * For all MCQA:
#   - [X] Spearman correlation between each metric and the human uncertainty (response distribution)
#   - [ ] ECE for each model-metric pair
# * For Coane MCQA:
#   - [X] same as DS1 and DS2, but additional comparison against human uncertainty (response time)
#   - [ ] ECE for each model-metric pair
# * For Coane FR:
#   - [X] Spearman correlations: (HRD = human response dist, HRT = human responses time)
#     * HRD vs full sequence metrics
#     * HRD vs bracketed subsequence metrics
#     * HRT vs full sequence metrics
#     * HRT vs bracketed subsequence metrics
#   - [ ] ECE for each model-metric pair
#     * may be infeasible given likely need for manual inspection
#   * Consider using Kendall's etc to compare the uncertainty ordering agreement with Coane MCQA
# 
# * For all:
#   - [ ] Alignment summary across models for each metric. Can use unpaired Wilcoxon over correlations with median (with bootstrapped CI) as summary effect size
#   - [ ] Alignment summary across metrics for each model. Not sure how best to handle this. Can either just look at the most aligned metric or some (linear?) combination of each of the well-aligned metric based on the previous point
#   - [ ] Instruction fine tuned effect analysis
#     - [ ] possibly paired Wilcoxon per metric (pair base model with its instruction fine-tuned counterpart). What would be the effect size? Median difference + bootstrap CI?
#     - [ ] Do the same for ECE?
#   - [ ] Model size analysis?
#     - [ ] Likely just, per metric, correlate the alignment correlations with model size
#     - [ ] Again, do the same for ECE?
# 
# - [ ] Maybe also compute human ECE for Coane and CamChoice for calibration comparison.

# # Notes

# ## Correlation Change Debuggin Notes
# ### Notes:
# * Ruled out (initial, see conclusion) normalization as the culprit. Removing normalization only affects choice-ent correlation noticeably (as mathematically predicted).
# * Claude suggested potential (definitely not confirmed) row-ordering bug. No change after fix, but should keep the update just in case as there's no reason to assume the previous version was more correct. After review, current logic seems valid.
# * batching + padding are both fine. To be safe, I refactored to always have every element of a given batch be the same length in tokens. Technically more scientifically sound by removing a potential confound/interference from padding.
# * measure calcs have been double checked. There were minor (edge-case) errors, but nothing that would explain the systematic correlation loss we're seeing.
# * prompt template change is possible, but there's nothing suspicious about the current template. If this is the culprit, it would point to our previous results being spurious.
# * pdf generation code is simplified alongside batching change to the point of literally just being to grab items logits[:,-1,:] from each batch's model output. Effectively no way for this to be wrong given guarantees against padding.
# * ~batching + padding was previously problematic? Should run a test run without the dynamic batching and without sorting to check.~ Checked and made no significant difference
# 
# ### Conclusion:
# * major issue found in old analysis code: when using denorm for correlation (which was the chart with ~0.8 correlation for many measreus), we always "denormalized" by dividing by log(choice_count), which is ONLY appropriate for choice-ent. This means all top-k and top-p ent measures were almost certaintly being converted into correlating choice_count with unnormalized human entropy (which itself is likely to correlate with choice_count).
# * I think this is the issue. Fixing this substantially reduces the correlations to the point where there is at best moderate correlation when both human and model are unnormalized. No correlation at all seen when both normalized. With a quick test, the same correlation is seen between normalized model with answer choice count. This, as far as I'm concerned, verifies that we were seeing a version of correlating choice count with choice count and the correlation we still see when both are unnormalized is fully explained by the uncertainties being correlated with choice count (not terribly interesting or surprising), but nearly no correlation with human uncertainty (maybe interesting).

# ## TODO List:
#   * Run pipeline with various models.
#   * Add functions that calculate and display important summary info about the datasets (Roper only)
#     * num answer choices per question
#       * Possibly also identify and show other important breakdowns (e.g. binary vs likert vs general multi-option)
#     * num human responses per question
#     * (normalized) entropy over human responses per question (can use scipy.stats.entropy, don't forget to convert to probabilities)
#         * Might be others worth considering (for both here and other analysis). E.g.
#           * Gini coefficients
#           * Herfindahl–Hirschman Index (I don't know much about this one; might be redundant)
#           * percent for top choice
#           * variance (i.e. std^2)
#     * date distribution (either a pdf or cdf)
#     * source distributions? Need to consider the ways this could be efficiently conveyed and overall utility
#     * percent of questions with "National adult" as the Demographic
#       * Maybe manually inspect for meaningful alternative categories that can be effectively combined and communicated
#         * e.g. Minority-population subsets (Black registered voters, Hispanic adults, etc), age subets (>65, 18-40, etc.), etc.
#     * Add to this list as ideas arise
#   * Update analysis scripts with the following
#       * Instruct vs base model analysis
#       * Visualizations that accomodate larger model pool
#       * (Optional) Analysis of new metrics (varentropy, choice-prob, choice-margin)
# 
# 
# ### New dataset found
# * This dataset has response time data for both free answer and MCQA for the same questions, which opens up interesting possibilities.
# * Consider adding to the pipeline to include a free response variation
#     * If doing so, would need to add a column for the free response answer and one new column per uncertainty measure.
#     * Need to consider how to calculate uncertainty for a multitoken sequence.
#         * Initial search suggests that the proper procedure for conditional entropy over a sequence as here (i.e. H(t0), H(t1|t0), H(t2|t0,t1),...) is to use the chain rule of entropy (technically defined as H(A,B,C,D) = H(A)+H(B|A)+H(C|A,B)+H(D|A,B,C)).
#         * If using chain rule of entropy, it should be noted that the proper way to normalize changes here. In general, we can likely still just divide by the maximum possible entropy, which is now sum(log(|t_i|)). For most measures (when outcome space size is fixed - everything except choice entropy and cloze entropy (not meaningful in this context - this simplifies to T*log(|t_0|))
#         * Some suggestions seen that something called entropy rate may also be useful, but I currently don't understand it well enough to determine.
