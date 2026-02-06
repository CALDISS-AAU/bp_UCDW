import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
import os
from os.path import join
import random
import numpy as np
from tqdm import tqdm
import json

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
models_dir = join(project_dir, 'models')
output_dir = join(project_dir, 'output', 'embeddings')
corpus_dir = join(workd_dir, 'we_corpus')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

## READ CORPUS
with open(join(corpus_dir, 'tokenized_corpus_all.json'), 'r') as f:
    tokenized_corpus_all = json.load(f)

with open(join(corpus_dir, 'tokenized_corpus_male.json'), 'r') as f:
    tokenized_corpus_male = json.load(f)

with open(join(corpus_dir, 'tokenized_corpus_female.json'), 'r') as f:
    tokenized_corpus_female = json.load(f)

## KEYWORDS
keyws = [
    'glæde', # happy
    'glad',  # happy
    'elske', # loved
    'passe', # cared for
    'ønske', # wanted
    'tryg', # safe
    'tryghed', # safe
    'sikkerhed', # safe
    'tapper', # brave / courageous
    'mod', # brave / courageous
    'modig', # brave / courageous
    'sund', # healthy
    'sundhed', # healthy
    'tillid', # trusting
    'stole', # trusting
    'fri', # free
    'frihed', # free
    'opmuntre', # encouraged
    'aktiv', # active
    'form', # fit
    'motion', # fit
    'energisk', # energetic
    'bange', # courageous (absence of)
    'håb', # hopeful
    'optimistisk', # hopeful
    'forbundet', # connected
    'forenet', # connected
    'forene', # connected
    'forbinde', # connected
    'synlig', # visible
    'medføle', # empathetic
    'empati', # empathetic
    'høre', # having a voice
    'venner', # friends
    'ven', # friends
    'veninde', # friends
    'kammerat', # friends
    'familie', # family
    'far', # family
    'mor', # family
    'hjemme', # family
    'skole',
    'klasse'
]

## SET SEED FOR BENCH KEYWS
seed_use = 48222

random.seed(seed_use)

## BENCH KEYWS
#bench_kws = random.sample(keyws, 5)
bench_kws = ['glad', 'bange', 'høre', 'ven', 'vendinde', 'hjem', 'skole', 'klasse']

## VALIDATION PIPELINE

def train_word2vec(corpus, vector_size=100, window=10, sg=1, min_count=5, workers=8):
    return Word2Vec(sentences=corpus, vector_size=vector_size, window=window, sg=sg, min_count=min_count, workers=workers)

def get_top_similar_words(model, keywords, topn=10):
    results = {}
    for kw in keywords:
        if kw in model.wv:
            results[kw] = [word for word, _ in model.wv.most_similar(kw, topn=topn)]
        else:
            results[kw] = []
    return results

def jaccard_index(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def cross_validate_word2vec(corpus, keywords, model_label, random_state=seed_use):
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    results = []
    for i, (train_idx, test_idx) in enumerate(tqdm(kf.split(corpus), total=10, desc=f"Training '{model_label}' models for cross-validation...")):
        train_data = [corpus[j] for j in train_idx]
        model = train_word2vec(train_data)
        top_words = get_top_similar_words(model, keywords)
        results.append({
            'model_id': f'{model_label}_{i}',
            'model': model,
            'top_words': top_words
        })
    return results

def collect_all_keywords(results):
    keywords_sets = []
    for result in results:
        all_words = set()
        for kw_words in result['top_words'].values():
            all_words.update(kw_words)
        keywords_sets.append(all_words)
    return keywords_sets

def compute_jaccard_matrix(keyword_sets, model_label):
    size = len(keyword_sets)
    jaccard_matrix = np.zeros((size, size))
    for i in tqdm(range(size), total=10, desc=f"Computing Jaccard indices for '{model_label}' word sets..."):
        for j in range(size):
            if i != j:
                jaccard_matrix[i, j] = jaccard_index(keyword_sets[i], keyword_sets[j])
            else:
                jaccard_matrix[i, j] = 1.0
    return pd.DataFrame(jaccard_matrix)

# Run cross-validation for each corpus
results_all = cross_validate_word2vec(tokenized_corpus_all, bench_kws, 'all')
results_male = cross_validate_word2vec(tokenized_corpus_male, bench_kws, 'male')
results_female = cross_validate_word2vec(tokenized_corpus_female, bench_kws, 'female')

# Construct sets of 50 words for each iteration
sets_all = collect_all_keywords(results_all)
sets_male = collect_all_keywords(results_male)
sets_female = collect_all_keywords(results_female)

# Compute Jaccard similarity matrices
jaccard_all = compute_jaccard_matrix(sets_all, 'all')
jaccard_male = compute_jaccard_matrix(sets_male, 'male')
jaccard_female = compute_jaccard_matrix(sets_female, 'female')

# --- Summarize upper triangle values (excluding diagonal) ---
def summarize_jaccard_matrix(jaccard_df, label):
    print(f"Summarize jaccard indices for '{label}' models...")
    matrix = jaccard_df.values
    upper_triangle_vals = matrix[np.triu_indices_from(matrix, k=1)]
    summary = {
        'model': label,
        'mean_jaccard': np.mean(upper_triangle_vals),
        'std_jaccard': np.std(upper_triangle_vals),
        'values': upper_triangle_vals
    }
    return summary

summary_all = summarize_jaccard_matrix(jaccard_all, 'all')
summary_male = summarize_jaccard_matrix(jaccard_male, 'male')
summary_female = summarize_jaccard_matrix(jaccard_female, 'female')

# --- Combine for plotting ---
summary_data = pd.DataFrame({
    'model': ['all'] * len(summary_all['values']) +
             ['male'] * len(summary_male['values']) +
             ['female'] * len(summary_female['values']),
    'jaccard_index': list(summary_all['values']) +
                     list(summary_male['values']) +
                     list(summary_female['values'])
})
summary_data['bench_keyws'] = ', '.join(bench_kws)

# --- Prepare and save summary CSV ---
print("Save summaries")
summary_df = pd.DataFrame([
    {'Model': summary_all['model'], 'Mean Jaccard': summary_all['mean_jaccard'], 'Std Dev': summary_all['std_jaccard']},
    {'Model': summary_male['model'], 'Mean Jaccard': summary_male['mean_jaccard'], 'Std Dev': summary_male['std_jaccard']},
    {'Model': summary_female['model'], 'Mean Jaccard': summary_female['mean_jaccard'], 'Std Dev': summary_female['std_jaccard']}
])
summary_df['bench_keyws'] = ', '.join(bench_kws)
csv_path = join(output_dir, 'jaccard_summary_statistics.csv')
summary_df.to_csv(csv_path, index=False)
summary_data.to_csv(join(output_dir, 'jaccard_indices.csv'), index=False)
