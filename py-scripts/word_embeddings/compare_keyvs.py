import pandas as pd
from gensim.models import Word2Vec
import os
from os.path import join
import random
import umap
import numpy as np

"""
Based on trained static embeddings (word2vec), derive main semantic associations across gender. 
Output: output/embeddings/keyword_projections.csv - Project defined keywords to 2D UMAP across genders
Output: output/embeddings/plotting_data_keywords_compare_gender_network.csv - Based on defined keywords, derive top 10 most similar words across genders
"""

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
models_dir = join(project_dir, 'models')
output_dir = join(project_dir, 'output', 'embeddings')
corpus_dir = join(workd_dir, 'we_corpus')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


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
seed_use = 45201

random.seed(seed_use)

## BENCH KEYWS
#bench_kws = random.sample(keyws, 10)
#bench_kws = ['glad', 'bange', 'høre', 'ven', 'veninde', 'hjem', 'skole', 'klasse']
bench_kws = ['skole', 'klasse', 'lærer', 'forælder', 'far', 'mor', 'hjem', 'hjemme', 'kæreste', 'ven', 'veninde']

## READ MODELS
# Paths
model_paths = {
    'model_m': join(models_dir, "bv_word2vec_male.model"),
    'model_f': join(models_dir, "bv_word2vec_female.model"),
    'model_a': join(models_dir, "bv_word2vec_all.model")
}

# Load models
models = {name: Word2Vec.load(path) for name, path in model_paths.items()}

# DataFrame to collect vectors and metadata
all_vectors = []

# Process each model
for model_name, model in models.items():
    for keyword in bench_kws:
        if keyword not in model.wv:
            continue
        # Keyword vectors
        all_vectors.append({
            'keyword': keyword,
            'model': model_name,
            'vector': model.wv[keyword]
        })

# Convert to DataFrame
df = pd.DataFrame(all_vectors)

# Project embeddings to 2D using UMAP
reducer = umap.UMAP(n_components=2, random_state=seed_use)
df[['x', 'y']] = pd.DataFrame(reducer.fit_transform(df['vector'].tolist()), index=df.index)

# Store as csv
df[['keyword', 'model', 'x', 'y']].to_csv(join(output_dir, 'keyword_projections.csv'), index=False)

## V2
models_to_use = ['model_f', 'model_m']

# Create filtered model dict
models_use = {name: models[name] for name in models_to_use}

keyword_y_positions = {kw: i for i, kw in enumerate(bench_kws)}

# Collect plotting data
plot_data = []

for keyword in bench_kws:
    plot_data.append({
        'word': keyword,
        'keyword': keyword,
        'model': 'keyword',
        'x': 0.0,
        'y': keyword_y_positions[keyword] + np.random.uniform(-0.2, 0.2),
        'is_keyword': True
    })

# Add similar words from each selected model
for model_name, model in models_use.items():
    for keyword in bench_kws:
        if keyword not in model.wv:
            continue
        similar_words = model.wv.most_similar(keyword, topn=10)
        for similar_word, similarity in similar_words:
            if similar_word in bench_kws:
                continue
            dist = 1 - similarity
            x = -dist if model_name == 'model_f' else dist
            plot_data.append({
                'word': similar_word,
                'keyword': keyword,
                'model': model_name,
                'x': x,
                'y': keyword_y_positions[keyword] + np.random.uniform(-0.2, 0.2),
                'is_keyword': False
            })


# Create DataFrame
df = pd.DataFrame(plot_data)

# Save
df.to_csv(join(output_dir, 'plotting_data_keywords_compare_gender.csv'), index=False)


# SIMILAR KEYWORDS BASED ON THRESHOLD
models_to_use = ['model_f', 'model_m']

# Create filtered model dict
models_use = {name: models[name] for name in models_to_use}

# Collect plotting data
plot_data = []

for keyword in bench_kws:
    plot_data.append({
        'word': keyword,
        'keyword': keyword,
        'model': 'keyword',
        'similarity': 0.0,
        'is_keyword': True
    })

# Add similar words from each selected model
for model_name, model in models_use.items():
    for keyword in bench_kws:
        if keyword not in model.wv:
            continue
        similar_words = model.wv.most_similar(keyword, topn=10)
        for similar_word, similarity in similar_words:
            plot_data.append({
                'word': similar_word,
                'keyword': keyword,
                'model': model_name,
                'similarity': similarity,
                'is_keyword': False
            })

# Create DataFrame
df = pd.DataFrame(plot_data)

# Save
df.to_csv(join(output_dir, 'plotting_data_keywords_compare_gender_network.csv'), index=False)