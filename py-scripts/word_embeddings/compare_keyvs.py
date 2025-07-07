import pandas as pd
from gensim.models import Word2Vec
import os
from os.path import join
import random
import umap
import numpy as np

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
bench_kws = ['glad', 'bange', 'høre', 'ven', 'veninde', 'hjem', 'skole', 'klasse']

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
    for keyword in keyws:
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

## Plot using plotnine
#p = (
#    ggplot(df, aes(x='x', y='y')) +
#    geom_point(aes(color='model'), size=3, alpha=0.8) +
#    geom_text(aes(label='word'), size=8, va='bottom', ha='right') +
#    geom_segment(
#        data=df[df['model'] != 'Keyword'],
#        mapping=aes(x='x', y='y', xend='xend', yend='yend', group='word'),
#        color='gray', linetype='dashed', size=0.4
#    ) +
#    theme_minimal() +
#    ggtitle("Nearest Neighbors by Embedding Model") +
#    theme(figure_size=(12, 10))
#)
#
## Prepare lines from similar words to keywords (per model)
#keywords_df = df[df['is_keyword']].set_index(['keyword', 'model'])[['x', 'y']]
#df = df.merge(keywords_df, left_on=['keyword', 'model'], right_index=True, suffixes=('', '_keyword'))
#df['xend'] = df['x_keyword']
#df['yend'] = df['y_keyword']
#
## Re-plot with updated df
#
#adjust_text_dict = {
#    'expand_points': (2, 2),
#    'arrowprops': {
#        'arrowstyle': '->',
#        'color': 'black'
#    }
#}
#
#p = (
#    ggplot(df, aes(x='x', y='y')) +
#    geom_point(aes(color='model', shape='is_keyword'), size=4, alpha=0.9) +
#    geom_text(aes(label='word'), size=8, va='bottom', ha='right', adjust_text = adjust_text_dict) +
#    #geom_segment(
#    #    data=df[df['is_keyword'] == False],
#    #    mapping=aes(x='x', y='y', xend='xend', yend='yend'),
#    #    color='gray', linetype='dashed', size=0.3
#    #) +
#    scale_color_manual(values={
#        'model_m': '#1f77b4',
#        'model_f': '#2ca02c',
#        'model_a': '#d62728',
#        'Keyword': 'black'
#    }) +
#    scale_shape_manual(values={True: 'o', False: 'x'}) +
#    theme_minimal() +
#    ggtitle("Nearest Neighbors by Embedding Model") +
#    theme(figure_size=(12, 10))
#)
#
## store plot
#ggsave(p, filename=join(output_dir, "embedding_neighbors.png"), dpi=300)


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

## Plot
#p = (
#    ggplot() +
#    #geom_point(aes(color='model', shape='is_keyword'), size=4, alpha=0.9) +
#    geom_text(
#        data=df[df['is_keyword'] == True],
#        mapping=aes(label='word', x='x', y='y'),
#        size=8,
#        va='bottom') + 
#    geom_text(
#        data=df[df['is_keyword'] == False],
#        mapping=aes(label='word', x='x', y='y', color='model'), 
#        size=8, 
#        va='bottom', 
#        adjust_text = adjust_text_dict) + 
#    scale_color_manual(values={
#        'model_f': '#2ca02c',
#        'model_m': '#d62728',
#        'keyword': 'black'
#    }) +
#    scale_shape_manual(values={True: 'o', False: 'x'}) +
#    geom_vline(xintercept=0, linetype='dashed', color='gray') +
#    theme_minimal() +
#    ggtitle("Cosine Distance of Nearest Neighbors (model_f: left, model_a: right)") +
#    theme(figure_size=(14, 8))
#)
#
## Save
#ggsave(p, filename=join(output_dir, "embedding_cosine_distance-2.png"), dpi=300)