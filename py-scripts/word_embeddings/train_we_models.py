from gensim.models import Word2Vec
import os
from os.path import join
import json

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
models_dir = join(project_dir, 'models')
corpus_dir = join(workd_dir, 'we_corpus')

os.makedirs(models_dir, exist_ok=True)

# cores
num_cores = os.cpu_count()

## READ CORPUS
with open(join(corpus_dir, 'tokenized_corpus_all.json'), 'r') as f:
    tokenized_corpus_all = json.load(f)

with open(join(corpus_dir, 'tokenized_corpus_male.json'), 'r') as f:
    tokenized_corpus_male = json.load(f)

with open(join(corpus_dir, 'tokenized_corpus_female.json'), 'r') as f:
    tokenized_corpus_female = json.load(f)


## TRAIN MODELS
# Train Word2Vec model using skip-gram (sg=1)
model_a = Word2Vec(sentences=tokenized_corpus_all, vector_size=100, window=10, sg=1, min_count=5, workers=8)
model_m = Word2Vec(sentences=tokenized_corpus_male, vector_size=100, window=10, sg=1, min_count=5, workers=8)
model_f = Word2Vec(sentences=tokenized_corpus_female, vector_size=100, window=10, sg=1, min_count=5, workers=8)

# Save models
model_a.save(join(models_dir, "bv_word2vec_all.model"))
model_m.save(join(models_dir, "bv_word2vec_male.model"))
model_f.save(join(models_dir, "bv_word2vec_female.model"))
