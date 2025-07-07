import pandas as pd
import spacy
from gensim.models import Word2Vec
import os
from os.path import join

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
models_dir = join(project_dir, 'models')

## READ MODEL
model = Word2Vec.load(join(models_dir, "bv_word2vec_test.model"))

## Similarities
model.wv.most_similar("skole", topn = 50)