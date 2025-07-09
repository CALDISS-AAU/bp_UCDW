import pandas as pd
import os
from os.path import join
import random
import numpy as np
import json
from itertools import chain

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
models_dir = join(project_dir, 'models')
output_dir = join(project_dir, 'output', 'embeddings')
corpus_dir = join(workd_dir, 'we_corpus')

## READ CORPUS
with open(join(corpus_dir, 'tokenized_corpus_all.json'), 'r') as f:
    tokenized_corpus_all = json.load(f)

## COMBINE LIST
all_words = list(chain(*tokenized_corpus_all))
all_words_s = pd.Series(all_words)
all_words_count = all_words_s.value_counts().to_frame().reset_index()

all_words_count.loc[all_words_count['index'].str.contains('fri')]
