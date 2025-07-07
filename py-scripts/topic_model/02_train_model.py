# Packages
import os
from os.path import join
import pandas as pd
import csv
import numpy as np
import transformers
from sentence_transformers import  SentenceTransformer

# BERTopic related stuff
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN 
from sklearn.feature_extraction.text import CountVectorizer

# Spacy 
import spacy
from spacy.lang.da import Danish


# PATH AND DIR 
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
tm_data_dir = join(data_dir, 'TM_data_final')
output_dir = join(project_dir, 'topic_output')
os.makedirs(output_dir,exist_ok=True)
modules_dir = join(project_dir, 'modules')


# stopwords
nlp = Danish()
stop_words = list(nlp.Defaults.stop_words)
stop_words.extend(['hej','okay',
'ok','chatten','chat','ja', 'godt',
'nummer','spørgsmål',
'ventetid', 'nej', 'rigtig'
'børnetelefonen','BørneTelefonen',
'hjælp','tak','hinanden',
'spurgte', 'hedder',
'[PER]', '[LOC]','siger',
'tænker', 'mening', 'snakke', 'snak', 'snakker',
'dreng', 'pige', 'år'])

# data 
bv_df = pd.read_csv(join(tm_data_dir, 'bv_data_chunked_200.csv'))
bv_df['chunked'] = bv_df['chunked'].astype(str)

df = df[df["text"].str.strip().astype(bool)]  # remove empty strings
df = df.dropna(subset=["text"])              # remove NaNs

# embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

# Define Umap cluster parameters
umap_model = UMAP(n_neighbors=5, # local (low value) vs global (high value)
n_components=15, # reduce to n dimensions
metric='cosine',
min_dist=0.1, # distance between two clusters - how different are the clusters
low_memory=False,
random_state=420)

# Defining hierarchical density based clustering model
hdbscan_model = HDBSCAN(
    min_cluster_size=40, # how big a cluster is before it is regocnised - how many words before it's recongnised
    cluster_selection_method='leaf',
    metric='euclidean',
    prediction_data=True)

# Define representation model
representation_model = KeyBERTInspired()

# Define CountVectorizer model
vectorizer_model = CountVectorizer(
    stop_words=stop_words, 
    min_df=10, # how often a word needs to be mentioned in a single cluster before it's considered relevant
    max_df=0.5, # how often a word needs to be mentioned acress clusters before it's considered relevant 
    ngram_range=(1, 1)) # how many words we want for each topic - e.g. (1,2) is one or two words

# Iniate model
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    top_n_words=15,
    verbose=True)

# Run model on text column
topics, probs = topic_model.fit_transform(bv_df['chunked'])

# Add topics, probs to data
data_topics = topic_model.get_document_info(bv_df['chunked'], bv_df)
data_topics['topic_prob'] = probs

# Topic info
topics_info = topic_model.get_topic_info()

# Save model
topic_model.save("/work/UCDW/models/model/bv_bertopic_200char", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)