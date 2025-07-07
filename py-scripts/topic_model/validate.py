import pandas as pd
import numpy as np
import os
from os.path import join 
import transformers
from sentence_transformers import SentenceTransformer
import bertopic
from bertopic import BERTopic
import topicwizard
from topicwizard.compatibility import BERTopicWrapper
from topicwizard.figures import topic_map
from topicwizard.figures import *
from topicwizard.figures import word_map, document_topic_timeline, topic_wordclouds, word_association_barchart
from topicwizard.pipeline import make_topic_pipeline
from umap import UMAP

# PATH AND DIR 
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
tm_data_dir = join(data_dir, 'TM_data_final')
output_dir = join(project_dir, 'topic_output')
os.makedirs(output_dir,exist_ok=True)
modules_dir = join(project_dir, 'modules')


# data 
bv_df = pd.read_csv(join(tm_data_dir, 'bv_data_chunked_200.csv'))
bv_df['chunked'] = bv_df['chunked'].astype(str)

corpus = bv_df['chunked'].tolist()

# Load from directory
# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
topic_model = BERTopic.load("/work/UCDW/models/model/bv_bertopic", embedding_model=embedding_model)

texts = bv_df['chunked'].tolist()
topics, _ = topic_model.transform(texts)
hier_doc_topics = topic_model.hierarchical_topics(texts)

fig = topic_model.visualize_hierarchical_documents(texts, hierarchical_topics=hier_doc_topics)
fig.write_html("hierarchical_before_merge.html")


# Manually merging topics

groups_to_merge = [
    [76,36,3,66],
    [34, 37],
    [19, 58],
    [75,28,89],
    [93, 38, 73,92],
    [56, 12, 21],
    [50, 59], 
    [40, 42, 15, 83],
    [35, 13, 30],
    [78, 44],
    [60, 77, 79],
    [67, 74, 39, 68, 62, 65, 78, 44]
]

topic_model.merge_topics(bv_df['chunked'], topics_to_merge=groups_to_merge)

# Step 1: Transform the documents
topics, _ = topic_model.transform(texts)

# Step 2: Generate hierarchical topics (per-doc)
hierarchical_doc_topics = topic_model.hierarchical_topics(texts)
print("Hierarchical topic distribution:", Counter(hierarchical_doc_topics))

# Step 3: UMAP reduction
reduced_embeddings = UMAP(
    n_neighbors=15, 
    n_components=2, 
    min_dist=0.1, 
    metric="cosine"
).fit_transform(embeddings)

# Step 4: Visualize
fig = topic_model.visualize_hierarchical_documents(
    texts,
    hierarchical_topics=hierarchical_doc_topics,
    reduced_embeddings=reduced_embeddings
)
fig.write_html("hierarchical_docs.html")