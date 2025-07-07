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

# Load from directory
# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
topic_model = BERTopic.load("/work/UCDW/models/model/bv_bertopic", embedding_model=embedding_model)

# Calculating hierarchy for topics
hierarchical_topics = topic_model.hierarchical_topics(bv_df['chunked'])
# Creating hiearchy plot
topic_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

# Saving to folder
topic_hierarchy.write_html('/work/UCDW/output/plots/topic_hierarchy.html')


# Transforming docs to topics
topics, probs = topic_model.transform(bv_df['chunked'])

# Reducing topics and creating new ones
new_topics = topic_model.reduce_outliers(bv_df['chunked'], topics, strategy="c-tf-idf", threshold=0.1)

# Checking for outliers
print("Outliers before:", np.sum(np.array(topics) == -1))
print("Outliers after :", np.sum(np.array(new_topics) == -1))


topic_model.topics_ = new_topics

embeddings = embedding_model.encode(bv_df['chunked'].tolist(), show_progress_bar=True)

# update model
topic_model.fit(bv_df['chunked'], embeddings=embeddings, y=new_topics)
 
topic_model.save("/work/UCDW/models/model/model_reduced", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)