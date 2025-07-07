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

######## VALIDATE AFTER REDUCING ########

# Loading reduced model
topic_model = BERTopic.load("/work/UCDW/models/model/model_reduced", embedding_model=embedding_model)

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
    [67, 74, 39, 68, 62, 65, 78, 44],
    [63, 17, 80],
    [8, 0, 9],
    [22, 18],
    [81, 82],
    [45, 41, 26],
    [5, 10],
    [27, 7, 6, 14, 53],
    [25, 1, 54]
]

topic_model.merge_topics(bv_df['chunked'], topics_to_merge=groups_to_merge)

# Save reduced model
topic_model.save("/work/UCDW/models/model/model_reduced_manual", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

# Retransforming
final_topics, final_probs = topic_model.transform(bv_df['chunked'])

# Merging back to data
bv_df["topic"] = final_topics
bv_df["topic_prob"] = final_probs

bv_df.to_csv(join(data_dir, 'bv_df_topics.csv'))

fig = topic_model.visualize_topics()
fig.write_html("/work/UCDW/output/plots/topics_overview.html")

fig = topic_model.visualize_barchart(top_n_topics=12)
fig.write_html("/work/UCDW/output/plots/topics_barchart.html")

# Calculating hierarchy for topics
hierarchical_topics = topic_model.hierarchical_topics(bv_df['chunked'])
# Creating hiearchy plot
topic_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

# Saving to folder
topic_hierarchy.write_html('/work/UCDW/output/plots/topic_hierarchy_reduced.html')


# Barchart
fig = topic_model.visualize_barchart(top_n_topics=20)
fig.write_html('/work/UCDW/output/plots/topics_barchart.html')


# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
hierarchy_doc_topic = topic_model.visualize_hierarchical_documents(bv_df['chunked'], hierarchical_topics, reduced_embeddings=reduced_embeddings)


# Saving to folder
hierarchy_doc_topic.write_html('/work/UCDW/output/plots/hierarchy_doc_topic.html')


fig = topic_model.visualize_topics()

fig.write_html('/work/UCDW/output/plots/topics_plot.html')


target_word = "trives"

# Find rows where the word appears
matched_indices = [i for i, text in enumerate(bv_df["chunked"]) if target_word in text]

# View the matched contexts and their embeddings
for idx in matched_indices:
    print(f"\n--- Context {idx} ---")
    print(bv_df["chunked"][idx])
    print(f"Embedding: {embeddings[idx]}")


# UMAP reduction
reduced_embeddings = UMAP(
    n_neighbors=10, 
    n_components=2, 
    min_dist=0.0, 
    metric='cosine'
).fit_transform(embeddings)

# Creating plot
fig = topic_model.visualize_hierarchical_documents(
    bv_df['chunked'],
    hierarchical_topics=hierarchical_topics,
    reduced_embeddings=reduced_embeddings
)

fig.write_html('/work/UCDW/output/plots/hierarchy_docs.html')


texts = bv_df["chunked"].tolist()

# Get per-document topic assignments
doc_topics, _ = topic_model.transform(texts)

# Get hierarchical mappings for *documents*, not just topics
hierarchical_doc_topics = topic_model.hierarchical_topics(texts)

# Now this will work
fig = topic_model.visualize_hierarchical_documents(
    texts,
    hierarchical_topics=hierarchical_doc_topics,
    reduced_embeddings=reduced_embeddings
)
fig.write_html("/work/UCDW/output/plots/hierarchy_docs.html")

fig = topic_model.visualize_hierarchical_documents(
    texts,
    hierarchical_topics=hierarchical_doc_topics
)

# Sanity checks
print("len(texts):", len(bv_df['chunked']))
print("len(embeddings):", len(embeddings))
print("len(hierarchical_topics):", len(hierarchical_topics))