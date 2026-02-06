import json
import pandas as pd
import collections
import os
from os.path import join

topics_words = []
data_dir = '/work/UCDW/data/for_presentation_200chr_chuncks'
os.makedirs(data_dir,exist_ok=True)

try:
    with open('/work/UCDW/topic_output_200chunk/topics.json', 'r', encoding='utf-8') as topics_file:
        data = json.load(topics_file)
except Exception as e:
    print(f'File could not be opened: {e}')

### Get word count
topic_reprecentations = data['topic_representations']

for topic_id, repr_list in topic_reprecentations.items():
    for word, score in repr_list:
        topics_words.append(word)

word_count = collections.Counter(topics_words)

df_wc = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
df_wc = df_wc.rename(columns={'index':'word', 0:'count'})

df_wc.to_csv(join(data_dir, 'word_count.csv'))


### Get topic_labels to csv
topic_labels = data['topic_labels']

df_tl = pd.DataFrame.from_dict(topic_labels, orient='index').reset_index()
df_tl = df_tl.rename(columns={'index':'topic number', 0:'label'})

df_tr = pd.DataFrame.from_dict(topic_reprecentations, orient='index').reset_index()
for col in df_tr.columns:
    # If every cell in col is a list, this takes the first item in that list:
    df_tr[col] = df_tr[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
df_tr = df_tr.rename(columns={'index':'topic number'})

df_tlr = pd.merge(df_tl, df_tr, on="topic number", how="inner")
df_tlr.to_csv(join(data_dir, 'topics.csv'))


### Combine ID and topics
try:
    with open('/work/UCDW/bv_data_chunked_200.csv', 'r', encoding='utf-8') as original_file:
        df_OG_data = pd.read_csv(original_file)
except Exception as e:
    print(f'File could not be opened: {e}')

topics = data['topics']

df_OG_data['topic'] = topics
df_OG_data.to_csv(join(data_dir, 'OG_data_plus_topics.csv'))