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

os.makedirs(models_dir, exist_ok=True)

## READ DATA
convos_df = pd.read_csv(join(workd_dir, 'bv_convos_all.csv'))

# Load spaCy Danish model
nlp = spacy.load("da_core_news_sm", disable=["parser", "tagger", "ner"])

## FILTER
convos_use = convos_df.loc[(~convos_df['is_fast_bruger']) & (convos_df['requester_channel'] != 'letter') & convos_df['is_incoming'] & (~convos_df['message'].isna()), :]
# TO-DO: Tilføj filtrering for alder >= 10 & <= 16


# concatenate messages from same conversation
convos_grouped_df = convos_use.groupby('conversation_code')['message'].apply('. '.join).reset_index()

# list of texts
convo_texts = convos_grouped_df['message'].tolist()

# tokenize texts
tokenized_corpus = []
for doc in nlp.pipe(convo_texts, batch_size=50):
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    tokenized_corpus.append(tokens)

# Train Word2Vec model using skip-gram (sg=1)
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=10, sg=1, min_count=5, workers=32)

# Save model
model.save(join(models_dir,"bv_word2vec_test.model"))
