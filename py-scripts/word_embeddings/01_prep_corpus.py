import pandas as pd
import spacy
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
os.makedirs(corpus_dir, exist_ok=True)

## READ DATA
convos_df = pd.read_csv(join(workd_dir, 'bv_convos_all.csv'))

# Load spaCy Danish model
#nlp = spacy.load("da_core_news_sm", disable=["parser", "ner"])
nlp = spacy.load("da_core_news_trf", disable=["parser", "ner"])

## DATA HANDLING
# age as numeric
convos_df['requester_age'] = convos_df['requester_age_name'].str.extract(r'(\d+)', expand = False).astype('float')

# simplified gender
gender_map = {
    'Dreng': 'male',
    'Ung mand': 'male',
    'Pige': 'female',
    'Ung kvinde': 'female',
    'Ukendt': 'other',
    'Ønsker ikke oplyse køn': 'other',
    'Ønsker ikke at oplyse': 'other',
    'Anden kønsidentitet': 'other'
}

convos_df['gender'] = convos_df['requester_gender_name'].replace(gender_map)

## FILTER
convos_use = convos_df.loc[(~convos_df['is_fast_bruger']) & (convos_df['requester_channel'] != 'letter') & convos_df['is_incoming'] & (~convos_df['message'].isna()), :]
convos_use = convos_use.loc[(convos_use['requester_age']) >= 10 & (convos_use['requester_age'] <= 16), :]
convos_use = convos_use.dropna(subset = ['requester_age', 'requester_gender_name', 'message'])


## CORPUS
# concatenate messages from same conversation
convos_grouped_df = convos_use.groupby(['gender', 'conversation_code'])['message'].apply('. '.join).reset_index()

# list of texts per gender
convo_texts_all = convos_grouped_df['message'].tolist()
convo_texts_male = convos_grouped_df.loc[convos_grouped_df['gender'] == 'male', 'message'].tolist()
convo_texts_female = convos_grouped_df.loc[convos_grouped_df['gender'] == 'female', 'message'].tolist()


## TOKENIZE
def tokenizer_pipe(corpus, nlp=nlp, batch_size=50):
    tokenized_corpus = []
    for doc in nlp.pipe(corpus, batch_size=batch_size):
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop and token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'PART']]
        tokenized_corpus.append(tokens)
    return(tokenized_corpus)

# tokenize texts
tokenized_corpus_all = tokenizer_pipe(convo_texts_all)
tokenized_corpus_male = tokenizer_pipe(convo_texts_male)
tokenized_corpus_female = tokenizer_pipe(convo_texts_female)

## STORE CORPUS
with open(join(corpus_dir, 'tokenized_corpus_all.json'), 'w') as f:
    json.dump(tokenized_corpus_all, f)

with open(join(corpus_dir, 'tokenized_corpus_male.json'), 'w') as f:
    json.dump(tokenized_corpus_male, f)

with open(join(corpus_dir, 'tokenized_corpus_female.json'), 'w') as f:
    json.dump(tokenized_corpus_female, f)
