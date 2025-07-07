# Packages
import os
from os.path import join, abspath
import sys
import pandas as pd
import json
from itertools import chain
from datetime import datetime
import re
from tqdm import tqdm
import pysbd

tqdm.pandas()

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
modules_dir = abspath(join(project_dir, 'modules'))

# Add modules_dir to sys.path
if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

# Importing functions
from bv_functions import split_text_into_chunks

## READ DATA
all_convos = pd.read_csv(join(workd_dir, 'bv_convos_all.csv'))

# filter dataset
all_convos = all_convos.query('is_incoming == True')
all_convos = all_convos.query('is_fast_bruger == False')
all_convos['message'] = all_convos['message'].astype(str)

# calc word count for each message
all_convos['word_count'] = all_convos['message'].str.split().str.len()
all_convos['word_count'].mean()
all_convos['word_count'].median()

# Removing trailing spaces
def remove_nlines(text):
    return text.replace('\n', '')

# Removing carriage return 
def remove_trailing(text):
    return text.replace('\r', ' ')

# Apply
all_convos['message'] = all_convos['message'].apply(remove_nlines)
all_convos['message'] = all_convos['message'].apply(remove_trailing)

# Dropping columns
all_convos = all_convos.drop(columns = ['requester_channel','last_contact',
'requester_age_name','date_received',
'requester_gender_selfreported','status',
'requester_age_selfreported','conversation_type',
'is_incoming','is_fast_bruger',
'last_contact_dt', 'requester_gender_name'])

# Renaming columns
all_convos = all_convos.rename(columns={'message': 'text', 'conversation_code': 'id'})

# Checking for missing values in ID
df_check = pd.read_csv('/work/UCDW/data/all_convos_TM_data_TAB.csv', sep='\t',dtype={'id': str, 'requester_gender_name': str})
print("Rows read back:", len(df_check))
print("Missing IDs after reload:", df_check['id'].isna().sum())
print("Empty string IDs:", (df_check['id'] == '').sum())
print(df_check[df_check['id'].isna() | (df_check['id'] == '')].head())

# Writing file as line sep
all_convos.to_csv(join(data_dir, 'all_convos_TM_data_TAB.csv'),sep = '\t', index=False)

# Loading interview data
interviews = pd.read_csv(
    join(data_dir,
    'all_interviews.csv'),
    sep = '\t')

# combining data for newline 
df_combined = pd.concat([all_convos, interviews], ignore_index=True)

df_combined.to_csv(join(data_dir, 'combined_df_newline.csv'), sep='\t', index=False)


# Combining data for chunking
# Grouping by ID for both dfs 
df_agg = df_combined.groupby('id').agg({
    'text': lambda x: '. '.join(x.astype(str))
}).reset_index()

# checking word count to determine  chunk size
df_agg['word_count'] = df_agg['text'].str.split().str.len()
df_agg['word_count'].mean()
df_agg['word_count'].median()

# Defining chunk size
min_chars = 200

# Chunking text
df_agg['chunked'] = df_agg['text'].progress_apply(lambda x: split_text_into_chunks(x, min_chars))
df_agg = df_agg.explode('chunked').reset_index(drop=True)

# Create chunk index
df_agg['chunk_index'] = df_agg.groupby('id').cumcount()

df_agg['chunked'] = df_agg['chunked'].astype(str)
df_agg.to_csv(join(data_dir, 'bv_data_chunked_200.csv'), index = False)