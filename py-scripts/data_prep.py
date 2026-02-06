import os
from os.path import join
import pandas as pd
import json
from itertools import chain
from datetime import datetime
import re

"""
Standardizes conversation data to shared structure and column names.
"""

## PATHS
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
rawd_dir = join(data_dir, 'raw')
workd_dir = join(data_dir, 'work')

## READ DATA
letters_df = pd.read_json(join(rawd_dir, 'Breve_01.01.2025-27.03.2025.json'))
convos_df = pd.read_json(join(rawd_dir, 'Samtaler_01.10.2024-27.03.2025.json'))

ids_filter = pd.read_excel(join(rawd_dir, 'SamtaleID faste brugere.xlsx'))
ids_filter.columns = ['conversation_code']
ids_filter['is_fast_bruger'] = True

## HANDLING LETTERS
letters_df['requester_age_selfreported'] = letters_df['age'].str.replace(' år', '').str.strip().astype('int') # age var
letters_df = letters_df.rename(columns={
    'gender': 'requester_gender_selfreported',
    'letter': 'message'
})
letters_df['requester_channel'] = 'letter'

letters_keep = letters_df[
    ['conversation_code', 'requester_age_selfreported', 'date_received', 
    'requester_gender_selfreported', 'message', 'status', 'requester_channel']]

## HANDLING CONVOS
convos_df = pd.merge(convos_df, ids_filter, on = 'conversation_code', how = 'left') # add faste brugere
convos_df['is_fast_bruger'] = convos_df['is_fast_bruger'].fillna(False) # False for not faste brugere
convos_df['last_contact_dt'] = convos_df['last_contact'].apply(lambda d: d.get('$date') if isinstance(d, dict) else d) # extract date from nested 

# Faste brugere
convos_df.loc[convos_df['is_fast_bruger'], 'conversation_type'] = 'Fast bruger'

# explode and normalize json
convos_long_df = convos_df.explode('messages').reset_index(drop=True)
convos_long_df = pd.merge(convos_long_df, pd.json_normalize(convos_long_df['messages']), left_index=True, right_index=True)

# join with letters
convos_long_df = pd.concat([convos_long_df, letters_keep]).reset_index(drop=True) 
convos_long_df['is_fast_bruger'] = convos_long_df['is_fast_bruger'].fillna(False) # False for not faste brugere
convos_long_df['is_incoming'] = convos_long_df['is_incoming'].fillna(True) # True for letters

# drop column
convos_long_df = convos_long_df.drop(['messages'], axis=1)

## SAVE
convos_long_df.to_csv(join(workd_dir, 'bv_convos_all.csv'), index = False)


## FIND TAGS
tag_re = re.compile(r'(\{.+?\})')

convos_long_df['message'] = convos_long_df['message'].fillna('').astype('str')
convos_long_df['tags'] = convos_long_df['message'].str.findall(tag_re)

tags = convos_long_df.explode(['tags']).drop_duplicates(subset = ['tags'])
tags['tags'].unique()
