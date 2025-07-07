import pandas as pd
import csv

# Folder for final TM data
data_directory = '/work/UCDW/data/TM/'
data_directory_combined = '/work/UCDW/data/TM_combined/'

# Combine each interview into single row for each interview
df_int = pd.read_csv(
        '/work/UCDW/data/all_interviews.csv',
        sep='\t', 
)

df_int_space = df_int.copy()
df_int_newline = df_int.copy()
df_int_space_5plus_words = df_int.copy()
df_int_newline_5plus_words = df_int.copy()

# Limit to 5 words pr sentence or more
df_int_space_5plus_words['word_count'] = df_int_space_5plus_words['text'].str.split().apply(len)
df_int_space_5plus_words = df_int_space_5plus_words[df_int_space_5plus_words['word_count'] >= 5]
df_int_newline_5plus_words['word_count'] = df_int_newline_5plus_words['text'].str.split().apply(len)
df_int_newline_5plus_words = df_int_newline_5plus_words[df_int_newline_5plus_words['word_count'] >= 5]

# Seperated by space
df_int_space['text'] = df_int_space.groupby(['id'])['text'].transform(lambda x : ' '.join(x))
df_int_space = df_int_space.drop_duplicates()

df_int_space.to_csv(
    f'{data_directory}interviews_combined_space.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_int_space_5plus_words = df_int_space_5plus_words[['id', 'text']]
df_int_space_5plus_words['text'] = df_int_space_5plus_words.groupby(['id'])['text'].transform(lambda x : ' '.join(x))
df_int_space_5plus_words = df_int_space_5plus_words.drop_duplicates()

df_int_space_5plus_words.to_csv(
    f'{data_directory}interviews_combined_space_5plus_words.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

# Seperated by newline
df_int_newline['text'] = df_int_newline.groupby(['id'])['text'].transform(lambda x : '\n'.join(x))
df_int_newline = df_int_newline.drop_duplicates()

df_int_newline.to_csv(
    f'{data_directory}interviews_combined_newline.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_int_newline_5plus_words = df_int_newline_5plus_words[['id', 'text']]
df_int_newline_5plus_words['text'] = df_int_newline_5plus_words.groupby(['id'])['text'].transform(lambda x : '\n'.join(x))
df_int_newline_5plus_words = df_int_newline_5plus_words.drop_duplicates()

df_int_newline_5plus_words.to_csv(
    f'{data_directory}interviews_combined_newline_5plus_words.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

# Combine each conversation into single row for each
df_conv = pd.read_csv('/work/UCDW/data/all_convos_TM_data_TAB.csv', sep='\t') 

df_conv = df_conv[['id', 'text']]
df_conv['id'] = df_conv['id'].astype(str)
df_conv['text'] = df_conv['text'].astype(str)

df_conv_space = df_conv.copy()
df_conv_newline = df_conv.copy()
df_conv_space_5plus_words = df_conv.copy()
df_conv_newline_5plus_words = df_conv.copy()

# Limit to 5 words pr sentence or more
df_conv_space_5plus_words['word_count'] = df_conv_space_5plus_words['text'].str.split().apply(len)
df_conv_space_5plus_words = df_conv_space_5plus_words[df_conv_space_5plus_words['word_count'] >= 5]
df_conv_newline_5plus_words['word_count'] = df_conv_newline_5plus_words['text'].str.split().apply(len)
df_conv_newline_5plus_words = df_conv_newline_5plus_words[df_conv_newline_5plus_words['word_count'] >= 5]

# Seperated by space
df_conv_space['text'] = df_conv_space.groupby(['id'])['text'].transform(lambda x : ' '.join(x))
df_conv_space = df_conv_space.drop_duplicates()

df_conv_space.to_csv(
    f'{data_directory}conversations_combined_space.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_conv_space_5plus_words = df_conv_space_5plus_words[['id', 'text']]
df_conv_space_5plus_words['text'] = df_conv_space_5plus_words.groupby(['id'])['text'].transform(lambda x : ' '.join(x))
df_conv_space_5plus_words = df_conv_space_5plus_words.drop_duplicates()

df_conv_space_5plus_words.to_csv(
    f'{data_directory}conversations_combined_space_5plus_words.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

# Seperated by newline
df_conv_newline['text'] = df_conv_newline.groupby(['id'])['text'].transform(lambda x : '\n'.join(x))
df_conv_newline = df_conv_newline.drop_duplicates()

df_conv_newline.to_csv(
    f'{data_directory}conversations_combined_newline.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_conv_newline_5plus_words = df_conv_newline_5plus_words[['id', 'text']]
df_conv_newline_5plus_words['text'] = df_conv_newline_5plus_words.groupby(['id'])['text'].transform(lambda x : '\n'.join(x))
df_conv_newline_5plus_words = df_conv_newline_5plus_words.drop_duplicates()

df_conv_newline_5plus_words.to_csv(
    f'{data_directory}conversations_combined_newline_5plus_words.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

# Combine all data into single dataset
df_all_space = pd.DataFrame()
df_all_newline = pd.DataFrame()
df_all_space_5plus_words = pd.DataFrame()
df_all_newline_5plus_words = pd.DataFrame()

df_all_space = df_all_space._append(df_int_space)
df_all_space = df_all_space._append(df_conv_space)

df_all_space_5plus_words = df_all_space_5plus_words._append(df_int_space_5plus_words)
df_all_space_5plus_words = df_all_space_5plus_words._append(df_conv_space_5plus_words)

df_all_newline = df_all_newline._append(df_int_newline)
df_all_newline = df_all_newline._append(df_conv_newline)

df_all_newline_5plus_words = df_all_newline_5plus_words._append(df_int_newline_5plus_words)
df_all_newline_5plus_words = df_all_newline_5plus_words._append(df_conv_newline_5plus_words)

df_all_space.to_csv(
    f'{data_directory_combined}all_space.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_all_space_5plus_words.to_csv(
    f'{data_directory_combined}all_space_5plus_words.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_all_newline.to_csv(
    f'{data_directory_combined}all_newline.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

df_all_newline_5plus_words.to_csv(
    f'{data_directory_combined}all_newline_5plus_words.csv',
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)

