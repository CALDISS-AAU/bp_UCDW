from spire.doc import *
from spire.doc.common import *
import pandas as pd
import csv
import re

"""
Converts interview transcripts in docx to csv
"""

interview_directories = [
    '/work/UCDW/data/Interview1.docx',
    '/work/UCDW/data/Interview2.docx',
    '/work/UCDW/data/Interview3.docx',
    '/work/UCDW/data/Interview4.docx',
    '/work/UCDW/data/Interview5.docx'
]

# names = ['I', 'Interviewer'] # For the interviewer
names = [
    # names of speakers to confirm transcribed speech - not meta-description (names censored)
]

pattern = (
    r'^(?:' 
        r'\(\d{2}[:.]\d{2}\)\s*'      # e.g. "(00:00)" or "(00.00)" + any spaces
    r')?'                             # make the whole timestamp part optional
    r'(?:' + '|'.join(map(re.escape, names)) + r'):'  # then "A:" or "B:" or "C:"
)

i = 1
df_combined = pd.DataFrame()
for interview_docx in interview_directories:
    # Create a Document object
    document = Document()

    # Load a Word file from disk
    document.LoadFromFile(interview_docx)

    # Split filename to remove '.docx' and create '.txt' and '.csv'
    interview = interview_docx.rsplit('.',1)[0]
    interview_txt = f'{interview}/only_text.txt'
    interview_csv = f'{interview}/only_text.csv'
    interview_csv_plus = f'{interview}/text_and_more.csv'
    interview_csv_final = f'{interview}/final.csv'
    interview_csv_combined = f'/work/UCDW/data/all_interviews.csv'

    # Creates ID
    ID = f'INT{i}'
    i += 1

    # Save the Word file in txt format
    document.SaveToFile(interview_txt, FileFormat.Txt)
    document.Close()

    # Converts txt file to csv split on tab
    with open(interview_txt, 'r') as infile, open(interview_csv, 'w', newline='') as outfile:
        writer = csv.writer(
            outfile,
            delimiter='\t',        # use tab instead of comma
            quoting=csv.QUOTE_NONE, # never wrap content in ""
            escapechar='\\'        # just in case your text ever has a tab
        )
        for raw in infile:
            text = raw.strip()
            if text:
                writer.writerow([text])

    # Opens csv as dataframe, specifying tab‐separator
    df = pd.read_csv(
        interview_csv,
        sep='\t', 
        header=None, 
        names=['text']
    )

    # Adds ID column
    df['id'] = ID

    # Makes ID comes first
    df = df[['id','text']]

    # Only true if spoken by a child or interviewer
    df['speech'] = df['text'].str.match(pattern)

    # Finds number of words and adding this as a column
    df['word_count'] = df['text'].str.split().apply(len)

    # Saving the csv (still tab‐delimited, no quotes)
    df.to_csv(
        interview_csv_plus,
        sep='\t',
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar='\\'
    )

    # Only answers given by children and interviewer
    answers_df = df[df['speech'] == True]
    # Remove 'meta'-rows
    meta_pattern = r'(Interviewer: Rasmus( Thastum)?)'
    mask = ~answers_df['text'].str.contains(meta_pattern, regex=True, na=False)
    answers_df = answers_df[mask].reset_index(drop=True)
    # Removeing childrens names from text
    answers_df['text'] = answers_df['text'].str.replace(pattern, '', regex=True)

    # # Only 5 words and fewer
    # answers_df = answers_df[answers_df['word_count'] >= 6]

    # print(f'Average words: {answers_df['word_count'].mean()}, median words: {answers_df['word_count'].median()}')

    # Saves final sorted data
    answers_df.to_csv(
        interview_csv_final,
        sep='\t',
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar='\\'
    )

    df_combined = df_combined._append(answers_df[['id', 'text']], ignore_index=True)

# Saves combined dataset for all interviews
df_combined.to_csv(
    interview_csv_combined,
    sep='\t',
    index=False,
    quoting=csv.QUOTE_NONE,
    escapechar='\\'
)
