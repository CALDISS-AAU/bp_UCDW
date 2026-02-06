import pandas as pd
import numpy as np
import os
from os.path import join 
from collections import Counter
import transformers
from sentence_transformers import SentenceTransformer
import bertopic
from bertopic import BERTopic
import topicwizard
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import bertopic
from bertopic import BERTopic
from PIL import Image

# Spacy 
import spacy
from spacy.lang.da import Danish

# PATH AND DIR 
project_dir = '/work/UCDW'
data_dir = join(project_dir, 'data')
workd_dir = join(data_dir, 'work')
tm_data_dir = join(data_dir, 'TM_data')
output_dir = join(project_dir, 'topic_output')
os.makedirs(output_dir,exist_ok=True)
modules_dir = join(project_dir, 'modules')

# data 
bv_df = pd.read_csv(join(tm_data_dir, 'bv_data_chunked_200.csv'))
bv_df['chunked'] = bv_df['chunked'].astype(str)

# Load from directory
# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

# Loading reduced model
topic_model = BERTopic.load("/work/UCDW/models/model/bv_bertopic", embedding_model=embedding_model)

# stopwords
nlp = Danish()
stop_words = list(nlp.Defaults.stop_words)
stop_words.extend(['hej','okay',
'ok','chatten','chat','ja', 'godt',
'nummer','spørgsmål',
'ventetid', 'nej', 'rigtig'
'børnetelefonen','BørneTelefonen','hjælpen',
'hjælp','tak',#'hinanden',
'spurgte', 'hedder',
'[PER]', '[LOC]','siger',
'tænker', 'mening', 'snakke', 'snak', 'snakker',
'dreng', 'pige', 'år', 'over_40_chars', 'over_40', 'loc','årig', 'hey', 
'årig pige', 'år pige'])

for topic_id, words in topic_model.get_topics().items():
    filtered_words = [
        (w, v) for w, v in words if w not in stop_words
    ]


image = Image.open('/work/UCDW/resource/bubbles-png-44346.png')
mask = np.array(image)

def create_combined_wordcloud(model):
    topics = model.get_topics()
    combined = Counter()

    for topic_id, words in topics.items():
        if topic_id == -1:
            continue
        combined.update(dict(words))

    combined_filtered = {k: v for k,v in combined.items() if k not in stop_words}

    wc = WordCloud(
        font_path='/work/UCDW/resource/Revolin/Revolin-Light.otf',
        mask=mask,
        random_state=876,
        #width=mask.shape[2],
        #height=mask.shape[1],
        stopwords = stop_words,
        max_font_size=350,
        background_color="white",
        contour_color='deepbluesky',
        colormap='twilight',
        max_words=1450
        )
    wc.generate_from_frequencies(combined_filtered)

    plt.figure(figsize=(20, 16))
    plt.imshow(wc, interpolation="None")
    plt.axis("off")
    plt.show()
    plt.savefig('/work/UCDW/output/plots/combined_cloud2.png')

create_combined_wordcloud(topic_model)