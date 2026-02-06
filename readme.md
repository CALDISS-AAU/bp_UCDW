# UCDW: Understanding Children's Discourses on Well-Being

## Project description

This repository contains all the code used to create the analyses for the pilot project "Understanding Childrens' Discourses on Well-Being" - a MASSHINE funded project between Children's Welfare and Aalborg University.

The project mainly served as a proof-of-concept and testing ground for the feasibility of using chat and message data from the helpline "BørneTelefonen" provided by Children's Welfare. Mainly, the project was interested in whether this data could be used as an empirical basis for the development of novel well-being indicators.

The repository consists of two main pipelines each with several scripts: one for creating a topic model of the data, and one for creating simple word-embedding model of the data. R and Quarto was used for presentation and reporting.

Please note that this project is not in active development and is meant for archival purposes.

## Pipelines (Python)

**Topic model (BERTopic)**

1. `py-scripts/topic_model/01_prep_data_4_TM.py` filters incoming, non–fast-user chats, cleans text, merges with interviews, and chunks conversations into ~200-character segments for modeling.
2. `py-scripts/topic_model/02_train_model.py` trains a BERTopic model using multilingual-e5-large embeddings, UMAP for dimensionality reduction, HDBSCAN for clustering, and a Danish stopword-augmented CountVectorizer for topic representations.
3. `py-scripts/topic_model/03_reduce_model.py` computes a topic hierarchy, reduces outliers, refits the model with updated topic assignments, and saves the reduced model.
4. `py-scripts/topic_model/04_validate_reduced_model.py` manually merges topic groups, reassigns topics, and exports model visualizations (topic overview, barcharts, hierarchy, and document plots).
5. `py-scripts/topic_model/05_wordcloud.py` builds a combined wordcloud across topics using custom stopwords and a mask image.

**Word embeddings (Word2Vec)**
1. `py-scripts/word_embeddings/01_prep_corpus.py` filters chats (incoming, non–fast-user, non-letter), derives age and gender, lemmatizes Danish text with spaCy, and stores tokenized corpora for all, male, and female groups.
2. `py-scripts/word_embeddings/02_word_embed_simple.py` trains a quick baseline Word2Vec model on unsegmented chats for sanity checks.
3. `py-scripts/word_embeddings/03_train_we_models.py` trains and saves Word2Vec models for all, male, and female corpora.
4. `py-scripts/word_embeddings/05_validate.py` runs 10-fold cross-validation and compares stability via Jaccard overlap of nearest-neighbor keyword sets, saving summary CSVs.
5. `py-scripts/word_embeddings/06_compare_keyvs.py` projects benchmark keywords into 2D with UMAP and exports keyword similarity data for gender comparisons.
