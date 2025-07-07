#!/usr/bin/env bash

## installerer python pakker
pip install --upgrade pip # opgraderer pip

pip install -r /work/UCDW/requirements.txt

## spacy
python -m spacy download da_core_news_sm

# run script
python /work/UCDW/py-scripts/word_embed_simple.py