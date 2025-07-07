import re

# Segmentation function
def split_text_into_chunks(text, min_chars):
    from pysbd import Segmenter
    segmenter = Segmenter(language="da", clean=True)
    sentences = segmenter.segment(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > min_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        last_chunk = " ".join(current_chunk)
        if len(last_chunk) < min_chars and chunks:
            chunks[-1] += " " + last_chunk
        else:
            chunks.append(last_chunk)
    return chunks
