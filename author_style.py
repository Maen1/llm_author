# style_analyzer.py
from collections import Counter
import spacy
import pandas as pd
import textstat
from typing import Dict, Any

nlp = spacy.load("en_core_web_lg")
nlp.max_length = 8000000
corpus_df = pd.read_parquet("data/processed/corpus.parquet")
def analyze_style(text: str) -> Dict[str, Any]:
    doc = nlp(text)
    print(doc)
    return {
        "avg_sentence_length": sum(len(sent) for sent in doc.sents)/len(list(doc.sents)),
        "lexical_diversity": len(set(token.text for token in doc)) / len(doc),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "top_ngrams": {
            "bigrams": Counter([f"{doc[i].text} {doc[i+1].text}" for i in range(len(doc)-1)]).most_common(5),
            "trigrams": Counter([f"{doc[i].text} {doc[i+1].text} {doc[i+2].text}" for i in range(len(doc)-2)]).most_common(3)
        }
    }

# Usage:
sample_size = min(128, len(corpus_df))
sample_text = corpus_df["text"].sample(sample_size).str.cat()
style_report = analyze_style(sample_text[:nlp.max_length])
print(style_report)
