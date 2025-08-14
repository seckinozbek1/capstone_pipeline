import pandas as pd
from keybert import KeyBERT
from itertools import chain
import json
import os

# Load data
df = pd.read_csv("Q154_mmr_selected_labeled_combined.csv")

valid_labels = ['Q154_1', 'Q154_2', 'Q154_3', 'Q154_4']

MAX_PHRASES = 3
KEYPHRASE_NGRAM_RANGE = (5, 7)
DIVERSITY = 0.3

# Initialize KeyBERT model (default: 'all-MiniLM-L6-v2')
kw_model = KeyBERT()

results = {}

for label in valid_labels:
    resp_texts = df.loc[df['combined_label'] == label, 'response_hypothesis'].dropna().unique()
    adapted_texts = df.loc[df['combined_label'] == label, 'adapted_hypotheses'].dropna().unique()
    adapted_flat = list(chain.from_iterable([a.split('|') for a in adapted_texts]))

    combined_texts = list(resp_texts) + adapted_flat
    combined_doc = ' '.join(combined_texts)  # Combine all texts into one document

    # Extract keywords/keyphrases
    keywords = kw_model.extract_keywords(
        combined_doc,
        keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
        stop_words='english',
        top_n=MAX_PHRASES,
        use_mmr=True,
        diversity=DIVERSITY
    )

    print(f"Top keyphrases for {label}:")
    for phrase, score in keywords:
        print(f" - {phrase} (score: {score:.3f})")
    print()

    # Store keywords in results dictionary
    results[label] = [{"phrase": phrase, "score": score} for phrase, score in keywords]

# Create output directory if it doesn't exist
os.makedirs("logs_and_metrics", exist_ok=True)

# Save results to JSON file
with open("logs_and_metrics/keybert_keyphrases.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Keyphrase extraction results saved to logs_and_metrics/keybert_keyphrases.json")