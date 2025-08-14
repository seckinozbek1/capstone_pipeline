import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from catboost import CatBoostClassifier
import pickle
import os
import torch
from itertools import chain
from keybert import KeyBERT
import tqdm


torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

# Load train labeled combined data for filtering
train_df = pd.read_csv("Q69_mmr_selected_labeled_combined.csv")
train_hashes = set(train_df['embedding_hash'].unique())

# Load scored sentences dataset
unseen_df = pd.read_csv("../top_scored_sentences.csv")

# Filter unseen sentences to exclude those present in training (by embedding_hash)
filtered_unseen_df = unseen_df[~unseen_df['embedding_hash'].isin(train_hashes)].copy()

print(f"Selected {len(filtered_unseen_df)} unseen sentences for prediction.")

# Load label mappings from saved_models folder
with open("saved_models/label2int.pkl", "rb") as f:
    label2int = pickle.load(f)
with open("saved_models/int2label.pkl", "rb") as f:
    int2label = pickle.load(f)

# Load trained CatBoost model from saved_models folder
model = CatBoostClassifier()
model.load_model("saved_models/catboost_multiclass_model.cbm")

# Initialize SBERT model - device aware
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('all-mpnet-base-v2', device=device)

# Encode unseen sentences using original 'sentence' column (not combined)
print("Encoding unseen sentences...")

all_embeddings = []
batch_size = 32
unseen_sentences = filtered_unseen_df['sentence'].tolist()

with torch.no_grad():
    for i in tqdm.tqdm(range(0, len(unseen_sentences), batch_size), desc="Encoding batches"):
        batch = unseen_sentences[i:i + batch_size]
        if device == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                emb = sbert_model.encode(batch, convert_to_tensor=True, device=device, normalize_embeddings=True)
        else:
            emb = sbert_model.encode(batch, convert_to_tensor=True, device=device, normalize_embeddings=True)
        all_embeddings.append(emb.cpu())

unseen_embeddings = torch.cat(all_embeddings).numpy()

# Prepare feature matrix
X_unseen = unseen_embeddings

# Predict classes and probabilities
pred_class_ints = model.predict(X_unseen).astype(int).flatten()
pred_probs_all = model.predict_proba(X_unseen)
pred_confidences = pred_probs_all[np.arange(len(pred_class_ints)), pred_class_ints]
pred_labels = [int2label[i] for i in pred_class_ints]

# Add prediction results to dataframe
filtered_unseen_df['predicted_combined_label'] = pred_labels
filtered_unseen_df['prediction_confidence'] = pred_confidences

# Calculate percentage above chance (adjust chance threshold based on your class count, e.g., 0.25 for 4 classes)
filtered_unseen_df['perc_above_chance'] = ((filtered_unseen_df['prediction_confidence'] - 0.25) / 0.75 * 100).clip(lower=0)

# Include semantic similarity filtering based on keyphrases

MAX_PHRASES = 3
KEYPHRASE_NGRAM_RANGE = (5, 7)
DIVERSITY = 0.3

valid_labels = list(label2int.keys())
kw_model = KeyBERT()

# Extract keyphrases function
def extract_keyphrases(texts, max_phrases=5):
    clean_texts = [t.replace('|', ' ') for t in texts if pd.notna(t)]
    combined_doc = ' '.join(clean_texts)
    keyphrases = kw_model.extract_keywords(
        combined_doc,
        keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
        stop_words='english',
        top_n=max_phrases,
        use_mmr=True,
        diversity=DIVERSITY
    )
    return [phrase for phrase, score in keyphrases]

print("Extracting keyphrases per label for semantic filtering...")
keyphrases_per_label = {}
for label in valid_labels:
    resp_texts = train_df.loc[train_df['combined_label'] == label, 'response_hypothesis'].dropna().unique()
    adapted_texts = train_df.loc[train_df['combined_label'] == label, 'adapted_hypotheses'].dropna().unique()
    adapted_flat = list(chain.from_iterable([a.split('|') for a in adapted_texts]))
    combined_texts = list(resp_texts) + adapted_flat
    keyphrases = extract_keyphrases(combined_texts, max_phrases=MAX_PHRASES)
    keyphrases_per_label[label] = keyphrases

# Encode keyphrases per label
label_to_keyphrase_embs = {}
for label, phrases in keyphrases_per_label.items():
    if phrases:
        embs = sbert_model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True).to(device)
    else:
        embs = torch.empty((0, sbert_model.get_sentence_embedding_dimension()), device=device)
    label_to_keyphrase_embs[label] = embs

# Semantic similarity filter function (Used only for scoring with 0 threshold, could be adjusted)
def semantic_filter_per_label(sent_emb, keyphrase_embs, threshold=0.0):
    if keyphrase_embs.shape[0] == 0:
        return False, 0.0
    sent_emb_tensor = torch.tensor(sent_emb).unsqueeze(0).to(device)
    cos_sims = torch.nn.functional.cosine_similarity(sent_emb_tensor, keyphrase_embs)
    max_sim = torch.max(cos_sims).item()
    is_match = max_sim >= threshold
    return is_match, max_sim

print("Applying semantic similarity filtering per predicted label...")
keep_mask = []
similarities = []

for sent_emb, pred_label in zip(X_unseen, pred_labels):
    keyphrase_embs = label_to_keyphrase_embs.get(pred_label, torch.empty(0, device=device))
    is_match, sim_score = semantic_filter_per_label(sent_emb, keyphrase_embs, threshold=0.0)
    keep_mask.append(is_match)
    similarities.append(sim_score)

filtered_unseen_df['semantic_keyphrase_similarity'] = similarities

# --- Joint score filtering combining confidence + semantic similarity ---

def joint_score_filter(
    df,
    confidence_col='perc_above_chance',
    similarity_col='semantic_keyphrase_similarity',
    confidence_weight=0.3,
    similarity_weight=0.7,
    confidence_threshold=2.0,
    similarity_threshold=0.45,
    joint_threshold=0.3
):
    conf_norm = df[confidence_col] / 100.0
    df['joint_score'] = conf_norm * confidence_weight + df[similarity_col] * similarity_weight

    keep = (
        (df[confidence_col] >= confidence_threshold) &
        (df[similarity_col] >= similarity_threshold) &
        (df['joint_score'] >= joint_threshold)
    )

    return df[keep].copy()

print("Applying joint scoring and filtering...")
filtered_unseen_df = joint_score_filter(filtered_unseen_df)
print(f"Kept {len(filtered_unseen_df)} predictions after joint scoring filtering.")

# Keep relevant columns only
keep_columns = [
    'sentence', 'embedding_hash', 'doc_id', 'B_COUNTRY_ALPHA', 'A_SESSION',
    'A_YEAR', 'speaker_name', 'speaker_post', 'predicted_combined_label',
    'prediction_confidence', 'perc_above_chance', 'semantic_keyphrase_similarity', 'joint_score'
]
filtered_unseen_df = filtered_unseen_df[keep_columns]

# Additional columns to merge from unseen_df
columns_to_fill = [
    'broad_qid', 'question_text', 'likert_scale',
    'response_text', 'response_hypothesis', 'adapted_hypotheses'
]

# Create lookup dataframe from unseen_df for merging
lookup_df = unseen_df[['combined_label'] + columns_to_fill].drop_duplicates(subset='combined_label')
lookup_df = lookup_df.rename(columns={'combined_label': 'predicted_combined_label'})

# Merge extra columns based on predicted_combined_label
filtered_unseen_df = filtered_unseen_df.merge(lookup_df, on='predicted_combined_label', how='left')

# Save results
os.makedirs("predictions", exist_ok=True)
output_path = "predictions/q69_predictions_top_score_filtered.csv"
filtered_unseen_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Saved the results to {output_path}")
