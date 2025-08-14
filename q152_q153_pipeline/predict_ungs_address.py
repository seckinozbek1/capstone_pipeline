import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from itertools import chain
from keybert import KeyBERT
import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

valid_labels = ['Q152_1', 'Q152_2', 'Q152_3', 'Q152_4']

# Hardcoded label mappings (keep for reference, but actual loaded below)
label2int = {label: i for i, label in enumerate(valid_labels)}
int2label = {i: label for i, label in enumerate(valid_labels)}

# --- Step 1: Load combined CSV and extract keyphrases using KeyBERT ---
print("Loading combined labeled CSV...")
df = pd.read_csv("Q152_mmr_selected_labeled_combined.csv")

kw_model = KeyBERT()
MAX_PHRASES = 3
KEYPHRASE_NGRAM_RANGE = (5, 7)
DIVERSITY = 0.3

def extract_keyphrases(texts, max_phrases=15):
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

print("Extracting top keyphrases per label...")
keyphrases_per_label = {}
for label in valid_labels:
    resp_texts = df.loc[df['combined_label'] == label, 'response_hypothesis'].dropna().unique()
    adapted_texts = df.loc[df['combined_label'] == label, 'adapted_hypotheses'].dropna().unique()
    adapted_flat = list(chain.from_iterable([a.split('|') for a in adapted_texts]))
    combined_texts = list(resp_texts) + adapted_flat
    keyphrases = extract_keyphrases(combined_texts, max_phrases=MAX_PHRASES)
    keyphrases_per_label[label] = '|'.join(keyphrases)

df['label_ngrams'] = df['combined_label'].map(keyphrases_per_label)

# --- Step 2: Load SBERT model and encode keyphrases per label ---
print("Loading SBERT model...")
sbert_model = SentenceTransformer('all-mpnet-base-v2', device=device)

label_to_keyphrase_list = {}
label_to_keyphrase_embs = {}

for label in valid_labels:
    phrases_str = keyphrases_per_label.get(label, '')
    phrases = [p.strip() for p in phrases_str.split('|') if p.strip()]
    label_to_keyphrase_list[label] = phrases
    if phrases:
        embs = sbert_model.encode(phrases, convert_to_tensor=True, normalize_embeddings=True).to(device)
    else:
        embs = torch.empty((0, sbert_model.get_sentence_embedding_dimension()), device=device)
    label_to_keyphrase_embs[label] = embs

# --- Step 3: Load unseen data to predict ---
print("Loading unseen data...")
ungs_address_df = pd.read_csv("../ungs_address_corpus.csv")
if 'sentence_text' in ungs_address_df.columns:
    ungs_address_df = ungs_address_df.rename(columns={'sentence_text': 'sentence'})

# --- Load label mappings from saved_models folder ---
with open("saved_models/label2int.pkl", "rb") as f:
    label2int = pickle.load(f)
with open("saved_models/int2label.pkl", "rb") as f:
    int2label = pickle.load(f)

# --- Step 4: Load CatBoost model ---
print("Loading CatBoost model...")
model = CatBoostClassifier()
model.load_model("saved_models/catboost_multiclass_model.cbm")

# --- Step 5: Encode unseen sentences ---
torch.cuda.empty_cache()

print("Encoding unseen sentences...")
unseen_sentences = ungs_address_df['sentence'].tolist()
batch_size = 32
all_embeddings = []

sbert_model.eval()  

with torch.no_grad():
    for i in tqdm.tqdm(range(0, len(unseen_sentences), batch_size), desc="Encoding batches"):
        batch = unseen_sentences[i:i+batch_size]
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            emb = sbert_model.encode(
                batch,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True
            )
        all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings).numpy()

# --- Step 6: Predict with CatBoost ---
print("Predicting classes and probabilities...")
pred_class_ints = model.predict(embeddings).astype(int).flatten()
pred_probs_all = model.predict_proba(embeddings)
pred_confidences = pred_probs_all[np.arange(len(pred_class_ints)), pred_class_ints]
pred_labels = [int2label[i] for i in pred_class_ints]

# --- Step 7: Semantic similarity filtering per predicted label (Used only for scoring with 0 threshold, could be adjusted)---
def semantic_filter_per_label(sent_emb, keyphrase_embs, keyphrase_list, threshold=0.0):
    if keyphrase_embs.shape[0] == 0:
        return False, 0.0, None
    sent_emb_tensor = torch.tensor(sent_emb).unsqueeze(0).to(device)
    cos_sims = torch.nn.functional.cosine_similarity(sent_emb_tensor, keyphrase_embs)
    max_sim, max_idx = torch.max(cos_sims, dim=0)
    is_match = max_sim >= threshold
    matched_phrase = keyphrase_list[max_idx] if is_match else None
    return is_match.item(), max_sim.item(), matched_phrase

print("Applying semantic similarity filtering per predicted label...")
keep_mask = []
similarities = []
matched_keyphrases = []

for sent_emb, pred_label in zip(embeddings, pred_labels):
    keyphrase_embs = label_to_keyphrase_embs.get(pred_label, torch.empty(0, device=device))
    keyphrase_list = label_to_keyphrase_list.get(pred_label, [])
    is_match, sim_score, matched_phrase = semantic_filter_per_label(sent_emb, keyphrase_embs, keyphrase_list)
    keep_mask.append(is_match)
    similarities.append(sim_score)
    matched_keyphrases.append(matched_phrase if matched_phrase is not None else '')

# --- Step 8: Assemble final DataFrame ---
ungs_address_df['predicted_combined_label'] = pred_labels
ungs_address_df['prediction_confidence'] = pred_confidences
ungs_address_df['perc_above_chance'] = ((ungs_address_df['prediction_confidence'] - 0.25) / 0.75 * 100).clip(lower=0)
ungs_address_df['semantic_keyphrase_similarity'] = similarities
ungs_address_df['matched_keyphrase'] = matched_keyphrases

# --- Step 9: Joint score filtering (weighted confidence + semantic similarity) ---
def joint_score_filter(
    df,
    confidence_col='perc_above_chance',
    similarity_col='semantic_keyphrase_similarity',
    confidence_weight=0.3,
    similarity_weight=0.7,
    confidence_threshold=2.0,
    similarity_threshold=0.4,
    joint_threshold=0.05
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
final_df = joint_score_filter(ungs_address_df)
print(f"Kept {len(final_df)} predictions after joint scoring filtering.")

# Keep relevant columns for output
keep_columns = [
    'sentence', 'doc_id',
    'A_YEAR', 'predicted_combined_label',
    'prediction_confidence', 'perc_above_chance', 'semantic_keyphrase_similarity', 'matched_keyphrase', 'joint_score'
]
final_df = final_df[keep_columns]

# --- Step 10: Save final filtered predictions ---
os.makedirs("predictions", exist_ok=True)
output_path = "predictions/q152_predictions_ungs.csv"
final_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"Saved filtered predictions to {output_path}")
