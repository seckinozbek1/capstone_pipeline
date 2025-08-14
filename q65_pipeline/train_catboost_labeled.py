import os
import json
import pandas as pd
import numpy as np
import random
import pickle
from itertools import chain
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import torch

torch.cuda.empty_cache()

os.makedirs("logs_and_metrics", exist_ok=True)

nltk.download('wordnet')
nltk.download('omw-1.4')

def synonym_augment(sentence, num_replacements=1, seed=None):
    if seed is not None:
        random.seed(seed)
    words = sentence.split()
    new_words = words.copy()
    candidates = [i for i, w in enumerate(words) if len(w) > 3 and len(wordnet.synsets(w)) > 0]
    if not candidates:
        return sentence
    replace_indices = random.sample(candidates, min(num_replacements, len(candidates)))
    for idx in replace_indices:
        synonyms = wordnet.synsets(words[idx])
        lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        lemmas.discard(words[idx])
        lemmas = [l for l in lemmas if '_' not in l and l.isalpha() and len(l) > 2]
        if lemmas:
            new_word = random.choice(lemmas)
            new_words[idx] = new_word
    return ' '.join(new_words)

print("Loading data...")
df = pd.read_csv("Q65_mmr_selected_labeled_combined.csv")

# *** Changed to 4-class labels ***
valid_labels = ['Q65_1', 'Q65_2', 'Q65_3', 'Q65_4']
df = df[df['combined_label'].isin(valid_labels)].copy()
df = df[df['sentence'].notna() & (df['sentence'].str.strip() != "")].reset_index(drop=True)

label_counts = df['combined_label'].value_counts()
df = df[df['combined_label'].isin(label_counts[label_counts >= 5].index)].copy()

label_encoder = LabelEncoder()
label_encoder.fit(valid_labels)  # fit on all 4 classes
df['label_int'] = label_encoder.transform(df['combined_label'])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading SBERT model...")
sbert = SentenceTransformer('all-mpnet-base-v2', device=device)

print("Encoding all sentences for embedding hash computation...")
embeddings = sbert.encode(df['sentence'].tolist(), convert_to_numpy=True, batch_size=32, show_progress_bar=True)
df['embedding_hash'] = [hash(tuple(emb)) for emb in embeddings]

print("Splitting data stratified by embedding_hash...")
unique_hashes = df[['embedding_hash', 'label_int']].drop_duplicates()
train_hashes, val_hashes = train_test_split(
    unique_hashes,
    test_size=0.27,
    stratify=unique_hashes['label_int'],
    random_state=42
)

train_df = df[df['embedding_hash'].isin(train_hashes['embedding_hash'])].copy()
val_df = df[df['embedding_hash'].isin(val_hashes['embedding_hash'])].copy()

print(f"Training samples before augmentation: {len(train_df)}")
print(f"Validation samples before filtering: {len(val_df)}")

print("Performing synonym augmentation on training set...")
aug_sentences = []
aug_labels = []

num_augmentations = 3

for idx, row in train_df.iterrows():
    orig_sent = row['sentence']
    label = row['label_int']
    aug_sentences.append(orig_sent)
    aug_labels.append(label)
    for n in range(num_augmentations):
        aug_sent = synonym_augment(orig_sent, num_replacements=1, seed=idx * 10 + n)
        aug_sentences.append(aug_sent)
        aug_labels.append(label)

train_aug_df = pd.DataFrame({'sentence': aug_sentences, 'label_int': aug_labels})

print(f"Training samples after augmentation: {len(train_aug_df)}")

print("Encoding training sentences after augmentation...")
train_embeddings = sbert.encode(train_aug_df['sentence'].tolist(), convert_to_numpy=True, batch_size=32, show_progress_bar=True)

print("Encoding validation sentences...")
val_embeddings = sbert.encode(val_df['sentence'].tolist(), convert_to_numpy=True, batch_size=32, show_progress_bar=True)

print("Encoding original training sentences for permutation tests...")
train_orig_embeddings = sbert.encode(train_df['sentence'].tolist(), convert_to_numpy=True, batch_size=32, show_progress_bar=True)

print("Filtering validation samples too close to training samples...")
train_emb_tensor = torch.tensor(train_embeddings)
val_emb_tensor = torch.tensor(val_embeddings)
cosine_sim_matrix = util.cos_sim(val_emb_tensor, train_emb_tensor)
max_similarities = cosine_sim_matrix.max(dim=1).values.cpu().numpy()
keep_indices = max_similarities < 0.85

filtered_val_df = val_df.iloc[np.where(keep_indices)[0]].reset_index(drop=True)
filtered_val_embeddings = val_embeddings[keep_indices]

print(f"Validation samples after cosine similarity filtering: {len(filtered_val_df)}")

train_pool = Pool(train_embeddings, label=train_aug_df['label_int'].values)
val_pool = Pool(filtered_val_embeddings, label=filtered_val_df['label_int'].values)

catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.0085,
    'depth': 4,
    'l2_leaf_reg': 32,
    'random_strength': 9,
    'bagging_temperature': 0.4,
    'loss_function': 'MultiClass',       # multiclass loss
    'eval_metric': 'MultiClass',         # multiclass eval
    'auto_class_weights': 'Balanced',
    'random_seed': 42,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 200,
}

EARLY_STOPPING_ROUNDS = 265

torch.cuda.empty_cache()

print("Training CatBoost classifier...")
model = CatBoostClassifier(**catboost_params)
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

eval_results = model.get_evals_result()
best_iter = model.get_best_iteration()

train_logloss_at_best = eval_results['learn']['MultiClass'][best_iter] if 'MultiClass' in eval_results['learn'] else None
val_logloss_at_best = eval_results['validation']['MultiClass'][best_iter] if 'MultiClass' in eval_results['validation'] else None

print(f"Training Multinomial LogLoss at Best Iteration ({best_iter}): {train_logloss_at_best:.4f}")
print(f"Validation Multinomial LogLoss at Best Iteration ({best_iter}): {val_logloss_at_best:.4f}")

# Add best iteration info into eval_results dict
eval_results['best_iteration'] = best_iter
eval_results['train_metric_at_best'] = train_logloss_at_best
eval_results['val_metric_at_best'] = val_logloss_at_best

with open("logs_and_metrics/training_eval_metrics.json", "w") as f:
    json.dump(eval_results, f, indent=2)

os.makedirs("saved_models", exist_ok=True)
model.save_model("saved_models/catboost_multiclass_model.cbm")

label2int = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
int2label = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

with open("saved_models/label2int.pkl", "wb") as f:
    pickle.dump(label2int, f)
with open("saved_models/int2label.pkl", "wb") as f:
    pickle.dump(int2label, f)

print("Saved model and label mappings.")

probs = model.predict_proba(filtered_val_embeddings)  # shape: (n_samples, 4)
val_preds = np.argmax(probs, axis=1)
macro_f1 = f1_score(filtered_val_df['label_int'], val_preds, average='macro')

print(f"\nFinal Macro F1: {macro_f1:.4f}")
print("Classification Report:")
print(classification_report(
    filtered_val_df['label_int'], val_preds,
    target_names=label_encoder.classes_,
    zero_division=0
))

# --- Save final classification report as JSON ---

def format_classification_report(labels, precisions, recalls, f1s, supports):
    return [
        {
            "class": label,
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1_score": float(f1s[i]),
            "support": int(supports[i])  # support as integer
        }
        for i, label in enumerate(labels)
    ]

# Compute precision, recall, f1, support for final preds
precision, recall, f1, support = precision_recall_fscore_support(filtered_val_df['label_int'], val_preds, zero_division=0)

final_report = {
    "final_macro_f1": float(macro_f1),
    "classification_report": format_classification_report(
        label_encoder.classes_,
        precision,
        recall,
        f1,
        support
    )
}

with open("logs_and_metrics/val_classification_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

# --------- Permutation tests ---------

n_permutations = 20

perm_f1_scores_orig = []
perm_class_reports_orig = []

perm_f1_scores_aug = []
perm_class_reports_aug = []

print("\nRunning permutation tests on original training data...")
for i in range(n_permutations):
    print(f"Permutation run {i+1}/{n_permutations} (original data)")
    torch.cuda.empty_cache()
    shuffled_labels = np.random.permutation(train_df['label_int'].values)
    shuffled_pool = Pool(train_orig_embeddings, label=shuffled_labels)

    # --- use best_iter for permutation runs; keep eval_set; no early stopping; no best-model selection
    perm_params = {**catboost_params, 'iterations': int(best_iter)}
    perm_model = CatBoostClassifier(**perm_params)
    perm_model.fit(
        shuffled_pool,
        eval_set=val_pool,
        use_best_model=False,
        verbose=0
    )

    perm_probs = perm_model.predict_proba(filtered_val_embeddings)
    perm_preds = np.argmax(perm_probs, axis=1)

    perm_f1 = f1_score(filtered_val_df['label_int'], perm_preds, average='macro')
    perm_f1_scores_orig.append(perm_f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_val_df['label_int'], perm_preds, zero_division=0
    )
    perm_class_reports_orig.append((precision, recall, f1, support))

avg_perm_f1_orig = np.mean(perm_f1_scores_orig)
precisions_orig = np.vstack([r[0] for r in perm_class_reports_orig])
recalls_orig = np.vstack([r[1] for r in perm_class_reports_orig])
f1s_orig = np.vstack([r[2] for r in perm_class_reports_orig])
supports_orig = perm_class_reports_orig[0][3]

avg_precision_orig = np.mean(precisions_orig, axis=0)
avg_recall_orig = np.mean(recalls_orig, axis=0)
avg_f1_orig = np.mean(f1s_orig, axis=0)

print(f"\nAverage Permutation Test Macro F1 on original training data over {n_permutations} runs: {avg_perm_f1_orig:.4f}\n")
print("Average classification report (per class) on original training data:")
print(f"{'Class':<10} {'Precision':>9} {'Recall':>7} {'F1-score':>9} {'Support':>8}")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label:<10} {avg_precision_orig[i]:9.3f} {avg_recall_orig[i]:7.3f} {avg_f1_orig[i]:9.3f} {supports_orig[i]:8d}")

print("\nRunning permutation tests on augmented training data...")
for i in range(n_permutations):
    print(f"Permutation run {i+1}/{n_permutations} (augmented data)")
    torch.cuda.empty_cache()
    shuffled_labels = np.random.permutation(train_aug_df['label_int'].values)
    shuffled_pool = Pool(train_embeddings, label=shuffled_labels)

    # --- use best_iter for permutation runs; keep eval_set; no early stopping; no best-model selection
    perm_params = {**catboost_params, 'iterations': int(best_iter)}
    perm_model = CatBoostClassifier(**perm_params)
    perm_model.fit(
        shuffled_pool,
        eval_set=val_pool,
        use_best_model=False,
        verbose=0
    )

    perm_probs = perm_model.predict_proba(filtered_val_embeddings)
    perm_preds = np.argmax(perm_probs, axis=1)

    perm_f1 = f1_score(filtered_val_df['label_int'], perm_preds, average='macro')
    perm_f1_scores_aug.append(perm_f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_val_df['label_int'], perm_preds, zero_division=0
    )
    perm_class_reports_aug.append((precision, recall, f1, support))

avg_perm_f1_aug = np.mean(perm_f1_scores_aug)
precisions_aug = np.vstack([r[0] for r in perm_class_reports_aug])
recalls_aug = np.vstack([r[1] for r in perm_class_reports_aug])
f1s_aug = np.vstack([r[2] for r in perm_class_reports_aug])
supports_aug = perm_class_reports_aug[0][3]

avg_precision_aug = np.mean(precisions_aug, axis=0)
avg_recall_aug = np.mean(recalls_aug, axis=0)
avg_f1_aug = np.mean(f1s_aug, axis=0)

print(f"\nAverage Permutation Test Macro F1 on augmented training data over {n_permutations} runs: {avg_perm_f1_aug:.4f}\n")
print("Average classification report (per class) on augmented training data:")
print(f"{'Class':<10} {'Precision':>9} {'Recall':>7} {'F1-score':>9} {'Support':>8}")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label:<10} {avg_precision_aug[i]:9.3f} {avg_recall_aug[i]:7.3f} {avg_f1_aug[i]:9.3f} {supports_aug[i]:8d}")

# --- Save permutation test results as JSON ---
perm_report_orig = {
    "average_macro_f1": float(avg_perm_f1_orig),
    "classification_report_per_class": format_classification_report(
        label_encoder.classes_,
        avg_precision_orig,
        avg_recall_orig,
        avg_f1_orig,
        supports_orig
    )
}

perm_report_aug = {
    "average_macro_f1": float(avg_perm_f1_aug),
    "classification_report_per_class": format_classification_report(
        label_encoder.classes_,
        avg_precision_aug,
        avg_recall_aug,
        avg_f1_aug,
        supports_aug
    )
}

with open("logs_and_metrics/permutation_test_report_original.json", "w") as f:
    json.dump(perm_report_orig, f, indent=2)

with open("logs_and_metrics/permutation_test_report_augmented.json", "w") as f:
    json.dump(perm_report_aug, f, indent=2)
