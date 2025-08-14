import os
import sys
import hashlib
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# === Helpers ===

def sentence_hash(text: str) -> str:
    """Generate SHA-256 hash of a single sentence string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def embed_and_cache(all_sentences_df, model_name="all-mpnet-base-v2", cache_dir="./cache", batch_size=64):
    """Embed all sentences with batching, fp16 and cache."""
    folder = os.path.join(cache_dir, "all_questions")
    os.makedirs(folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer(model_name)
    model.to(device)
    model.eval()

    sentences = all_sentences_df['sentence'].values
    hashes = [sentence_hash(s) for s in sentences]

    with torch.no_grad():
        with tqdm(total=len(sentences), desc="Embedding all sentences") as pbar:
            for start_idx in range(0, len(sentences), batch_size):
                end_idx = min(start_idx + batch_size, len(sentences))
                batch_sentences = sentences[start_idx:end_idx]
                batch_hashes = hashes[start_idx:end_idx]

                missing_indices = []
                for i, h in enumerate(batch_hashes):
                    cache_path = os.path.join(folder, f"{model_name}_{h}.npy")
                    if not os.path.exists(cache_path):
                        missing_indices.append(i)

                if missing_indices:
                    to_embed = [batch_sentences[i] for i in missing_indices]

                    with torch.amp.autocast(device_type='cuda'):
                        embeddings = model.encode(to_embed, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

                    for i, emb in zip(missing_indices, embeddings):
                        cache_path = os.path.join(folder, f"{model_name}_{batch_hashes[i]}.npy")
                        np.save(cache_path, emb)

                pbar.update(len(batch_sentences))

def load_embeddings_by_hashes(df, embedding_type, model_name="all-mpnet-base-v2", cache_dir="./cache", show_progress=True):
    folder = os.path.join(cache_dir, embedding_type)
    embeddings = []
    missing = []
    sentences = df['sentence'].values
    hashes = [sentence_hash(s) for s in sentences]

    iterator = tqdm(hashes, desc=f"Loading embeddings from {embedding_type}") if show_progress else hashes
    for h in iterator:
        cache_path = os.path.join(folder, f"{model_name}_{h}.npy")
        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            embeddings.append(emb)
        else:
            missing.append(h)

    if missing:
        print(f"Warning: Missing {len(missing)} embeddings in {embedding_type} cache. Examples: {missing[:10]}")

    if len(embeddings) == 0:
        raise ValueError(f"No embeddings loaded for {embedding_type}!")

    return np.vstack(embeddings)

def mmr(embeddings, query_embedding=None, top_k=20, diversity=0.7):
    if query_embedding is None:
        query_embedding = np.mean(embeddings, axis=0, keepdims=True)
    selected = []
    candidates = list(range(len(embeddings)))
    while len(selected) < top_k and candidates:
        mmr_scores = []
        for idx in candidates:
            emb = embeddings[idx].reshape(1, -1)
            sim_to_query = cosine_similarity(emb, query_embedding)[0][0]
            sim_to_selected = max(cosine_similarity(emb, embeddings[selected])[0]) if selected else 0
            score = diversity * sim_to_query - (1 - diversity) * sim_to_selected
            mmr_scores.append((score, idx))
        mmr_scores.sort(reverse=True)
        best_score, best_idx = mmr_scores[0]
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()

    IQR_SAMPLES_DIR = "iqr_samples"
    MMR_OUTPUT_DIR = "mmr_selected"
    os.makedirs(MMR_OUTPUT_DIR, exist_ok=True)
    MODEL_NAME = "all-mpnet-base-v2"
    CACHE_BASE_DIR = "cache"
    TOP_K = 100

    all_files = [f for f in os.listdir(IQR_SAMPLES_DIR) if f.endswith("_iqr_samples.csv")]

    logger.info(f"Found {len(all_files)} per-question IQR sample files.")

    dfs = []
    for idx, filename in enumerate(all_files):
        filepath = os.path.join(IQR_SAMPLES_DIR, filename)
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning(f"{filename} is empty, skipping.")
            continue
        df['question_idx'] = idx
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Total combined sentences from all questions: {len(all_df)}")

    # Embed and cache all sentences with a single overall tqdm
    embed_and_cache(all_df, model_name=MODEL_NAME, cache_dir=CACHE_BASE_DIR, batch_size=64)

    # Load all embeddings with a single overall tqdm
    all_embeddings = load_embeddings_by_hashes(all_df, embedding_type="all_questions", model_name=MODEL_NAME, cache_dir=CACHE_BASE_DIR, show_progress=True)

    for q_idx, filename in enumerate(all_files):
        broad_qid = filename.replace("_iqr_samples.csv", "")
        logger.info(f"Processing question {broad_qid} ({q_idx + 1}/{len(all_files)})...")

        df_question = all_df[all_df['question_idx'] == q_idx].reset_index(drop=True)
        indices = df_question.index.to_numpy()
        embeddings_question = all_embeddings[indices]

        unique_labels = df_question['combined_label'].unique()
        num_labels = len(unique_labels)
        per_label_k = max(1, TOP_K // num_labels)
        logger.info(f"{broad_qid}: {num_labels} unique combined_labels, selecting ~{per_label_k} per label.")

        selected_dfs = []
        for label in unique_labels:
            df_label = df_question[df_question['combined_label'] == label].reset_index(drop=True)
            label_indices = df_label.index.to_numpy()
            embeddings_label = embeddings_question[label_indices]

            current_k = min(per_label_k, len(df_label))
            selected_indices = mmr(embeddings_label, top_k=current_k, diversity=0.7)
            selected_dfs.append(df_label.iloc[selected_indices])

        selected_df = pd.concat(selected_dfs).reset_index(drop=True)
        logger.info(f"{broad_qid}: Selected {len(selected_df)} sentences after balanced MMR.")

        out_path = os.path.join(MMR_OUTPUT_DIR, f"{broad_qid}_mmr_selected.csv")
        selected_df.to_csv(out_path, index=False)

    logger.info("Finished balanced MMR selection for all questions.")

if __name__ == "__main__":
    main()
