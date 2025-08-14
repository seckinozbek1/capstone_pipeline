import os
import sys
import re
import hashlib
import logging
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from keybert import KeyBERT

def load_filter(file_path, valid_countries, valid_years):
    """Load and filter UNGA sentences by country and year."""
    unga_sentences = pd.read_csv(file_path)
    filtered_unga = unga_sentences[
        (unga_sentences['A_YEAR'].isin(valid_years)) &
        (unga_sentences['B_COUNTRY_ALPHA'].isin(valid_countries))
    ].copy()
    filtered_unga.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(unga_sentences)} sentences; filtered down to {len(filtered_unga)}; no sampling performed.")
    return filtered_unga

def extract_keywords(wvs_df, combined_label_col='combined_label',
                     adapted_col='adapted_hypotheses',
                     response_col='response_hypothesis',
                     top_k_keywords=3,
                     top_k_bonuses_response=3,
                     top_k_bonuses_adapted=3):
    """
    Extract keywords and bonuses using KeyBERT for semantically meaningful phrases.
    Removes '[SEP]' tokens before extraction.
    Keywords are extracted only from response hypotheses and re-ranked by rarity
    computed per broad_qid across all combined_labels.
    """
    keyword_dict = {}
    bonus_dict_response = {}
    bonus_dict_adapted = {}

    # Initialize KeyBERT model
    kw_model = KeyBERT('all-mpnet-base-v2') 

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = text.replace('[SEP]', ' ')  # Replace [SEP] with space to avoid artifacts
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.lower()

    grouped = wvs_df.groupby(combined_label_col)

    for label, group in tqdm(grouped, desc="Extracting keywords and bonuses with KeyBERT + rarity"):
        response_texts = group[response_col].dropna().astype(str).apply(clean_text).tolist()
        adapted_texts = group[adapted_col].dropna().astype(str).apply(clean_text).tolist()

        # Extract keywords only from response texts combined
        resp_doc = " ".join(response_texts).strip()

        if not resp_doc:
            keyword_dict[label] = []
        else:
            # Extract candidate keywords (phrases) with KeyBERT
            candidates = kw_model.extract_keywords(
                resp_doc,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=20,
                use_mmr=True,
                diversity=0.7
            )
            # Just take the phrases (ignore scores here)
            keyword_dict[label] = [phrase for phrase, _ in candidates[:top_k_keywords]]

        # Extract bonus phrases from response texts
        if resp_doc:
            bonuses_resp = kw_model.extract_keywords(
                resp_doc,
                keyphrase_ngram_range=(2, 3),
                stop_words='english',
                top_n=top_k_bonuses_response,
                use_mmr=True,
                diversity=0.7
            )
            bonus_dict_response[label] = [b[0] for b in bonuses_resp]
        else:
            bonus_dict_response[label] = []

        # Extract bonus phrases from adapted texts
        adap_doc = " ".join(adapted_texts).strip()
        if adap_doc:
            bonuses_adap = kw_model.extract_keywords(
                adap_doc,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_k_bonuses_adapted,
                use_mmr=True,
                diversity=0.7
            )
            bonus_dict_adapted[label] = [b[0] for b in bonuses_adap]
        else:
            bonus_dict_adapted[label] = []

    # Print keywords and bonuses per combined_label
    for label in sorted(keyword_dict.keys()):
        print(f"\nCombined Label: {label}")
        print(f"  Keywords: {keyword_dict[label]}")
        print(f"  Bonus phrases (Response): {bonus_dict_response.get(label, [])}")
        print(f"  Bonus phrases (Adapted): {bonus_dict_adapted.get(label, [])}")

    return keyword_dict, bonus_dict_response, bonus_dict_adapted

def sentence_hash(text: str) -> str:
    """Generate SHA-256 hash of a single sentence string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def embed_cache(texts, embedding_type="unga_speech_embeddings",
                model_name="all-mpnet-base-v2", cache_dir="cache",
                batch_size=64):
    folder = os.path.join(cache_dir, embedding_type)
    os.makedirs(folder, exist_ok=True)
    model = SentenceTransformer(model_name)
    embeddings_list = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding sentences with caching ({embedding_type})"):
        batch_texts = texts[i:i+batch_size]
        batch_hashes = [sentence_hash(t) for t in batch_texts]

        to_compute = []
        to_compute_indices = []
        for idx, h in enumerate(batch_hashes):
            cache_path = os.path.join(folder, f"{model_name}_{h}.npy")
            cache_path = os.path.normpath(cache_path)
            if not os.path.exists(cache_path):
                to_compute.append(batch_texts[idx])
                to_compute_indices.append(idx)

        batch_embeddings = [None] * len(batch_texts)

        # Load embeddings from cache if available
        for idx, h in enumerate(batch_hashes):
            cache_path = os.path.join(folder, f"{model_name}_{h}.npy")
            cache_path = os.path.normpath(cache_path)
            if os.path.exists(cache_path):
                emb = np.load(cache_path)
                batch_embeddings[idx] = emb

        # Compute embeddings for missing sentences and assign them back
        if to_compute:
            with torch.amp.autocast(device_type=device):
                computed_embs = model.encode(to_compute, show_progress_bar=False)
            for i, emb in enumerate(computed_embs):
                cache_path = os.path.join(folder, f"{model_name}_{batch_hashes[to_compute_indices[i]]}.npy")
                cache_path = os.path.normpath(cache_path)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, emb)
                batch_embeddings[to_compute_indices[i]] = emb

        # Verify no None entries remain before stacking
        if any(be is None for be in batch_embeddings):
            raise ValueError("Missing embeddings in batch! Some sentences were not computed or loaded correctly.")

        embeddings_list.extend(batch_embeddings)

    embeddings = np.vstack(embeddings_list)
    return embeddings


def compute_rarity(sentences, stop_words='english', rare_token_threshold=100):
    """Compute normalized rarity scores based on rare token proportions."""
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words, max_features=5000)
    X = vectorizer.fit_transform(sentences)
    token_freq = np.asarray(X.sum(axis=0)).ravel()
    token_names = np.array(vectorizer.get_feature_names_out())
    top_tokens_idx = np.argsort(-token_freq)[:rare_token_threshold]
    top_tokens = set(token_names[top_tokens_idx])
    rarity_scores = []
    for sent in sentences:
        tokens = set(re.findall(r'\b\w+\b', sent.lower()))
        rare_token_hits = len([t for t in tokens if t not in top_tokens])
        rarity = rare_token_hits / (len(tokens) + 1e-5)
        rarity_scores.append(rarity)
    rarity_array = np.array(rarity_scores)
    rarity_norm = (rarity_array - rarity_array.min()) / (rarity_array.max() - rarity_array.min() + 1e-8)
    return rarity_norm

def compute_score(sentences, combined_labels, sentence_embeddings,
                  combined_label_hypotheses_dict,
                  keyword_dict, bonus_dict_response, bonus_dict_adapted,
                  model, batch_size=64,
                  aggregation='max'):
    """
    Compute composite scores for each sentence and combined label using GPU-accelerated
    cosine similarity via PyTorch with autocast for mixed precision.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    rarity_scores = compute_rarity(sentences)
    sentence_embeddings_tensor = torch.tensor(sentence_embeddings, device=device)  # [num_sentences, emb_dim]

    hypothesis_embeddings_cache = {}

    def embed_hypotheses(hypos):
        if tuple(hypos) in hypothesis_embeddings_cache:
            return hypothesis_embeddings_cache[tuple(hypos)]
        embeddings = []
        for i in range(0, len(hypos), batch_size):
            batch = hypos[i:i+batch_size]
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type):
                batch_emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.append(batch_emb)
        embeddings = np.vstack(embeddings)
        hypothesis_embeddings_cache[tuple(hypos)] = embeddings
        return embeddings

    def keyword_score(text, keywords):
        return min(sum(1 for kw in keywords if kw in text.lower()), 2)

    def bonus_score(text, bonuses):
        return 1.0 if any(b in text.lower() for b in bonuses) else 0.0

    # Precompute aggregated hypothesis embeddings per label as PyTorch tensors
    aggregated_hypo_embeddings = {}

    for clabel in tqdm(combined_labels, desc="Embedding and aggregating hypotheses"):
        raw_hypos = combined_label_hypotheses_dict.get(clabel, [])
        if not raw_hypos:
            continue

        split_hypos = []
        for hypo in raw_hypos:
            if isinstance(hypo, str) and '[SEP]' in hypo:
                split_hypos.extend([s.strip() for s in hypo.split('[SEP]') if s.strip()])
            else:
                split_hypos.append(hypo.strip())

        split_hypos = list(set([h for h in split_hypos if h]))

        hypo_embs = embed_hypotheses(split_hypos)
        aggregated_emb = hypo_embs.mean(axis=0).reshape(1, -1)
        aggregated_hypo_embeddings[clabel] = torch.tensor(aggregated_emb, device=device)

    for clabel in tqdm(combined_labels, desc="Scoring sentences"):
        if clabel not in aggregated_hypo_embeddings:
            continue
        agg_emb_tensor = aggregated_hypo_embeddings[clabel]  # [1, emb_dim]
        agg_emb_expanded = agg_emb_tensor.expand(sentence_embeddings_tensor.size(0), -1)  # [num_sentences, emb_dim]

        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            cos_sim_tensor = F.cosine_similarity(sentence_embeddings_tensor, agg_emb_expanded)  # [num_sentences]

        cos_sim_array = cos_sim_tensor.cpu().numpy()

        kw_list = keyword_dict.get(clabel, [])
        bonus_resp_list = bonus_dict_response.get(clabel, [])
        bonus_adap_list = bonus_dict_adapted.get(clabel, [])

        for i, sentence in enumerate(sentences):
            kw_sc = keyword_score(sentence, kw_list)
            bon_sc_resp = bonus_score(sentence, bonus_resp_list)
            bon_sc_adap = bonus_score(sentence, bonus_adap_list)
            rar_sc = rarity_scores[i]
            cos_sc = cos_sim_array[i]

            total = 0.6 * cos_sc + 0.2 * (kw_sc / 2) + 0.1 * bon_sc_resp + 0.05 * bon_sc_adap + 0.05 * rar_sc

            results.append({
                "combined_label": clabel,
                "sentence": sentence,
                "cosine_score": cos_sc,
                "keyword_score": kw_sc,
                "bonus_score_response": bon_sc_resp,
                "bonus_score_adapted": bon_sc_adap,
                "rarity_score": rar_sc,
                "total_score": total
            })

    return results

def select_top(scoring_results, unga_metadata_df, wvs_metadata_df, top_n=20):
    """Select top N scored sentences per label, then remove duplicate sentences globally."""
    scores_df = pd.DataFrame(scoring_results)

    merged = scores_df.merge(
        unga_metadata_df,
        left_on="sentence",
        right_on="sentence_text",
        how="left",
        suffixes=("", "_unga")
    )

    merged = merged.merge(
        wvs_metadata_df.drop_duplicates(subset=["combined_label"]),
        on="combined_label",
        how="left",
        suffixes=("", "_wvs")
    )

    merged_sorted = merged.sort_values(by=["combined_label", "total_score"], ascending=[True, False])
    top_sentences = merged_sorted.groupby("combined_label").head(top_n).reset_index(drop=True)

    top_sentences = top_sentences.sort_values(by='total_score', ascending=False)
    top_sentences = top_sentences.drop_duplicates(subset=['sentence']).reset_index(drop=True)

    if 'sentence_text' in top_sentences.columns:
        top_sentences = top_sentences.drop(columns=['sentence_text'])

    return top_sentences

def main():
    UNGA_SENTENCE_CSV = "unga_speech_corpus.csv"
    WVS_DATASET_CSV = "group_WVC_stem_encoding_complete_long.csv"
    CACHE_DIR = "cache"
    OUTPUT_CSV = "top_scored_sentences.csv"
    UNGA_HASHED_CSV = "unga_wvs7_hashed_corpus.csv"
    TOP_K_KEYWORDS = 4
    TOP_K_BONUSES_RESPONSE = 4
    TOP_K_BONUSES_ADAPTED = 7
    TOP_N_SENTENCES = 150
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    VALID_YEARS = list(range(2017, 2023))
    VALID_COUNTRIES = {
        "AND", "ARG", "AUS", "BGD", "ARM", "BOL", "BRA", "MMR", "CAN", "CHL", "CHN",
        "COL", "CYP", "CZE", "ECU", "ETH", "DEU", "GRC", "GTM", "IND", "IDN", "IRN",
        "IRQ", "JPN", "KAZ", "JOR", "KEN", "KOR", "KGZ", "LBN", "LBY", "MYS", "MDV",
        "MEX", "MNG", "MAR", "NLD", "NZL", "NIC", "NGA", "PAK", "PER", "PHL", "ROU",
        "RUS", "SRB", "SGP", "SVK", "VNM", "ZWE", "TJK", "THA", "TUN", "TUR", "UKR",
        "EGY", "GBR", "USA", "URY", "UZB", "VEN"
    }
    BATCH_SIZE = 64
    AGGREGATION_METHOD = "max"

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()

    try:
        logger.info("Loading and filtering UNGA sentences...")
        unga_df = load_filter(
            file_path=UNGA_SENTENCE_CSV,
            valid_countries=VALID_COUNTRIES,
            valid_years=VALID_YEARS
        )

        unga_df['embedding_hash'] = unga_df['sentence_text'].apply(sentence_hash)
        logger.info(f"Added embedding_hash column to UNGA sentences.")

        # Save filtered and hashed UNGA corpus CSV
        unga_df.to_csv(UNGA_HASHED_CSV, index=False, encoding='utf-8')
        logger.info(f"Saved filtered and hashed UNGA corpus to {UNGA_HASHED_CSV}.")

        logger.info("Loading WVS dataset...")
        wvs_df = pd.read_csv(WVS_DATASET_CSV)

        logger.info("Extracting keywords and bonus phrases (response and adapted) per combined_label...")
        keyword_dict, bonus_dict_response, bonus_dict_adapted = extract_keywords(
            wvs_df=wvs_df,
            combined_label_col='combined_label',
            adapted_col='adapted_hypotheses',
            response_col='response_hypothesis',
            top_k_keywords=TOP_K_KEYWORDS,
            top_k_bonuses_response=TOP_K_BONUSES_RESPONSE,
            top_k_bonuses_adapted=TOP_K_BONUSES_ADAPTED
        )

        logger.info("Preparing combined_label to adapted hypotheses mapping...")
        combined_labels = wvs_df['combined_label'].unique().tolist()
        combined_label_hypotheses_dict = {}
        for label in combined_labels:
            sub_df = wvs_df[wvs_df['combined_label'] == label]
            hypos = []
            for val in sub_df['adapted_hypotheses'].dropna().unique():
                if isinstance(val, str) and '[SEP]' in val:
                    parts = [p.strip() for p in val.split('[SEP]') if p.strip()]
                    hypos.extend(parts)
                else:
                    hypos.append(val.strip())
            hypos = list(set([h for h in hypos if h]))
            combined_label_hypotheses_dict[label] = hypos

        logger.info(f"Loaded {len(combined_label_hypotheses_dict)} combined_labels with hypotheses.")

        logger.info("Embedding UNGA sentences...")
        sentence_texts = unga_df['sentence_text'].tolist()
        sentence_embeddings = embed_cache(
            texts=sentence_texts,
            embedding_type="unga_speech_embeddings",
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=CACHE_DIR,
            batch_size=BATCH_SIZE
        )

        logger.info("Loading sentence transformer model for hypotheses embedding...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        logger.info("Computing composite scores...")
        scoring_results = compute_score(
            sentences=sentence_texts,
            combined_labels=combined_labels,
            sentence_embeddings=sentence_embeddings,
            combined_label_hypotheses_dict=combined_label_hypotheses_dict,
            keyword_dict=keyword_dict,
            bonus_dict_response=bonus_dict_response,
            bonus_dict_adapted=bonus_dict_adapted,
            model=model,
            batch_size=BATCH_SIZE,
            aggregation=AGGREGATION_METHOD
        )

        logger.info(f"Selecting top {TOP_N_SENTENCES} sentences per combined_label...")
        final_df = select_top(
            scoring_results=scoring_results,
            unga_metadata_df=unga_df,
            wvs_metadata_df=wvs_df,
            top_n=TOP_N_SENTENCES
        )

        logger.info(f"Exporting results to {OUTPUT_CSV}...")
        final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

        # --- Hash consistency check ---
        logger.info("Checking hash consistency between final sentences and UNGA corpus...")
        unga_hash_map = dict(zip(unga_df['sentence_text'], unga_df['embedding_hash']))
        common_sentences = set(final_df['sentence']).intersection(unga_hash_map.keys())
        mismatches = final_df[final_df['sentence'].isin(common_sentences)].copy()
        mismatches = mismatches[mismatches['sentence'].map(unga_hash_map) != mismatches['embedding_hash']] if 'embedding_hash' in mismatches.columns else pd.DataFrame()

        if mismatches.empty:
            logger.info("All hashes match for common scored sentences.")
        else:
            logger.warning(f"{len(mismatches)} hash mismatches found among common scored sentences.")
            logger.warning(mismatches[['sentence', 'embedding_hash']].head())

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
