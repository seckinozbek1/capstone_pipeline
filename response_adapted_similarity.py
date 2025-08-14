import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# Define expected model cache paths
from transformers.utils import default_cache_path

sbert_path = os.path.join(default_cache_path, 'sentence-transformers', 'all-roberta-large-v1')
cross_path = os.path.join(default_cache_path, 'cross-encoder', 'stsb-roberta-large')

# Load SBERT if not already downloaded
if not os.path.exists(sbert_path):
    print("Downloading SentenceTransformer model to GPU...")
    sbert_model = SentenceTransformer('all-roberta-large-v1', device='cuda')
else:
    print("SentenceTransformer model already cached.")
    sbert_model = SentenceTransformer('all-roberta-large-v1', device='cuda')

# Load CrossEncoder if not already downloaded
if not os.path.exists(cross_path):
    print("Downloading CrossEncoder model to GPU...")
    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device='cuda')
else:
    print("CrossEncoder model already cached.")
    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large', device='cuda')


# Clear the CUDA cache to free up memory for safe execution of subsequent operations.
import torch
torch.cuda.empty_cache()


### Compare cosine similarities between response hypotheses and bagged adapted hypothesis embeddings ###


from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
import torch
from tqdm import tqdm
from IPython.display import display, HTML

# ---------------------------------------------------------
# Download NLTK tokenizer (silent)
# ---------------------------------------------------------
nltk.download('punkt', quiet=True)

# ---------------------------------------------------------
# Load SBERT model (uses GPU if available)
# ---------------------------------------------------------
model = SentenceTransformer('all-roberta-large-v1', device='cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# Load hypothesis table (includes group and response columns)
# ---------------------------------------------------------
df = pd.read_csv("group_WVC_stem_encoding_complete.csv")

# ---------------------------------------------------------
# Encode 5-adapted hypothesis bags into averaged embeddings
# Each row yields: (qid, scale, avg_embedding)
# ---------------------------------------------------------
group_embeddings = []
group_metadata = []

for _, row in df.iterrows():
    hyps = [row[f'adapted_hypothesis_{j}'] for j in range(1, 6) if pd.notna(row[f'adapted_hypothesis_{j}'])]
    if hyps:
        hyp_embs = model.encode(hyps, convert_to_tensor=True)
        avg_emb = hyp_embs.mean(dim=0)
        group_embeddings.append(avg_emb)
        group_metadata.append((row['broad_qid'], row['likert_scale']))

# ---------------------------------------------------------
# Compare response_hypothesis column to bagged embeddings
# Each row matched to best-fit group (based on cosine score)
# ---------------------------------------------------------
reranked_results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring response_hypotheses"):
    response = row.get("response_hypothesis")
    if pd.isna(response):
        continue

    response_emb = model.encode(response, convert_to_tensor=True)

    best_score = -float('inf')
    best_qid = None
    best_scale = None

    for group_emb, (qid, scale) in zip(group_embeddings, group_metadata):
        score = util.cos_sim(response_emb, group_emb).item()
        if score > best_score:
            best_score = score
            best_qid = qid
            best_scale = scale

    reranked_results.append({
        'response_hypothesis': response,
        'true_qid': row['broad_qid'],
        'true_scale': row['likert_scale'],
        'matched_qid': best_qid,
        'matched_scale': best_scale,
        'similarity_score': best_score
    })

# ---------------------------------------------------------
# Output results: dataframe, HTML table, CSV export
# ---------------------------------------------------------
response_adapted_similarity = pd.DataFrame(reranked_results)
response_adapted_similarity = response_adapted_similarity.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
response_adapted_similarity['similarity_score'] = response_adapted_similarity['similarity_score'].round(4)

# Display top 10 results in notebook
display(response_adapted_similarity.head(10))

# Save all results to CSV
response_adapted_similarity.to_csv("response_adapted_similarity.csv", index=False)

# ---------------------------------------------------------
# Clear GPU memory
# ---------------------------------------------------------
torch.cuda.empty_cache()
