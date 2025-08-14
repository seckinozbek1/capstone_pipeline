from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
import torch
from tqdm import tqdm
import random

# Download NLTK tokenizer (silent)
nltk.download('punkt', quiet=True)

# Load SBERT model (uses GPU if available)
model = SentenceTransformer('all-roberta-large-v1', device='cuda' if torch.cuda.is_available() else 'cpu')

# Load speech corpus and hypothesis table
speeches = pd.read_csv("unga_speech_corpus.csv")
df = pd.read_csv("group_WVC_stem_encoding_complete.csv")

# Sample 500 valid speech sentences (change k for more/less)
all_sentences = speeches['sentence_text'].dropna().tolist()
random.seed(42)
all_sentences = random.sample(all_sentences, k=500)

# Encode hypothesis groups as average embeddings
group_embeddings = []
group_metadata = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding hypothesis groups"):
    hyps = [row[f'adapted_hypothesis_{j}'] for j in range(1, 6) if pd.notna(row[f'adapted_hypothesis_{j}'])]
    if hyps:
        hyp_embs = model.encode(hyps, convert_to_tensor=True)
        avg_emb = hyp_embs.mean(dim=0)  # Mean vector for the group
        group_embeddings.append(avg_emb)
        group_metadata.append((row['broad_qid'], row['likert_scale']))

# Score each sentence against all group embeddings
reranked_results = []

for sentence in tqdm(all_sentences, desc="Scoring sampled sentences"):
    sent_emb = model.encode(sentence, convert_to_tensor=True)
    best_score = -float('inf')
    best_qid = None
    best_scale = None

    for group_emb, (qid, scale) in zip(group_embeddings, group_metadata):
        score = util.cos_sim(sent_emb, group_emb).item()
        if score > best_score:
            best_score = score
            best_qid = qid
            best_scale = scale

    reranked_results.append({
        'best_matching_sentence': sentence,
        'broad_qid': best_qid,
        'likert_scale': best_scale,
        'similarity_score': best_score
    })

# Output: sorted results
adapted_speech_similarity = pd.DataFrame(reranked_results)
adapted_speech_similarity = adapted_speech_similarity.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)

from IPython.display import display
display(adapted_speech_similarity.head(10))

# Save to CSV
adapted_speech_similarity.to_csv("adapted_speech_similarity.csv", index=False)

# Optional: Clear GPU memory
torch.cuda.empty_cache()
