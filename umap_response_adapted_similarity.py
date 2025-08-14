import pandas as pd
import numpy as np
import umap.umap_ as umap
import plotly.express as px
from sentence_transformers import SentenceTransformer
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------
# Load the full dataset
# ------------------------------
df = pd.read_csv("group_WVC_stem_encoding_complete.csv")

# ------------------------------
# Load SBERT model (GPU if available)
# ------------------------------
model = SentenceTransformer('all-roberta-large-v1', device='cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Encode response hypotheses
# ------------------------------
response_rows = df[df['response_hypothesis'].notna()]
response_texts = response_rows['response_hypothesis'].tolist()
response_qids = response_rows['broad_qid'].tolist()
response_scales = response_rows['likert_scale'].tolist()

print("Encoding response hypotheses...")
response_embs = []
for text in tqdm(response_texts, desc="Response encoding"):
    emb = model.encode(text, convert_to_numpy=True)
    response_embs.append(emb)
response_embs = np.vstack(response_embs)

# ------------------------------
# Encode adapted hypothesis bags (mean of 5)
# ------------------------------
adapted_embs = []
adapted_qids = []
adapted_scales = []

print("Encoding adapted hypothesis bags...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Adapted encoding"):
    hyps = [row[f'adapted_hypothesis_{j}'] for j in range(1, 6) if pd.notna(row[f'adapted_hypothesis_{j}'])]
    if hyps:
        emb = model.encode(hyps, convert_to_numpy=True).mean(axis=0)
        adapted_embs.append(emb)
        adapted_qids.append(row['broad_qid'])
        adapted_scales.append(row['likert_scale'])
adapted_embs = np.vstack(adapted_embs)

# ------------------------------
# Combine and run UMAP
# ------------------------------
print("Scaling and running UMAP projection...")
combined_embs = np.vstack([response_embs, adapted_embs])
scaled = StandardScaler().fit_transform(combined_embs)

reducer = umap.UMAP(n_components=2, random_state=42)
proj_2d = reducer.fit_transform(scaled)

# ------------------------------
# Split and organize UMAP outputs
# ------------------------------
n_response = len(response_embs)
response_2d = proj_2d[:n_response]
adapted_2d = proj_2d[n_response:]

# ------------------------------
# Create plot DataFrame
# ------------------------------
df_plot = pd.DataFrame({
    'Dim1': np.concatenate([response_2d[:, 0], adapted_2d[:, 0]]),
    'Dim2': np.concatenate([response_2d[:, 1], adapted_2d[:, 1]]),
    'QID': response_qids + adapted_qids,
    'Likert': response_scales + adapted_scales,
    'Type': ['Response'] * n_response + ['Adapted'] * len(adapted_embs)
})

# ------------------------------
# Plot
# ------------------------------
fig = px.scatter(
    df_plot,
    x='Dim1',
    y='Dim2',
    color='QID',
    symbol='Type',
    hover_data={'QID': True, 'Likert': True, 'Type': True},
    title="UMAP of Response vs Adapted Hypotheses"
)

fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color='black')))
fig.update_layout(legend_title_text='Question Groups', height=750)
fig.show()
