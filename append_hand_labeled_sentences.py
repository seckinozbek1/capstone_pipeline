### (Annotation) Preprocess the exported labels, append the exported labeled dataset to the group_WVC_stem_encoding_complete.csv file (main dataset) ###

import pandas as pd
from IPython.display import display

# Load the already long-format dataset
df = pd.read_csv("group_WVC_stem_encoding_complete_long.csv")

# Load and clean exported labels
labels = pd.read_csv("exported_labels_q152_q154.csv")
labels = labels.dropna(how='all')
labels = labels[['combined_label', 'candidate_sentence', 'match']].copy()

# Normalize 'match' column to 0/1 if needed
unique_vals = set(labels['match'].dropna().unique())
if not unique_vals.issubset({0, 1}):
    labels['match'] = labels['match'].map({'Match': 1, 'No Match': 0}).astype('Int64')

# Save cleaned labels back (optional)
labels.to_csv("exported_labels_q152_q154.csv", index=False)

# Merge labels into main dataframe by combined_label
df = df.merge(labels[['combined_label', 'candidate_sentence', 'match']], on='combined_label', how='left')

# Rename 'candidate_sentence' to 'sentence_text' for consistency
df.rename(columns={'candidate_sentence': 'sentence_text'}, inplace=True)

# Load UNGA speech corpus metadata (select only relevant columns)
cov_cols = ['sentence_text', 'doc_id', 'B_COUNTRY_ALPHA', 'A_SESSION', 'A_YEAR', 'speaker_name', 'speaker_post']
unga_meta_cols = [c for c in cov_cols if c in pd.read_csv("unga_speech_corpus.csv", nrows=1).columns]
unga = pd.read_csv("unga_speech_corpus.csv", usecols=unga_meta_cols)

# Normalize text for matching (lowercase + strip)
df['sentence_text'] = df['sentence_text'].astype(str).str.strip().str.lower()
unga['sentence_text'] = unga['sentence_text'].astype(str).str.strip().str.lower()

# Merge metadata from UNGA corpus on normalized sentence_text
df = df.merge(unga, how='left', on='sentence_text')

# Convert session and year columns to nullable integer types if present
for col in ['A_SESSION', 'A_YEAR']:
    if col in df.columns:
        df[col] = df[col].astype('Int64')

# Display first rows for verification
display(df.head())

# Save the augmented long-format dataset
df.to_csv("group_WVC_stem_encoding_complete_long.csv", index=False)
print("Saved as group_WVC_stem_encoding_complete_long.csv")
