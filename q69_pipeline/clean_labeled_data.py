import pandas as pd

print("Loading Q69_mmr_selected_labeled.csv ...")
labeled_df = pd.read_csv("Q69_mmr_selected_labeled.csv")

print("Loading ../top_scored_sentences.csv ...")
top_scored_df = pd.read_csv("../top_scored_sentences.csv")

# Filter labeled_df to only keep rows where match == 1
labeled_df = labeled_df[labeled_df['match'] == 1]

# Select only necessary columns from top_scored to avoid duplicates
top_scored_subset = top_scored_df[['combined_label', 'response_hypothesis', 'adapted_hypotheses']].drop_duplicates(subset=['combined_label'])

# Merge on combined_label, include 'sentence', 'match', and 'embedding_hash' from labeled_df
merged_df = pd.merge(
    labeled_df[['combined_label', 'sentence', 'match', 'embedding_hash']],
    top_scored_subset,
    on='combined_label',
    how='left',
    validate='many_to_one'  # many labeled rows per unique combined_label in top_scored
)

# Replace [SEP] with | in adapted_hypotheses
merged_df['adapted_hypotheses'] = merged_df['adapted_hypotheses'].fillna('').astype(str).str.replace(r'\[SEP\]', ' | ', regex=True).str.strip()

# Combine sentence, response_hypothesis, and adapted_hypotheses with ' | ' separator
def combine_text(row):
    parts = [str(row['sentence']).strip()]
    if pd.notna(row['response_hypothesis']) and row['response_hypothesis'].strip() != '':
        parts.append(row['response_hypothesis'].strip())
    if row['adapted_hypotheses'] != '':
        parts.append(row['adapted_hypotheses'])
    return ' | '.join(parts)

print("Creating combined sentence column...")
merged_df['sentence_combined'] = merged_df.apply(combine_text, axis=1)

# Keep only the columns you need including embedding_hash
final_df = merged_df[['combined_label', 'sentence', 'match', 'embedding_hash', 'response_hypothesis', 'adapted_hypotheses', 'sentence_combined']].copy()

print("Saving to Q69_mmr_selected_labeled_combined.csv ...")
final_df.to_csv("Q69_mmr_selected_labeled_combined.csv", index=False)
print("Done.")
