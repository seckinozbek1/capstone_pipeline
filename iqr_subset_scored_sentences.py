import os
import pandas as pd

IQR_SAMPLES_DIR = "iqr_samples"
os.makedirs(IQR_SAMPLES_DIR, exist_ok=True)

# Load scored sentences CSV
df = pd.read_csv("top_scored_sentences.csv")

# Get unique broad_qid values
unique_qids = df['broad_qid'].unique()

for qid in unique_qids:
    # Filter rows for current broad_qid
    df_qid = df[df['broad_qid'] == qid]

    # Calculate 25th and 75th percentile (IQR) for total_score in this group
    lower_bound = df_qid['total_score'].quantile(0.25)
    upper_bound = df_qid['total_score'].quantile(0.75)

    # Filter sentences within the IQR range
    iqr_samples = df_qid[(df_qid['total_score'] >= lower_bound) & (df_qid['total_score'] <= upper_bound)]

    print(f"broad_qid: {qid} - Selected {len(iqr_samples)} uncertain samples for annotation.")

    # Save to CSV file named with the broad_qid inside the directory
    filename = os.path.join(IQR_SAMPLES_DIR, f"{qid}_iqr_samples.csv")
    iqr_samples.to_csv(filename, index=False)
