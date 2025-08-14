import pandas as pd
from IPython.display import display
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load predictions CSV (labels + embedding_hash)
predictions_path = "predictions/q69_predictions_filtered.csv"
df_pred = pd.read_csv(predictions_path)

# Convert A_YEAR to int for consistency
df_pred['A_YEAR'] = df_pred['A_YEAR'].astype(int)

# Now group by country-year and predicted label, count occurrences
grouped = (
    df_pred.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label'])
             .size()
             .reset_index(name='count')
)

# Sort by country-year and descending count
grouped = grouped.sort_values(
    ['B_COUNTRY_ALPHA', 'A_YEAR', 'count'],
    ascending=[True, True, False]
)

# Aggregate to get the most frequent label and its count per country-year
top_labels_df = grouped.groupby(['B_COUNTRY_ALPHA', 'A_YEAR']).agg(
    most_frequent_label=('predicted_combined_label', 'first'),
    most_frequent_count=('count', 'first')
).reset_index()

# Load WVS full data
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Ensure WVS year column is int
wvs_df['A_YEAR'] = wvs_df['A_YEAR'].astype(int)

# Create set of valid country-year pairs from WVS data
wvs_country_years = set(zip(wvs_df['B_COUNTRY_ALPHA'], wvs_df['A_YEAR']))

# Filter top_labels_df to keep only pairs present in WVS data
top_labels_df = top_labels_df[
    top_labels_df.apply(lambda row: (row['B_COUNTRY_ALPHA'], row['A_YEAR']) in wvs_country_years, axis=1)
].reset_index(drop=True)

# Display final results
display(top_labels_df)

# Save to CSV
top_labels_df.to_csv("q69_country_year_top_labels.csv", index=False)
