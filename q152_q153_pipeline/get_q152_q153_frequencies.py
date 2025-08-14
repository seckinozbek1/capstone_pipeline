import pandas as pd
from IPython.display import display
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load predictions CSV
df = pd.read_csv("predictions/q152_predictions_filtered.csv")

# Group by country-year and combined_label, count occurrences
grouped = df.groupby(['B_COUNTRY_ALPHA', 'A_YEAR', 'predicted_combined_label']).size().reset_index(name='count')

# Sort by country-year and descending count
grouped = grouped.sort_values(['B_COUNTRY_ALPHA', 'A_YEAR', 'count'], ascending=[True, True, False])

# Helper function to get top 2 labels per country-year
def top_two_labels(sub_df):
    top_labels = sub_df.head(2)
    labels = top_labels['predicted_combined_label'].tolist()
    counts = top_labels['count'].tolist()
    while len(labels) < 2:
        labels.append(None)
        counts.append(0)
    has_second_label = labels[1] is not None
    return pd.Series({
        'most_frequent_label': labels[0],
        'most_frequent_count': counts[0],
        'second_most_frequent_label': labels[1],
        'second_most_frequent_count': counts[1],
        'has_second_label': has_second_label
    })

# Apply helper per country-year
top_labels_df = grouped.groupby(['B_COUNTRY_ALPHA', 'A_YEAR']).apply(top_two_labels).reset_index()

# Load WVS full data
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Get unique country-year pairs from WVS
wvs_country_years = set(zip(wvs_df['B_COUNTRY_ALPHA'], wvs_df['A_YEAR']))

# Filter top_labels_df to keep only country-year pairs that exist in WVS data
top_labels_df = top_labels_df[
    top_labels_df.apply(lambda row: (row['B_COUNTRY_ALPHA'], row['A_YEAR']) in wvs_country_years, axis=1)
].reset_index(drop=True)

# Separate country-years with and without second most frequent label
without_second_label = top_labels_df[~top_labels_df['has_second_label']]

# Print the list of country-year pairs without second most frequent label
print("\nCountry-year pairs without a second most frequent label:")
for idx, row in without_second_label.iterrows():
    print(f"{row['B_COUNTRY_ALPHA']} - {row['A_YEAR']}")

# Print the full dataset
display(top_labels_df)
# Save the results
top_labels_df.to_csv("q152_country_year_top2_labels.csv", index=False)