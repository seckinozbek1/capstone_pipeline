import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Load scored sentences CSV ---
top_scored_df = pd.read_csv("predictions/q69_predictions_filtered.csv")

q69_label = 'Q69_1'  # Label to analyze

# Filter to rows matching the Q69 label
df_q69_label = top_scored_df[top_scored_df['predicted_combined_label'] == q69_label]

# Count sentences per country-year for Q69_label
agg_scored = df_q69_label.groupby(['B_COUNTRY_ALPHA', 'A_YEAR']).agg(
    sentence_count=('predicted_combined_label', 'count')
).reset_index()

# Count total Q69 sentences per country-year (any Q69_*)
total_q69_per_country_year = top_scored_df[
    top_scored_df['predicted_combined_label'].str.startswith('Q69_')
].groupby(['B_COUNTRY_ALPHA', 'A_YEAR']).size().reset_index(name='total_q69_count')

# Merge counts and compute proportion scored per country-year
agg_scored = agg_scored.merge(total_q69_per_country_year, on=['B_COUNTRY_ALPHA', 'A_YEAR'])
agg_scored['proportion_scored'] = agg_scored['sentence_count'] / agg_scored['total_q69_count']

# --- Load WVS Wave 7 data ---
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)

# Filter WVS Q69 responses by valid country-year pairs in scored data
valid_country_years = set(zip(agg_scored['B_COUNTRY_ALPHA'], agg_scored['A_YEAR']))

wvs_q69 = wvs_df[['B_COUNTRY_ALPHA', 'A_YEAR', 'Q69', 'S018']].dropna(subset=['B_COUNTRY_ALPHA', 'A_YEAR', 'Q69', 'S018'])
wvs_q69 = wvs_q69[wvs_q69.apply(lambda row: (row['B_COUNTRY_ALPHA'], row['A_YEAR']) in valid_country_years, axis=1)]
wvs_q69['Q69'] = pd.to_numeric(wvs_q69['Q69'], errors='coerce')
wvs_q69 = wvs_q69.dropna(subset=['Q69'])

# Compute weighted proportion per country-year for Q69 = target label
target_label = int(q69_label[-1])

def weighted_proportion(group):
    weights = group['S018']
    label_mask = (group['Q69'] == target_label).astype(int)
    weighted_count = np.sum(weights * label_mask)
    total_weight = np.sum(weights)
    return weighted_count / total_weight if total_weight > 0 else np.nan

agg_wvs = wvs_q69.groupby(['B_COUNTRY_ALPHA', 'A_YEAR']).apply(
    lambda g: pd.Series({
        'weighted_proportion_wvs': weighted_proportion(g),
        'total_weight': g['S018'].sum()
    })
).reset_index()

# --- Merge scored and WVS weighted data on country-year ---
merged = pd.merge(
    agg_scored[['B_COUNTRY_ALPHA', 'A_YEAR', 'proportion_scored']],
    agg_wvs[['B_COUNTRY_ALPHA', 'A_YEAR', 'weighted_proportion_wvs']],
    on=['B_COUNTRY_ALPHA', 'A_YEAR']
)

merged = merged.sort_values(['B_COUNTRY_ALPHA', 'A_YEAR']).reset_index(drop=True)
labels = merged.apply(lambda row: f"{row['B_COUNTRY_ALPHA']}_{row['A_YEAR']}", axis=1)

fig = go.Figure()

# Add vertical lines connecting WVS and scored proportions per country-year
for _, row in merged.iterrows():
    fig.add_trace(go.Scatter(
        x=[f"{row['B_COUNTRY_ALPHA']}_{row['A_YEAR']}", f"{row['B_COUNTRY_ALPHA']}_{row['A_YEAR']}"],
        y=[row['weighted_proportion_wvs'], row['proportion_scored']],
        mode='lines',
        line=dict(color='gray'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Points for WVS weighted proportions
fig.add_trace(go.Scatter(
    x=labels,
    y=merged['weighted_proportion_wvs'],
    mode='markers',
    name=f'WVS Weighted {q69_label}',
    marker=dict(color='blue', size=8),
    hovertemplate='%{x}<br>WVS Weighted Proportion: %{y:.2f}<extra></extra>'
))

# Points for scored sentences
fig.add_trace(go.Scatter(
    x=labels,
    y=merged['proportion_scored'],
    mode='markers',
    name=f'Scored Sentences {q69_label}',
    marker=dict(color='red', size=8),
    hovertemplate='%{x}<br>Sentence Proportion: %{y:.2f}<extra></extra>'
))

fig.update_layout(
    title=f'WVS Weighted vs Scored Sentences Proportion of {q69_label} by Country-Year',
    xaxis_title='Country-Year',
    yaxis_title='Proportion',
    yaxis=dict(range=[0, 1.1]),
    height=600,
    hovermode='x unified'
)

fig.show()

# Calculate Pearson correlation between WVS weighted proportions and scored sentence proportions
correlation = merged['weighted_proportion_wvs'].corr(merged['proportion_scored'])
print(f"Pearson correlation between WVS weighted and scored sentence proportions: {correlation:.4f}")
