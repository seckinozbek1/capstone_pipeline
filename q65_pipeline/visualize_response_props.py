import pandas as pd
import plotly.express as px

# Load top scored sentences CSV to get countries list
top_scored_df = pd.read_csv("../top_scored_sentences.csv")
valid_countries = set(top_scored_df['B_COUNTRY_ALPHA'].unique())

# Load WVS Wave 7 data CSV
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)  # low_memory=False to avoid dtype warnings

# Filter rows where Q65 is not missing and country is in top_scored_sentences
wvs_q65 = wvs_df[['B_COUNTRY_ALPHA', 'Q65']].dropna()
wvs_q65 = wvs_q65[wvs_q65['B_COUNTRY_ALPHA'].isin(valid_countries)]

# Convert Q65 to numeric if not already (e.g., if stored as string)
wvs_q65['Q65'] = pd.to_numeric(wvs_q65['Q65'], errors='coerce')

# Drop rows with non-numeric Q65
wvs_q65 = wvs_q65.dropna(subset=['Q65'])

# Compute per country:
agg = wvs_q65.groupby('B_COUNTRY_ALPHA').agg(
    total_responses=('Q65', 'count'),
    count_ones=('Q65', lambda x: (x == 1).sum())
).reset_index()

# Calculate proportion of Q65=1 responses
agg['proportion_ones'] = agg['count_ones'] / agg['total_responses']

# Sort countries for plotting order
agg['B_COUNTRY_ALPHA'] = pd.Categorical(
    agg['B_COUNTRY_ALPHA'],
    categories=sorted(agg['B_COUNTRY_ALPHA'].unique()),
    ordered=True
)

# Plot proportion of Q65=1 per country
fig = px.scatter(
    agg,
    x='B_COUNTRY_ALPHA',
    y='proportion_ones',
    size='total_responses',
    color='B_COUNTRY_ALPHA',
    hover_name='B_COUNTRY_ALPHA',
    labels={
        'B_COUNTRY_ALPHA': 'Country',
        'proportion_ones': 'Proportion of Q65=1',
        'total_responses': 'Total Responses'
    },
    title='Proportion of Q65 Responses Equal to 1 by Country',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg['B_COUNTRY_ALPHA'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.show()
