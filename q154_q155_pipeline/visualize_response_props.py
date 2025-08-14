import pandas as pd
import plotly.express as px

# Load top scored sentences CSV to get countries list
top_scored_df = pd.read_csv("../top_scored_sentences.csv")
valid_countries = set(top_scored_df['B_COUNTRY_ALPHA'].unique())

# Load WVS Wave 7 data CSV
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)  # low_memory=False to avoid dtype warnings

# Filter rows where Q154 is not missing and country is in top_scored_sentences
wvs_q154 = wvs_df[['B_COUNTRY_ALPHA', 'Q154']].dropna()
wvs_q154 = wvs_q154[wvs_q154['B_COUNTRY_ALPHA'].isin(valid_countries)]

# Convert Q154 to numeric if not already (e.g., if stored as string)
wvs_q154['Q154'] = pd.to_numeric(wvs_q154['Q154'], errors='coerce')

# Drop rows with non-numeric Q154
wvs_q154 = wvs_q154.dropna(subset=['Q154'])

# Compute per country:
agg = wvs_q154.groupby('B_COUNTRY_ALPHA').agg(
    total_responses=('Q154', 'count'),
    count_ones=('Q154', lambda x: (x == 1).sum())
).reset_index()

# Calculate proportion of Q154=1 responses
agg['proportion_ones'] = agg['count_ones'] / agg['total_responses']

# Sort countries for plotting order
agg['B_COUNTRY_ALPHA'] = pd.Categorical(
    agg['B_COUNTRY_ALPHA'],
    categories=sorted(agg['B_COUNTRY_ALPHA'].unique()),
    ordered=True
)

# Plot proportion of Q154=1 per country
fig = px.scatter(
    agg,
    x='B_COUNTRY_ALPHA',
    y='proportion_ones',
    size='total_responses',
    color='B_COUNTRY_ALPHA',
    hover_name='B_COUNTRY_ALPHA',
    labels={
        'B_COUNTRY_ALPHA': 'Country',
        'proportion_ones': 'Proportion of Q154=1',
        'total_responses': 'Total Responses'
    },
    title='Proportion of Q154 Responses Equal to 1 by Country',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg['B_COUNTRY_ALPHA'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.show()
