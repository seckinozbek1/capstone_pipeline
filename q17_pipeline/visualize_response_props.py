import pandas as pd
import plotly.express as px

# Load top scored sentences CSV to get countries list
top_scored_df = pd.read_csv("../top_scored_sentences.csv")
valid_countries = set(top_scored_df['B_COUNTRY_ALPHA'].unique())

# Load WVS Wave 7 data CSV
wvs_path = r"C:\Users\secki\OneDrive\Desktop\MY498 Capstone Under Supervision\WVS Wave 7 Full Package for EFA\F00010736-WVS_Cross-National_Wave_7_rdata_v6_0\wvs7_full_data.csv"
wvs_df = pd.read_csv(wvs_path, low_memory=False)  # low_memory=False to avoid dtype warnings

# Filter rows where Q17 is not missing and country is in top_scored_sentences
wvs_q17 = wvs_df[['B_COUNTRY_ALPHA', 'Q17']].dropna()
wvs_q17 = wvs_q17[wvs_q17['B_COUNTRY_ALPHA'].isin(valid_countries)]

# Convert Q17 to numeric if not already (e.g., if stored as string)
wvs_q17['Q17'] = pd.to_numeric(wvs_q17['Q17'], errors='coerce')

# Drop rows with non-numeric Q17
wvs_q17 = wvs_q17.dropna(subset=['Q17'])

# Compute per country:
agg = wvs_q17.groupby('B_COUNTRY_ALPHA').agg(
    total_responses=('Q17', 'count'),
    count_ones=('Q17', lambda x: (x == 1).sum())
).reset_index()

# Calculate proportion of Q17=1 responses
agg['proportion_ones'] = agg['count_ones'] / agg['total_responses']

# Sort countries for plotting order
agg['B_COUNTRY_ALPHA'] = pd.Categorical(
    agg['B_COUNTRY_ALPHA'],
    categories=sorted(agg['B_COUNTRY_ALPHA'].unique()),
    ordered=True
)

# Plot proportion of Q17=1 per country
fig = px.scatter(
    agg,
    x='B_COUNTRY_ALPHA',
    y='proportion_ones',
    size='total_responses',
    color='B_COUNTRY_ALPHA',
    hover_name='B_COUNTRY_ALPHA',
    labels={
        'B_COUNTRY_ALPHA': 'Country',
        'proportion_ones': 'Proportion of Q17=1',
        'total_responses': 'Total Responses'
    },
    title='Proportion of Q17 Responses Equal to 1 by Country',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg['B_COUNTRY_ALPHA'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.show()
