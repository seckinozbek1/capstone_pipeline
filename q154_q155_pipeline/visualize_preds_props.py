import pandas as pd
import plotly.express as px

df = pd.read_csv("predictions/q154_predictions_filtered.csv")

q154_labels = ['Q154_1', 'Q154_2', 'Q154_3', 'Q154_4']
df_q154 = df[df['predicted_combined_label'].isin(q154_labels)]

# Count rows per country and label (sentence counts)
agg = df_q154.groupby(['B_COUNTRY_ALPHA', 'predicted_combined_label']).size().reset_index(name='sentence_count')

# Total sentences per country (all Q154)
total_counts = agg.groupby('B_COUNTRY_ALPHA')['sentence_count'].sum().reset_index(name='total_q154_count')

agg = agg.merge(total_counts, on='B_COUNTRY_ALPHA')

agg['proportion'] = agg['sentence_count'] / agg['total_q154_count']

# Create explicit copy to avoid SettingWithCopyWarning
agg_q154_1 = agg[agg['predicted_combined_label'] == 'Q154_1'].copy()

agg_q154_1['B_COUNTRY_ALPHA'] = pd.Categorical(
    agg_q154_1['B_COUNTRY_ALPHA'],
    categories=sorted(agg_q154_1['B_COUNTRY_ALPHA'].unique()),
    ordered=True
)

fig = px.scatter(
    agg_q154_1,
    x='B_COUNTRY_ALPHA',
    y='proportion',
    size='sentence_count',
    color='B_COUNTRY_ALPHA',
    hover_name='B_COUNTRY_ALPHA',
    labels={'B_COUNTRY_ALPHA': 'Country', 'proportion': 'Proportion of Q154_1', 'sentence_count': 'Sentence Count'},
    title='Proportion of Q154_1 Sentences Relative to Total Q154 Sentences by Country',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg_q154_1['B_COUNTRY_ALPHA'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.show()
